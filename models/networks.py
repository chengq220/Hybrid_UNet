import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models.layers.layer import recConvBlock,MeshDownConv,MeshUpConv
from models.layers.mesh import Mesh
import numpy as np
from models.loss import DiceLoss,DiceBCELoss
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import SplineConv
from utils.util import pad,unpad
import wandb
import time


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_net(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    return net

def define_classifier(ncf, classes, opt, gpu_ids, arch):
    net = None
    #Regular unet
    if arch == 'unet':
        down = opt.ncf
        bottleneck = down[-1]
        up = down[len(down)-2::-1]
        net = Unet(down,bottleneck,up,classes)
    #Hybrid-UNET    
    elif arch == 'hybrid':
        ncf = opt.ncf

        rec_down = ncf[:3]
        rec_up = rec_down[::-1]

        mesh_down = ncf[3:]
        mesh_up = mesh_down[::-1]
        
        net = HybridUnet(classes,rec_down,rec_up,mesh_down,mesh_up)
    
    elif arch == 'test':
        net = TestNet()
    
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, gpu_ids)

def define_loss(opt):
    if opt.loss_func == "bce":
        loss =  BCEWithLogitsLoss()
    elif opt.loss_func == "dice":
        loss = DiceLoss()
    elif opt.loss_func == "dice_bce":
        loss = DiceBCELoss()
    else:
        raise NotImplementedError('Loss function is not recognized' %loss_func)
    return loss

##############################################################################
# Segmentation Network
##############################################################################
class Unet(nn.Module):
    """UNET implementation 
    """
    def __init__(self, down_convs, bottleneck, up_convs, output):
        super(Unet,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down_convs = down_convs[:len(down_convs)-1]

        ## Enocder
        #===========================================================
        in_channels = 3
        self.down = nn.ModuleList()
        for out_channels in self.down_convs:
            self.down.append(recConvBlock(in_channels, out_channels,3))
            in_channels = out_channels
        
        ##bottleneck
        #===========================================================
        self.bottleneck = recConvBlock(self.down_convs[-1],bottleneck,3)

        ## Decoder
        #===========================================================
        in_channel = bottleneck
        self.unpool = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for out_channels in up_convs:
            self.unpool.append(nn.ConvTranspose2d(in_channel,out_channels,kernel_size=2,stride=2))
            self.up_conv.append(recConvBlock(in_channel,out_channels,3))
            in_channel = out_channels

        self.output = nn.Conv2d(up_convs[-1],output,kernel_size = 1, padding="same")

    def forward(self,x):
        fe = x
        skips = []
        for down_conv in self.down:
            fe = down_conv(fe)
            skips.append(fe)
            fe = self.maxpool(fe)
        
        fe = self.bottleneck(fe)

        skips = skips[::-1]
        for i in range(len(skips)):
            fe = self.unpool[i](fe)
            fe = torch.cat((fe,skips[i]),1)
            fe = self.up_conv[i](fe)
        
        fe = self.output(fe).squeeze(1)
        return fe

    def __call__self(self,x):
        return self.forward(x)


class HybridUnet(nn.Module):
    def __init__(self, output,rec_down_convs,rec_up_convs,down_convs,up_convs):
        super(HybridUnet, self).__init__()
        self.rec_down_channel = rec_down_convs
        self.rec_up_channel = rec_up_convs
        
        #UNET encoder
        #===========================================================
        self.rec_down = nn.ModuleList()
        in_channel = 3
        for out in rec_down_convs:
            down = recConvBlock(in_channel,out,3)
            self.rec_down.append(down)
            in_channel = out
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Mesh parameters
        #===========================================================
        self.down_convs = [rec_down_convs[-1]] + down_convs 
        self.up_convs = up_convs

        # UNet Decoder
        #===========================================================
        self.unpool = nn.ModuleList()
        self.conv = nn.ModuleList()
        in_channel = self.up_convs[-1]
        for out_channel in rec_up_convs:
            up = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)
            self.unpool.append(up)
            self.conv.append(recConvBlock(in_channel,out_channel,3))
            in_channel = out_channel

        self.output = nn.Conv2d(rec_up_convs[-1],output,kernel_size=1,padding="same")

    def forward(self,x):
        # Regular UNET downsampling
        #===========================================================
        fe = x
        skips = []
        for conv in self.rec_down:
            fe = conv(fe)
            skips.append(fe)
            fe = self.maxpool(fe)

        #Mesh Pre-processing
        #===========================================================
        meshes = []
        for image in fe:
            mesh = Mesh(file=image)
            meshes.append(mesh)
        meshes = np.array(meshes)
        
        #Mesh Encoder/Decoder
        #===========================================================
        encoder = MeshEncoder(self.down_convs)
        mesh_before_pool = encoder(meshes)
        decoder = MeshDecoder(self.up_convs)
        decoder(meshes, mesh_before_pool[:-1])

        #Mesh Post Processing
        #===========================================================
        out = []
        for mesh in meshes:
            out.append(mesh.image)
        out = torch.transpose(torch.stack(out),2,1)
        out = out.reshape(out.shape[0],out.shape[1],fe.shape[2]+2,fe.shape[3]+2)
        fe = unpad(out)

        ##################################################################
        # Regular Unet UpSampling
        skips = skips[::-1]
        for i in range(len(self.unpool)):
            fe = self.unpool[i](fe)
            fe = torch.cat((fe, skips[i]),1)
            fe = self.conv[i](fe)
        fe = self.output(fe).squeeze(1)
        return fe

    def __call__self(self,x):
        return self.forward(x)

class MeshEncoder(nn.Module):
    def __init__(self, convs):
        super(MeshEncoder, self).__init__()
        in_channel = convs[0]
        self.down = nn.ModuleList()
        for idx, out_channel in enumerate(convs[1:]):
            if idx > len(convs[1:])-2:
                down = MeshDownConv(in_channel,convs[-1],False)
            else:
                down = MeshDownConv(in_channel,out_channel,True)
            self.down.append(down)
            in_channel = out_channel

    def forward(self, meshes):
        before_pool = []
        for down in self.down:
            before_pool.append(down(meshes))
        return before_pool

    def __call__(self, meshes):
        return self.forward(meshes)

class MeshDecoder(nn.Module):
    def __init__(self, up_convs):
        super(MeshDecoder, self).__init__()
        self.up = nn.ModuleList()
        in_channel = up_convs[0]
        up_convs = up_convs[1:]
        for idx,out_channel in enumerate(up_convs):
            up = MeshUpConv(in_channel, out_channel)
            self.up.append(up)
            in_channel = out_channel

    def forward(self, meshes, encoder_outs):
        encoder_outs = encoder_outs[::-1]
        for idx, up in enumerate(self.up):
            meshes = up(meshes,encoder_outs[idx])

    def __call__(self, x, encoder_outs):
        return self.forward(x, encoder_outs)

class TestNet(nn.Module):
    """UNET implementation 
    """
    def __init__(self):
        super(TestNet,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down1 = recConvBlock(3,64,3)
        self.down2 = recConvBlock(64,128,3)
        self.down3 = recConvBlock(128,256,3)
        self.down4 = recConvBlock(256,512,3)
        self.bn = recConvBlock(512,1024,3)
        self.unpool1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up1 = recConvBlock(1024,512,3)
        self.unpool2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.up2 = recConvBlock(512,256,3)
        self.unpool3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.up3 = recConvBlock(256,128,3)
        self.unpool4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up4 = recConvBlock(128,64,3)
        self.output = nn.Conv2d(64,1,kernel_size=3,padding="same")

        self.meshDown1 = MeshDownConv(128,256,True)
        self.meshDown2 = MeshDownConv(256,512,True)
        self.meshBn = MeshDownConv(512,1024,False)
        self.meshUp1 = MeshUpConv(1024,512)
        self.meshUp2 = MeshUpConv(512,256)
        # self.conv1 = SplineConv(1024, 512,dim=2,kernel_size=[3,3],degree=2,aggr='add').cuda()

    def forward(self,x):
        fe = x
        b_pool1 = self.down1(fe)
        fe = self.maxpool(b_pool1)
        b_pool2 = self.down2(fe)
        fe = self.maxpool(b_pool2)

        meshes = []
        adjs = []
        images = pad(fe)

        for image in images:
            mesh = Mesh([images.shape[2],images.shape[3]])
            adjs.append(mesh.get_adjacency())
            meshes.append(mesh)
        meshes = np.array(meshes)
        adjs = torch.stack(adjs)
        images = torch.transpose(images.reshape(images.shape[0],images.shape[1],images.shape[2]*images.shape[3]),2,1)
        
        before_pool1 = self.meshDown1(meshes)
        before_pool2 = self.meshDown2(meshes)

        self.meshBn(meshes)

        meshes = self.meshUp1(meshes,before_pool2)
        meshes = self.meshUp2(meshes,before_pool1)
    
        fe = []
        for mesh in meshes:
            fe.append(mesh.image)
        fe = torch.transpose(torch.stack(fe),2,1)
        fe = fe.reshape(1,256,66,66)
        fe = unpad(fe)

        unpool3 = self.unpool3(fe)
        fe = torch.cat((b_pool2,unpool3),1)
        fe = self.up3(fe)

        unpool4 = self.unpool4(fe)
        fe = torch.cat((b_pool1, unpool4),1)
        fe = self.up4(fe)

        out = self.output(fe).squeeze(1)
        print(out.shape)
        exit()
        return out

    def __call__self(self,x):
        return self.forward(x)