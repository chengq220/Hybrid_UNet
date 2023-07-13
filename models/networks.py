import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
from models.layers.mesh import Mesh
import numpy as np
from models.loss import DiceLoss,DiceBCELoss
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import SplineConv
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
    elif arch == 'hybrid':
        ncf = opt.ncf
        rec_down = ncf[:2]
        rec_up = rec_down[::-1]

        mesh_down = ncf[2:]
        mesh_up = mesh_down[::-1] 

        net = HybridUnet(classes,rec_down,rec_up,mesh_down,mesh_up)
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
    def __init__(self, down_convs, bottleneck, up_convs,output):
        super(Unet,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down_convs = down_convs[:len(down_convs)-1]

        ## Enocder
        in_channels = 3
        self.down = nn.ModuleList()
        for out_channels in self.down_convs:
            self.down.append(rectangular_conv_block(in_channels, out_channels,3))
            in_channels = out_channels

        ##bottleneck
        self.bottleneck = rectangular_conv_block(self.down_convs[-1],bottleneck,3)

        ## Decoder
        in_channel = bottleneck
        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for out_channels in up_convs:
            self.up.append(nn.ConvTranspose2d(in_channel,out_channels,kernel_size=2,stride=2))
            self.up_conv.append(rectangular_conv_block(in_channel,out_channels,3))
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
            fe = self.up[i](fe)
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
        #the regular UNET encoder
        #=================================================================
        self.rec_down = nn.ModuleList()
        in_channel = 3
        for out in rec_down_convs:
            down = rectangular_conv_block(in_channel,out,3)
            self.rec_down.append(down)
            in_channel = out
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #=================================================================
        #the mesh unet [params]
        #=================================================================
        self.down_convs = [rec_down_convs[-1]] + down_convs 
        self.up_convs = up_convs

        #===========================================================
        # Regular UNet Decoder
        #===========================================================
        self.rec_up_conv = nn.ModuleList()
        self.conv = nn.ModuleList()
        in_channel = self.up_convs[-1]
        for out_channel in rec_up_convs:
            up = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)
            self.rec_up_conv.append(up)
            self.conv.append(rectangular_conv_block(in_channel,out_channel,3))
            in_channel = out_channel

        ##output
        self.output = nn.Conv2d(rec_up_convs[-1],output,kernel_size=1,padding="same")

    def forward(self,x):
        start_time = time.time()
        #################################################
        # Regular UNET downsampling
        fe = x
        skips = []
        for conv in self.rec_down:
            fe = conv(fe)
            skips.append(fe)
            fe = self.maxpool(fe)

        image_size = fe.shape[2]*fe.shape[3]
        ##################################################
        #Mesh Unet with spline conv
        meshes = []
        #create mesh for the batch
        for image in fe:
            mesh = Mesh(file=image)
            meshes.append(mesh)
        meshes = np.array(meshes)
        
        encoder = MeshEncoder(self.down_convs)
        mesh_before_pool = encoder(meshes)
        decoder = MeshDecoder(self.up_convs)
        decoder(meshes, mesh_before_pool[:-1])

        out = []
        for mesh in meshes:
            out.append(mesh.get_feature())
        out = torch.transpose(torch.stack(out),2,1)
        fe = out.reshape(out.shape[0],out.shape[1],fe.shape[2],fe.shape[3])

        # ##################################################################
        # # Regular Unet UpSampling
        skips = skips[::-1]
        for i in range(len(self.rec_up_conv)):
            fe = self.rec_up_conv[i](fe)
            fe = torch.cat((fe, skips[i]),1)
            fe = self.conv[i](fe)
        fe = self.output(fe).squeeze(1)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Network Elapsed time:", elapsed_time, " seconds")
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
                down = DownConv(in_channel,convs[-1],False)
            else:
                down = DownConv(in_channel,out_channel,True)
            self.down.append(down)
            in_channel = out_channel

    def forward(self, meshes):
        before_pool = []
        for down in self.down:
            before_pool.append(down(meshes))
        return before_pool

    def __call__(self, meshes):
        return self.forward(meshes)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pool):
        super(DownConv, self).__init__()
        self.conv1 = SplineConv(in_channels,out_channels,dim=2, kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.conv2 = SplineConv(out_channels, out_channels, dim=2 ,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.pool = None
        if(pool):
            self.pool = MeshPool()

    def __call__(self, meshes):
        return self.forward(meshes)

    def forward(self, meshes):
        before_pool = []
        #Spline Convolution
        for idx,mesh in enumerate(meshes):     
            v_f = mesh.get_feature()
            edges = mesh.get_undirected_edges()
            edge_attribute = mesh.get_attributes(edges).cuda()
            v_f = self.conv1(v_f,edges.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = self.conv2(v_f,edges.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            if self.pool is not None:
                before_pool.append(v_f)
                mesh.update_dictionary(edges,"edge")
            mesh.update_feature(v_f)

        if self.pool is not None:
            meshes = self.pool(meshes)
            before_pool = torch.stack(before_pool)
            return before_pool


class MeshDecoder(nn.Module):
    def __init__(self, up_convs):
        super(MeshDecoder, self).__init__()
        self.up = nn.ModuleList()
        in_channel = up_convs[0]
        up_convs = up_convs[1:]
        for idx,out_channel in enumerate(up_convs):
            up = UpConv(in_channel, out_channel)
            self.up.append(up)
            in_channel = out_channel

    def forward(self, meshes, encoder_outs):
        encoder_outs = encoder_outs[::-1]
        for idx, up in enumerate(self.up):
            up(meshes,encoder_outs[idx])

    def __call__(self, x, encoder_outs):
        return self.forward(x, encoder_outs)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.conv1 = SplineConv(in_channels, out_channels,dim=2,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.conv2 = SplineConv(out_channels, out_channels,dim=2,kernel_size=[3,3],degree=2,aggr='add').cuda()
        self.unpool = MeshUnpool()

    def __call__(self, meshes, skips):
        return self.forward(meshes,skips)

    def forward(self, meshes,skips):
        meshes = self.unpool(meshes)
        for idx,mesh in enumerate(meshes): 
            v_f = mesh.get_feature()
            edge = mesh.get_undirected_edges()
            edge_attribute = mesh.get_attributes(edge).cuda()
            v_f = self.conv1(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = torch.cat((v_f,skips[idx]),1)
            v_f = self.conv1(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            v_f = self.conv2(v_f,edge.cuda(),edge_attribute)
            v_f = F.relu(v_f)
            mesh.update_feature(v_f)


def reset_params(model): # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def rectangular_conv_block(in_channel,out_channel,kernel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel,padding="same"),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=kernel,padding="same"),
        nn.ReLU()
    )
    return conv