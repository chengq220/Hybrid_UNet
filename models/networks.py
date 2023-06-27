import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models.layers.mesh_pool import MeshPool
from models.layers.mesh_unpool import MeshUnpool
from models.layers.mesh import Mesh
from models.layers.mesh_conv import MeshConv
import numpy as np
from models.loss import DiceLoss,DiceBCELoss
from torch.nn import BCEWithLogitsLoss
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
    def __init__(self, output,rec_down_convs,rec_up_convs,down_convs,up_convs, transfer_data=True):
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
        #the dimensional of the resulting image to be inputted into mesh unet
        self.dim = None
        self.nedges = None
        self.pool_res = None

        #the mesh unet
        #=================================================================
        self.transfer_data = transfer_data
        self.down_convs = down_convs
        self.up_convs = up_convs
        #===========================================================
        # Regular UNet Decoder
        #===========================================================
        self.rec_up_conv = nn.ModuleList()
        self.conv = nn.ModuleList()
        in_channel = self.rec_down_channel[-1] * 2 #times by two because all the edges has the feature of 2 vertices
        for out_channel in rec_up_convs:
            up = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)
            self.rec_up_conv.append(up)
            self.conv.append(rectangular_conv_block(in_channel,out_channel,3))
            in_channel = out_channel

        ##output
        self.output = nn.Conv2d(rec_up_convs[-1],output,kernel_size=1,padding="same")

    def forward(self,x):
        fe = x
        skips = []
        for conv in self.rec_down:
            fe = conv(fe)
            skips.append(fe)
            fe = self.maxpool(fe)
        self.dim = [fe.shape[2],fe.shape[3],fe.shape[1]]

        ##################################################
        #Mesh Unet
        meshes = []
        x = []
        #change the channel to the last number
        r_fe = fe.permute(0,2,3,1)
        #create mesh for the batch
        start_time = time.time()
        for image in r_fe:
            mesh = Mesh(file=image, hold_history=True)
            meshes.append(mesh)
            x.append(mesh.features)
        x = torch.stack(x)
        meshes = np.array(meshes)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print("Create Mesh Elapsed time:", elapsed_time, "seconds")
 
        cty = meshes[0].edges.copy() #creates a deep copy of the input edge connectivity
        self.nedges = len(cty)
        self.pool_res = []
        div_coef = len(self.down_convs) + 1
        for i in range(len(self.down_convs)-1):
            self.nedges = self.nedges//div_coef
            self.pool_res.append(self.nedges)
            div_coef = div_coef - 1
        unrolls = self.pool_res[:-1]
        unrolls = unrolls[::-1] + [len(cty)]

        start_time = time.time()

        encoder = MeshEncoder(self.rec_down_channel[-1], self.down_convs, self.pool_res)
        decoder = MeshDecoder(unrolls,self.up_convs)
        fe,before_pool, mask, order = encoder((x,meshes))
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print("Mesh Encoder Elapsed time:", elapsed_time, "seconds")
        
        fe = decoder((fe, meshes), before_pool[:-1])
        ##################################################################
        skips = skips[::-1]
        for i in range(len(self.rec_up_conv)):
            fe = self.rec_up_conv[i](fe)
            fe = torch.cat((fe, skips[i]),1)
            fe = self.conv[i](fe)
        fe = self.output(fe).squeeze(1)
        return fe

    def __call__self(self,x):
        return self.forward(x)

class MeshEncoder(nn.Module):
    def __init__(self, input_channel, convs, pool_res):
        super(MeshEncoder, self).__init__()
        in_channel = input_channel * 2
        self.down = nn.ModuleList()
        for idx, out_channel in enumerate(convs):
            if(idx == len(convs)-1):
                #bottleneck
                down = DownConv(in_channel,out_channel, None)
            else:
                down = DownConv(in_channel,out_channel,pool_res[idx])
            self.down.append(down)
            in_channel = out_channel

    def forward(self, x):
        encoder_outs = []
        mask = []
        order = []
        fe, meshes = x
        for down in self.down:
            before_pool, out_image, out_mask,pool_order,fe = down((fe,meshes))
            encoder_outs.append(before_pool)
            mask.append(out_mask)
            order.append(pool_order)
        return fe, encoder_outs, mask, order

    def __call__(self, x):
        return self.forward(x)

## need to change it so it takes in pool order and pool mask
class MeshDecoder(nn.Module):
    def __init__(self, unrolls, up_convs):
        super(MeshDecoder, self).__init__()
        self.up = nn.ModuleList()
        self.conv1 = UpConv(up_convs[0],up_convs[1],unrolls[0])
        in_channel = up_convs[0]
        up_convs = up_convs[1:]

        for idx,out_channel in enumerate(up_convs):
            up = UpConv(in_channel, out_channel,unrolls[idx])
            self.up.append(up)
            in_channel = out_channel

    def forward(self, x, encoder_outs, mask, order):
        fe, meshes = x
        encoder_outs = encoder_outs[::-1]
        # fe = self.conv1((fe,meshes),encoder_outs[0])
        for idx, up in enumerate(self.up):
            fe = up((fe,meshes),encoder_outs[idx])
        return fe

    def __call__(self, x, encoder_outs):
        return self.forward(x, encoder_outs)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pool):
        super(DownConv, self).__init__()
        self.pool = None
        self.conv1 = MeshConv(in_channels, out_channels)
        self.conv2 = MeshConv(out_channels, out_channels)
        if(pool is not None):
            self.pool = MeshPool(pool)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        fe, meshes = x
        x1 = self.conv1(fe, meshes)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, meshes)
        x1 = F.relu(x1)
        x1 = x1.squeeze(3)
        before_pool = None
        out_image = None
        image_mask = None
        collapse_order = None
        if(self.pool is not None):
            start_time = time.time()
            images = []
            for mesh in meshes:
                images.append(mesh.image)
            before_pool = torch.stack(images)

            #the new image, which vertex is taken out of the original img, order of edge collapse, edge feature
            out_image, image_mask, collapse_order, x1 = self.pool(images,x1,meshes)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Mesh pool Elapsed time:", elapsed_time, "seconds")
        #before_pool, pooled_image,image_mask,collapse_order,edge_features
        return before_pool, out_image, image_mask, collapse_order, x1

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels,unroll): #what about unroll and meshunpool
        super(UpConv, self).__init__()
        self.conv1 = MeshConv(in_channels,out_channels)
        self.conv2 = MeshConv(out_channels,out_channels)
        self.unpool = MeshUnpool()

    def __call__(self, pool_vmask,pooled_image, pool_order, x, from_down=None):
        return self.forward(pool_vmask,pooled_image,pool_order, x, from_down)

    def forward(self, pool_mask, pooled_image, pooled_order, x, from_down):
        from_up, meshes = x
        print(from_down.shape)
        x1 = from_up
        image = self.unpool(from_down,pool_mask,pooled_image,pooled_order)
        print(image.shape)
        exit()
        x1 = self.unpool(x1,meshes)
        x1 = self.conv1(x1,meshes).squeeze(3)
        x1 = torch.cat((x1,from_down),1)
        x1 = self.conv1(x1,meshes)
        x1 = F.relu(x1)
        x1 = self.conv2(x1,meshes)
        x1 = F.relu(x1).squeeze(3)
        return x1

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
