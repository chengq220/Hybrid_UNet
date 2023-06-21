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
        self.encoder = None
        self.decoder = None
        #===========================================================

        #trainable parameter to convert mesh back to image
        self.net2 = Mesh_To_Grid_Decoder()

        # Regular UNet Decoder
        #===========================================================
        self.rec_up_conv = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(len(rec_up_convs)-1):
            up = rectangular_conv_block(rec_up_convs[i],rec_up_convs[i+1],3)
            self.rec_up_conv.append(up)
            conv = nn.ConvTranspose2d(rec_up_convs[i],rec_up_convs[i+1],kernel_size=2,stride=2)
            self.conv.append(conv)

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
        for image in r_fe:
            mesh = Mesh(file=image, hold_history=True)
            meshes.append(mesh)
            x.append(mesh.features)
        x = torch.stack(x)
        meshes = np.array(meshes)
 
        cty = meshes[0].edges.copy() #creates a deep copy of the input edge connectivity
        self.nedges = len(cty)
        self.pool_res = [self.nedges]
        for i in range(len(self.down_convs)):
            self.nedges = self.nedges//2
            self.pool_res.append(self.nedges)
        unrolls = self.pool_res[:-1].copy()
        unrolls.reverse()
        self.encoder = MeshEncoder(self.rec_down_channel[-1],self.pool_res, self.down_convs, blocks=len(self.pool_res)-1)
        self.decoder = MeshDecoder(unrolls,self.up_convs, blocks=len(self.pool_res)-1,transfer_data=self.transfer_data)
        
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        fe = self.net2(fe,cty,self.dim)
        return fe

    def __call__self(self,x):
        return self.forward(x)

class MeshEncoder(nn.Module):
    def __init__(self, input_c, pools, convs, fcs=None, blocks=0, global_pool=None):
        super(MeshEncoder, self).__init__()
        self.pool = pools
        self.convs = convs
        self.skip_count = blocks
        in_channel = input_c
        self.down = nn.ModuleList()
        # for out_channel in self.convs:
        #     down = MeshConv(in_channel,out_channel)
        exit()

    def forward(self, x):
        print(self.pool)
        print(self.convs)
        print(self.skip_count)
        fe, meshes = x
        print(fe.shape)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, blocks=0, batch_norm=True, transfer_data=True):
        super(MeshDecoder, self).__init__()

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


# Decoder that converts mesh back to lattice grid
# Parameter: image output size, mesh features, mesh connectivity
class Mesh_To_Grid_Decoder(nn.Module):
    def __init__(self):
        #========= The edges being passing in is wrong for some reason==========
        super(Mesh_To_Grid_Decoder,self).__init__()
        self.output = None
        self.mesh_feature = None
        #contains the edges of how it is connected
        self.edges = None
        self.image_out = None

    def __deconv_block(self,in_channel,out_channel):
        deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, (int)(in_channel/2), kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(int(in_channel/2), out_channel, kernel_size=1),
            nn.ReLU()
        )
        return deconv
    #extract the feature into lattice grid
    def __extract_to_lattice(self):
        images_index = torch.zeros(self.output[0]*self.output[1])
        for i in range(self.edges.shape[0]):
            vertex = self.edges[i] #vertex position
            vertex_feature = self.mesh_feature[:,i] #the feature of the vertex
            start_index = int(images_index[vertex]) #where to insert the feature
            end_index = start_index+int(vertex_feature.shape[1]) #where is the end of vertex
            self.image_out[:,vertex,start_index:end_index] = vertex_feature
            images_index[vertex] += vertex_feature.shape[1]

    def forward(self,features,connectivity,output_dim):
        #output_dim contains length x width x channel in this order
        self.output = output_dim
        self.edges = connectivity.flatten()
        self.mesh_feature = torch.transpose(features,1,2) #cut the arrays in half
        batch = self.mesh_feature.shape[0]
        self.mesh_feature = self.mesh_feature.reshape(batch,self.mesh_feature.shape[1]*2,int(self.mesh_feature.shape[2]/2))
        dim_channel = 6 * int(self.mesh_feature.shape[2])
        self.image_out = torch.zeros((batch,self.output[0] * self.output[1],dim_channel)).to(self.mesh_feature.device)
        self.__extract_to_lattice()
        self.image_out = self.image_out.reshape((batch,self.output[0],self.output[1],dim_channel))
        self.image_out = self.image_out.permute(0, 3, 1, 2)
        d_c = self.__deconv_block(dim_channel,self.output[2]*2).to(self.image_out.device) #??????
        fe = d_c(self.image_out)
        return fe

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