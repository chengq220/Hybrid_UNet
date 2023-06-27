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
        fe,before_pool = encoder((x,meshes))
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
        fe, meshes = x
        for down in self.down:
            fe, before_pool = down((fe,meshes))
            encoder_outs.append(before_pool)
        return fe, encoder_outs

    def __call__(self, x):
        return self.forward(x)

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

    def forward(self, x, encoder_outs):
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
        start_time = time.time()
        x1 = self.conv1(fe, meshes)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print("Mesh conv1 Elapsed time:", elapsed_time, "seconds")
        x1 = F.relu(x1)
        start_time = time.time()
        x1 = self.conv2(x1, meshes)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print("Mesh conv2 Elapsed time:", elapsed_time, "seconds")
        x1 = F.relu(x1)
        x1 = x1.squeeze(3)
        before_pool = None ### Maybe need to create a deep copy of it
        if(self.pool is not None):
            before_pool = x1
            start_time = time.time()
            x1 = self.pool(x1,meshes)
            end_time = time.time()
            elapsed_time = end_time - start_time
            # print("Mesh pool Elapsed time:", elapsed_time, "seconds")
        return x1, before_pool

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels,unroll): #what about unroll and meshunpool
        super(UpConv, self).__init__()
        self.conv1 = MeshConv(in_channels,out_channels)
        self.conv2 = MeshConv(out_channels,out_channels)
        self.unpool = MeshUnpool(unroll)

    def __call__(self, x, from_down=None):
        return self.forward(x, from_down)

    def forward(self, x, from_down):
        from_up, meshes = x
        x1 = from_up
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




# Decoder that converts mesh back to lattice grid
# Parameter: image output size, mesh features, mesh connectivity
# class Mesh_To_Grid_Decoder(nn.Module):
#     def __init__(self):
#         #========= The edges being passing in is wrong for some reason==========
#         super(Mesh_To_Grid_Decoder,self).__init__()
#         self.output = None
#         self.mesh_feature = None
#         #contains the edges of how it is connected
#         self.edges = None
#         self.image_out = None

#     def __deconv_block(self,in_channel,out_channel):
#         deconv = nn.Sequential(
#             nn.ConvTranspose2d(in_channel, (int)(in_channel/2), kernel_size=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(int(in_channel/2), out_channel, kernel_size=1),
#             nn.ReLU()
#         )
#         return deconv
#     #extract the feature into lattice grid
#     def __extract_to_lattice(self):
#         images_index = torch.zeros(self.output[0]*self.output[1])
#         for i in range(self.edges.shape[0]):
#             vertex = self.edges[i] #vertex position
#             vertex_feature = self.mesh_feature[:,i] #the feature of the vertex
#             start_index = int(images_index[vertex]) #where to insert the feature
#             end_index = start_index+int(vertex_feature.shape[1]) #where is the end of vertex
#             self.image_out[:,vertex,start_index:end_index] = vertex_feature
#             images_index[vertex] += vertex_feature.shape[1]

#     def forward(self,features,connectivity,output_dim):
#         #output_dim contains length x width x channel in this order
#         self.output = output_dim
#         self.edges = connectivity.flatten()
#         self.mesh_feature = torch.transpose(features,1,2) #cut the arrays in half
#         batch = self.mesh_feature.shape[0]
#         self.mesh_feature = self.mesh_feature.reshape(batch,self.mesh_feature.shape[1]*2,int(self.mesh_feature.shape[2]/2))
#         dim_channel = 6 * int(self.mesh_feature.shape[2])
#         self.image_out = torch.zeros((batch,self.output[0] * self.output[1],dim_channel)).to(self.mesh_feature.device)
#         self.__extract_to_lattice()
#         self.image_out = self.image_out.reshape((batch,self.output[0],self.output[1],dim_channel))
#         self.image_out = self.image_out.permute(0, 3, 1, 2)
#         d_c = self.__deconv_block(dim_channel,self.output[2]*2).to(self.image_out.device) #??????
#         fe = d_c(self.image_out)
#         # print("=====================")
#         # print(dim_channel)
#         # print(self.output[2])
#         # print("=======================")
#         return fe