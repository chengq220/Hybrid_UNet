import os
import torch
from data.base_dataset import BaseDataset
import numpy as np
from PIL import Image, ImageOps
import albumentations as A

class SegmentationData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)
        self.mask_path = self.get_image_files(self.paths,os.path.join(self.root,'GroundTruth'),ext='.png')
        self.classes = np.loadtxt(os.path.join(self.root,'classes.txt'))
#         self.classes, self.offset = self.get_n_segs(os.path.join(self.root, 'classes.txt'), self.seg_paths)
        self.nclasses = opt.classes
        self.size = len(self.paths)
        # self.get_mean_std()
        self.resize = opt.resize
        self.feature_transform = A.Compose([
            A.CenterCrop(self.resize[0],self.resize[1]),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])
        self.label_transform = A.Compose([
            A.CenterCrop(self.resize[0],self.resize[1]),
        ])
        # # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels


    def __getitem__(self, index):
        path = self.paths[index]
        feature = read_img(path,"feature")
        feature = self.feature_transform(image=feature)["image"]
        feature = feature.transpose(2, 0, 1)
        label = read_img(self.mask_path[index],"label")
        label = self.label_transform(image=label)["image"]
        return (feature, label)

    def __len__(self):
        return self.size

    @staticmethod
    def get_image_files(paths, img_dir, ext='.png'):
        imgs = []
        for path in paths:
            img_file = os.path.join(img_dir, os.path.splitext(os.path.basename(path))[0] + ext)
            # print(img_file)
            assert(os.path.isfile(img_file))
            imgs.append(img_file)
        return imgs

    @staticmethod
    def make_dataset(path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                path = os.path.join(root, fname)
                meshes.append(path)

        return meshes

    
def read_img(file,category):
    label = Image.open(file)
    if(category == "label"):
        label = label.convert('L')
    label = np.array(label)
    
    #if it is label, convert all values to 0 or 1s
    if(category == "label"):
        label = label/255
        label = label.astype(np.uint8)
    return label