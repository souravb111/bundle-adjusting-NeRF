from typing import List
import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
import camera
from util import log,debug

class Dataset(base.Dataset):

    def __init__(self, opt, split="train", subset=None):
        self.raw_H,self.raw_W = opt.data.image_size
        super().__init__(opt,split)
        assert opt.data.root is not None
        self.root = opt.data.root
        self.path = os.path.join(self.root, opt.data.scene)
        self.path_image = os.path.join(self.path, "images")
        self.list = [l for l in sorted(os.listdir(self.path_image),key=lambda f: int(f.split(".")[0])) if "depth" not in l]
        # manually split train/val subsets
        num_val_split = int(len(self) * opt.data.val_ratio)
        self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        if subset: self.list = self.list[:subset]
        # open json file to get camera transforms
        with open(os.path.join(self.path, "transforms.json"), "r") as f:
            self.transforms = json.load(f)["frames"]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt):
        # poses are unknown, so just return some dummy poses (identity transform)
        return camera.pose(t=torch.zeros(len(self),3))

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self, opt, idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self, opt, idx):
        cy = self.transforms[idx]["cy"]
        cx = self.transforms[idx]["cx"]
        fy = self.transforms[idx]["fl_y"]
        fx = self.transforms[idx]["fl_x"]
        assert self.raw_H == self.transforms[idx]["h"]
        assert self.raw_W == self.transforms[idx]["w"]
        transform_matrix = torch.tensor(self.transforms[idx]["transform_matrix"]).float()
        intr = torch.tensor([[fx,0,cx],
                             [0,fy,cy],
                             [0,0,1]]).float()
        pose = camera.pose(R=transform_matrix[:3, :3], t=transform_matrix[:3, -1]) # dummy pose, won't be used
        return intr, pose