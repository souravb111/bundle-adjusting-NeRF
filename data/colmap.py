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
        #self.list = [l for l in sorted(os.listdir(self.path_image),key=lambda f: int(f.split(".")[0])) if "depth" not in l]
        # manually split train/val subsets
        # open json file to get camera transforms
        with open(os.path.join(self.path, "transforms.json"), "r") as f:
            self.transforms = json.load(f)

        self.transforms["frames"] = [f for f in self.transforms["frames"] if os.path.exists(os.path.join(self.path_image, f["file_path"].split("/")[-1]))]
        

        # filter to only contain those images that exist on disk
        if hasattr(opt.data, "subsample"):
            self.transforms["frames"] = self.transforms["frames"][::opt.data.subsample]

        self.list = [f["file_path"].split("/")[-1] for f in self.transforms["frames"]]

        num_val_split = int(len(self) * opt.data.val_ratio)
        self.list = self.list[:-num_val_split] if split=="train" else self.list[-num_val_split:]
        self.transforms["frames"] = self.transforms["frames"][:-num_val_split] if split=="train" else self.transforms["frames"][-num_val_split:]

        if subset:
            self.list = self.list[:subset]
            self.transforms["frames"] = self.transforms["frames"][:subset]

        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt):
        pose_raw_all = [torch.tensor(f["transform_matrix"],dtype=torch.float32) for f in self.transforms["frames"]]
        pose_canon_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_canon_all

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
            time=idx/len(self.list)
        )
        return sample

    def get_image(self, opt, idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self, opt, idx):
        cy = self.transforms["cy"]
        cx = self.transforms["cx"]
        fy = self.transforms["fl_y"]
        fx = self.transforms["fl_x"]
        assert self.raw_H == self.transforms["h"]
        assert self.raw_W == self.transforms["w"]
        intr = torch.tensor([[fx,0,cx],
                             [0,fy,cy],
                             [0,0,1]]).float()
        pose_raw = torch.tensor(self.transforms["frames"][idx]["transform_matrix"]).float()
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr, pose

    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose