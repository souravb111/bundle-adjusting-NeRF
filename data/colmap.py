from functools import cached_property
import queue
import threading
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

import tqdm

from . import base
import camera
from util import log,debug
import random



class Dataset(base.Dataset):

    def __init__(self, opt, split="train", subset=None):
        assert subset is None
        self.opt = opt
        torch.manual_seed(0)
        random.seed(0)
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
        # self.transforms["frames"] = sorted(self.transforms["frames"], key= lambda x: int(x["file_path"].split("/")[-1].split(".")[0]))
        self.list = [f["file_path"].split("/")[-1] for f in self.transforms["frames"]]

        # re-orient around a canonical pose
        self.max_timestamp = -1.0
        for frame in self.transforms["frames"]:
            if "timestamp" in frame:
                self.max_timestamp = max(self.max_timestamp, frame["timestamp"])

        assert len(self.items) > 0
        # preload dataset
        # if opt.data.preload:
        #     self.images = self.preload_threading(opt,self.get_image)
        #     self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def __len__(self):
        return len(self.items)

    def preload_threading(self,opt,load_func,data_str="images"):
        data_list = [None]*len(self)
        q = queue.Queue(maxsize=len(self))
        idx_tqdm = tqdm.tqdm(range(len(self)),desc="preloading {}".format(data_str),leave=False)
        for i in range(len(self)): q.put(i)
        lock = threading.Lock()
        for ti in range(opt.data.num_workers):
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list,load_func,q,lock,idx_tqdm),daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None,data_list)))
        return {item: data_list[i] for i, item in enumerate(self.items)}

    def subsample(self, items):

        if hasattr(self.opt.data, "shuffle") and self.opt.data.shuffle:
            random.shuffle(items)

        if hasattr(self.opt.data, "subsample"):
            items = items[::self.opt.data.subsample]

        all_inds = list(range(len(items)))
        random.shuffle(all_inds)

        num_val_split = int(len(items) * self.opt.data.val_ratio)
        if self.split != "train":
            items = [items[i] for i in all_inds[-num_val_split:]]
        #if self.split == "train":
        #    items = [items[i] for i in all_inds[:-num_val_split]]
        #else:
        #    items = [items[i] for i in all_inds[-num_val_split:]]

        items = sorted(items, key=lambda i: self.get_time(self.transforms["frames"][i], i))

        return items
    
    def get_time(self, transform, frame_id):
        if "timestamp" in transform:
            assert self.max_timestamp > 0
            return transform["timestamp"] / self.max_timestamp
        else:
            mock_timestamp = int(self.list[frame_id].split(".")[0])
            return mock_timestamp / len(self.list)

    @cached_property
    def items(self):
        items = range(len(self.list))
        return self.subsample(items)

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt):
        pose_raw_all = [torch.tensor(self.transforms["frames"][i]["transform_matrix"], dtype=torch.float32) for i in self.items]
        pose_canon_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_canon_all

    def __getitem__(self,idx):
        frame_id = self.items[idx]
        opt = self.opt
        sample = dict(idx=idx)
        sample["time"] = self.get_time(self.transforms["frames"][frame_id], frame_id)
        sample["frame_id"] = frame_id
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.get_image(opt, frame_id) # self.images[frame_id] if opt.data.preload else self.get_image(opt,frame_id)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.get_camera(opt, frame_id) # self.cameras[frame_id] if opt.data.preload else self.get_camera(opt,frame_id)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self, opt, frame_id):
        image_fname = "{}/{}".format(self.path_image,self.list[frame_id])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self, opt, frame_id):
        cy = self.transforms["cy"]
        cx = self.transforms["cx"]
        fy = self.transforms["fl_y"]
        fx = self.transforms["fl_x"]
        assert self.raw_H == self.transforms["h"]
        assert self.raw_W == self.transforms["w"]
        intr = torch.tensor([[fx,0,cx],
                             [0,fy,cy],
                             [0,0,1]]).float()
        pose_raw = torch.tensor(self.transforms["frames"][frame_id]["transform_matrix"]).float()
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr, pose

    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose # t_camera_world