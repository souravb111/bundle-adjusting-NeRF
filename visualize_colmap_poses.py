

from matplotlib import pyplot as plt
import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log
import util,util_vis

def main():

    log.process(os.getpid())

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    data = importlib.import_module("data.{}".format(opt.data.dataset))
    log.info("loading training data...")
    train_data = data.Dataset(opt,split="train",subset=opt.data.train_sub)

    pose_GT = train_data.get_all_camera_poses(opt).to("cpu")

    fig = plt.figure(figsize=(10,10) if opt.data.dataset=="blender" else (16,8))
    cam_path = "{}/poses".format(opt.output_path)
    os.makedirs(cam_path,exist_ok=True)
    util_vis.plot_save_poses(opt,fig,pose_GT,pose_ref=None,path=cam_path,ep=0,cam_depth=0.2)

if __name__=="__main__":
    main()
