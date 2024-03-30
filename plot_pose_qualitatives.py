import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log
import util,util_vis
import matplotlib.pyplot as plt
import camera
from easydict import EasyDict as edict

@torch.no_grad()
def prealign_cameras(opt,pose,pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1,1,3,device=opt.device)
    center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
    center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
    try:
        sim3 = camera.procrustes_analysis(center_GT,center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=opt.device))
    # align the camera poses
    return sim3
    # center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    # R_aligned = pose[...,:3]@sim3.R.t()
    # t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    # pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
    # return pose_aligned,sim3

@torch.no_grad()
def generate_videos_pose(model, opt):
    model.graph.eval()
    fig = plt.figure(figsize=(10,10) if opt.data.dataset=="blender" else (16,8))
    cam_path = "{}/poses_for_paper".format(opt.output_path)
    os.makedirs(cam_path,exist_ok=True)

    pose,_ = model.get_all_training_poses(opt)

    if not hasattr(model, "max_iters"):
        model.max_iters = 200000

    util.restore_checkpoint(opt,model,resume=(model.max_iters//opt.freq.ckpt)*opt.freq.ckpt)
    pose_to_align,pose_ref = model.get_all_training_poses(opt)
    sim3 = prealign_cameras(opt,pose_to_align,pose_ref)

    ep_list = []
    for ep in range(0,model.max_iters+1 - ((opt.schedule.finalize_num_iters - 20000) if hasattr(opt, "schedule") else 0), opt.freq.ckpt):
        # load checkpoint (0 is random init)
        if ep!=0:
            try: util.restore_checkpoint(opt,model,resume=ep)
            except: 
                print(f"skipping {ep}")
                continue
        # get the camera poses
            pose,_ = model.get_all_training_poses(opt)
        center = torch.zeros(1,1,3,device=opt.device)
        center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
        center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
        R_aligned = pose[...,:3]@sim3.R.t()
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
        pose_aligned,pose_ref = pose_aligned.detach().cpu(),pose_ref.detach().cpu()

        transform = camera.pose.invert(pose_ref[0].clone())
        ref_poses = []
        pred_poses = []
        for i in range(len(pose_ref)):
            ref_pose = camera.pose.compose([transform, pose_ref[i]])
            ref_poses.append(ref_pose)
            pred_pose = camera.pose.compose([transform, pose_aligned[i]])
            pred_poses.append(pred_pose)
        util_vis.plot_save_poses(opt,fig,torch.stack(pred_poses,dim=0),pose_ref=torch.stack(ref_poses,dim=0),path=cam_path,ep=ep,cam_depth=0.2)
        ep_list.append(ep)
    plt.close()
    # write videos
    print("writing videos...")
    list_fname = "{}/temp.list".format(cam_path)
    with open(list_fname,"w") as file:
        for ep in ep_list: file.write("file {}.png\n".format(ep))
    cam_vid_fname = "{}/poses_for_paper.mp4".format(opt.output_path)
    os.system("ffmpeg -y -r 2 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname,cam_vid_fname))
    os.remove(list_fname)

def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for evaluating NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt,eval_split="test")
        m.build_networks(opt)

        generate_videos_pose(m,opt)
        # if opt.data.dataset in ["blender","llff"]:
        # m.evaluate_full(opt)
        # m.generate_videos_synthesis(opt)

if __name__=="__main__":
    main()
