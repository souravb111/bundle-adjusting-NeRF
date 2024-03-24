import warnings
import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import visdom
import matplotlib.pyplot as plt

import util,util_vis
from util import log,debug
from . import nerf
import camera

# ============================ main engine for training and evaluation ============================

class DelayedExponentialLR(torch.optim.lr_scheduler.LRScheduler):
    """Starts ExponentialLR at start_epoch
    """

    def __init__(self, optimizer, gamma, start_epoch=0, last_epoch=-1, verbose="deprecated"):
        self.gamma = gamma
        self.start_epoch = start_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.start_epoch:
            return [0.0 for _ in self.base_lrs]
        else:
            relative_epoch = self.last_epoch - self.start_epoch
            res = [base_lr * self.gamma ** relative_epoch for base_lr in self.base_lrs]
            return res

class Model(nerf.Model):

    def __init__(self,opt):
        super().__init__(opt)


    def build_networks(self,opt):
        super().build_networks(opt)
        if opt.camera.noise:
            # pre-generate synthetic pose perturbation
            se3_noise = torch.randn(len(self.train_data),6,device=opt.device)*opt.camera.noise
            self.graph.pose_noise = camera.lie.se3_to_SE3(se3_noise)
        self.graph.se3_refine = torch.nn.ModuleList([torch.nn.Embedding(1, 6).to(opt.device) for _ in range(len(self.train_data))])
        # self.graph.se3_refine = torch.nn.Embedding(len(self.train_data),6).to(opt.device)
        for i in range(len(self.train_data)):
            torch.nn.init.zeros_(self.graph.se3_refine[i].weight)

        num_images = len(self.train_data)
        self.max_iters = opt.schedule.init_num_iters + ((num_images - opt.schedule.init_num_images) * opt.schedule.per_image_num_iters) + opt.schedule.finalize_num_iters
        print(f"Max iters: {self.max_iters}")

        self.per_image_loss_weighting = []
        for i in range(opt.schedule.init_num_images):
            self.per_image_loss_weighting.append(
                lambda it: 1.0
            )
        for i in range(opt.schedule.init_num_images, len(self.train_data)):
            start_step = opt.schedule.init_num_iters + ((i - opt.schedule.init_num_images) * opt.schedule.per_image_num_iters)
            end_step = start_step + opt.schedule.duration_increase
            # cosine increase from 0.0 to 1.0 over duration increase
            self.per_image_loss_weighting.append(
                lambda it, ss=start_step, es=end_step: 0.0 if it < ss else (0.5 * (1.0 - np.cos(np.pi * (it - ss) / (es - ss))) if it < es else 1.0)
            )

        self.pose_lr_schedules = []
        # exponential decay from pose_lr to pose_lr_end over duration
        self.pose_lr_schedules.append(
            lambda it: opt.schedule.pose_lr * (opt.schedule.pose_lr_end / opt.schedule.pose_lr) ** (it / self.max_iters)
        )
        for i in range(opt.schedule.init_num_images, len(self.train_data)):
            start_step = opt.schedule.init_num_iters + ((i - opt.schedule.init_num_images) * opt.schedule.per_image_num_iters)
            # 0 until start step, then exponential decay
            self.pose_lr_schedules.append(
                lambda it, ss=start_step: 0.0 if it < ss else opt.schedule.pose_lr * (opt.schedule.pose_lr_end / opt.schedule.pose_lr) ** ((it - ss) / (self.max_iters - ss))
            )


    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer
        # training
        # self.validate(opt,0)

        # pre-compute rays for training:
        loader = tqdm.trange(self.max_iters,desc="training",leave=False)
        var = self.train_data.all
        self.var = var

        for self.it in loader:

            self.graph.per_image_loss_weighting = [self.per_image_loss_weighting[i](self.it) for i in range(len(self.train_data))]
            if self.it<self.iter_start: continue
            # set var to all available images
            temp_var = {}
            if self.it < opt.schedule.init_num_iters:
                num_images_available = opt.schedule.init_num_images
            else:
                num_images_available = int( (self.it - opt.schedule.init_num_iters) / opt.schedule.per_image_num_iters ) + opt.schedule.init_num_images
            num_images_available = min(num_images_available, len(self.train_data))

            mask = torch.zeros(len(self.train_data), dtype=torch.bool, device=opt.device)
            mask[:num_images_available] = True

            for key, val in var.items():
                temp_var[key] = val[mask]
            self.mask_update(mask)
            temp_var = edict(temp_var)
            self.train_iteration(opt,temp_var,loader)

            if opt.optim.sched: self.sched.step()
            if self.it%opt.freq.val==0: self.validate(opt,self.it)
            if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    def mask_update(self, mask):
        top_index = mask.nonzero().max()
        with torch.no_grad():
            for i in range(top_index+1, len(self.train_data)):
                self.graph.se3_refine[i].weight.copy_(self.graph.se3_refine[top_index].weight.clone())

    def setup_optimizer(self,opt):
        super().setup_optimizer(opt)
        optimizer = getattr(torch.optim,opt.optim.algo)
        # make a parameter group for each self.graph.se3_refine.weight
        self.optim_pose = optimizer(
            [
                dict(params=self.graph.se3_refine[:opt.schedule.init_num_images].parameters(), lr=opt.schedule.pose_lr),
            ] + 
            [
                dict(params=self.graph.se3_refine[i:i+1].parameters(), lr=opt.schedule.pose_lr) for i in range(opt.schedule.init_num_images, len(self.train_data))
            ]
        )

    def train_iteration(self,opt,var,loader):
        # self.optim_pose.zero_grad()
        self.optim_pose.zero_grad()
        # if opt.optim.warmup_pose:
        #     # simple linear warmup of pose learning rate
        #     self.optim_pose.param_groups[0]["lr_orig"] = self.optim_pose.param_groups[0]["lr"] # cache the original learning rate
        #     self.optim_pose.param_groups[0]["lr"] *= min(1,self.it/opt.optim.warmup_pose)
        loss = super().train_iteration(opt,var,loader)
        self.optim_pose.step()
        # update lr
        for i, param_group in enumerate(self.optim_pose.param_groups):
            param_group["lr"] = self.pose_lr_schedules[i](self.it)

        self.graph.nerf.progress.data.fill_(self.it/self.max_iters)
        return loss

    @torch.no_grad()
    def validate(self,opt,ep=None):
        pose,pose_GT = self.get_all_training_poses(opt)
        # _,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        super().validate(opt,ep=ep)

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train"):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        if split=="train":
            # log learning rate
            for i,param_group in enumerate(self.optim_pose.param_groups):
                lr = param_group["lr"]
                self.tb.add_scalar("{0}/{1}".format(split,f"lr_pose_{i}"),lr,step)
                self.tb.add_scalar("{0}/{1}".format(split,f"loss_weight_{i}"),self.graph.per_image_loss_weighting[i],step)
            if step%2000 == 0:
                fig = plt.figure(figsize=(10,10) if opt.data.dataset=="blender" else (16,8))
                cam_path = "{}/training_poses".format(opt.output_path)
                os.makedirs(cam_path,exist_ok=True)
                pose,pose_ref = self.get_all_training_poses(opt)
                if opt.data.dataset in ["blender","llff"]:
                    pose_aligned,_ = self.prealign_cameras(opt,pose,pose_ref)
                    pose_aligned,pose_ref = pose_aligned.detach().cpu(),pose_ref.detach().cpu()
                    dict(
                        blender=util_vis.plot_save_poses_blender,
                        llff=util_vis.plot_save_poses,
                    )[opt.data.dataset](opt,fig,pose_aligned,pose_ref=pose_ref,path=cam_path,ep=step)
                else:
                    pose = pose.detach().cpu()
                    util_vis.plot_save_poses(opt,fig,pose,pose_ref=None,path=cam_path,ep=step,cam_depth=0.2)
        # compute pose error
        if split=="train" and opt.data.dataset in ["blender","llff"]:
            pose,pose_GT = self.get_all_training_poses(opt)
            pose_aligned,_ = self.prealign_cameras(opt,pose,pose_GT)
            error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
            self.tb.add_scalar("{0}/error_R".format(split),error.R.mean(),step)
            self.tb.add_scalar("{0}/error_t".format(split),error.t.mean(),step)

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train",ind=0):
        super().visualize(opt,var,step=step,split=split,ind=ind)
        if opt.visdom:
            if split=="val":
                pose,pose_GT = self.get_all_training_poses(opt)
                util_vis.vis_cameras(opt,self.vis,step=step,poses=[pose,pose_GT])

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        # add synthetic pose perturbation to all training data
        if opt.data.dataset=="blender":
            pose = pose_GT
            if opt.camera.noise:
                pose = camera.pose.compose([self.graph.pose_noise,pose])
        else: pose = self.graph.pose_eye
        # add learned pose correction to all training data
        pose_refine = camera.lie.se3_to_SE3(self.graph.get_se3_refine_weight())
        pose = camera.pose.compose([pose_refine,pose])
        return pose,pose_GT

    @torch.no_grad()
    def prealign_cameras(self,opt,pose,pose_GT):
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
        center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
        R_aligned = pose[...,:3]@sim3.R.t()
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
        return pose_aligned,sim3

    @torch.no_grad()
    def evaluate_camera_alignment(self,opt,pose_aligned,pose_GT):
        # measure errors in rotation and translation
        R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
        R_GT,t_GT = pose_GT.split([3,1],dim=-1)
        R_error = camera.rotation_distance(R_aligned,R_GT)
        t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
        error = edict(R=R_error,t=t_error)
        return error

    @torch.no_grad()
    def evaluate_full(self,opt):
        self.graph.eval()
        # evaluate rotation/translation
        pose,pose_GT = self.get_all_training_poses(opt)
        pose_aligned,self.graph.sim3 = self.prealign_cameras(opt,pose,pose_GT)
        error = self.evaluate_camera_alignment(opt,pose_aligned,pose_GT)
        print("--------------------------")
        print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
        print("trans: {:10.5f}".format(error.t.mean()))
        print("--------------------------")
        # dump numbers
        quant_fname = "{}/quant_pose.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,(err_R,err_t) in enumerate(zip(error.R,error.t)):
                file.write("{} {} {}\n".format(i,err_R.item(),err_t.item()))
        # evaluate novel view synthesis
        super().evaluate_full(opt)

    @torch.enable_grad()
    def evaluate_test_time_photometric_optim(self,opt,var):
        # use another se3 Parameter to absorb the remaining pose errors
        var.se3_refine_test = torch.nn.Parameter(torch.zeros(1,6,device=opt.device))
        optimizer = getattr(torch.optim,opt.optim.algo)
        optim_pose = optimizer([dict(params=[var.se3_refine_test],lr=opt.optim.lr_pose)])
        iterator = tqdm.trange(opt.optim.test_iter,desc="test-time optim.",leave=False,position=1)
        for it in iterator:
            optim_pose.zero_grad()
            var.pose_refine_test = camera.lie.se3_to_SE3(var.se3_refine_test)
            var = self.graph.forward(opt,var,mode="test-optim")
            loss = self.graph.compute_loss(opt,var,mode="test-optim")
            loss = self.summarize_loss(opt,var,loss)
            loss.all.backward()
            optim_pose.step()
            iterator.set_postfix(loss="{:.3f}".format(loss.all))
        return var

    @torch.no_grad()
    def generate_videos_pose(self,opt):
        self.graph.eval()
        fig = plt.figure(figsize=(10,10) if opt.data.dataset=="blender" else (16,8))
        cam_path = "{}/poses".format(opt.output_path)
        os.makedirs(cam_path,exist_ok=True)
        ep_list = []
        for ep in range(0,self.max_iters+1,opt.freq.ckpt):
            # load checkpoint (0 is random init)
            if ep!=0:
                try: util.restore_checkpoint(opt,self,resume=ep)
                except: continue
            # get the camera poses
            pose,pose_ref = self.get_all_training_poses(opt)
            if opt.data.dataset in ["blender","llff"]:
                pose_aligned,_ = self.prealign_cameras(opt,pose,pose_ref)
                pose_aligned,pose_ref = pose_aligned.detach().cpu(),pose_ref.detach().cpu()
                dict(
                    blender=util_vis.plot_save_poses_blender,
                    llff=util_vis.plot_save_poses,
                )[opt.data.dataset](opt,fig,pose_aligned,pose_ref=pose_ref,path=cam_path,ep=ep)
            else:
                pose = pose.detach().cpu()
                util_vis.plot_save_poses(opt,fig,pose,pose_ref=None,path=cam_path,ep=ep,cam_depth=0.2)
            ep_list.append(ep)
        plt.close()
        # write videos
        print("writing videos...")
        list_fname = "{}/temp.list".format(cam_path)
        with open(list_fname,"w") as file:
            for ep in ep_list: file.write("file {}.png\n".format(ep))
        cam_vid_fname = "{}/poses.mp4".format(opt.output_path)
        os.system("ffmpeg -y -r 30 -f concat -i {0} -pix_fmt yuv420p {1} >/dev/null 2>&1".format(list_fname,cam_vid_fname))
        os.remove(list_fname)

# ============================ computation graph for forward/backprop ============================

class Graph(nerf.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)
        self.pose_eye = torch.eye(3,4).to(opt.device)
        self.per_image_loss_weighting = None

    def get_se3_refine_weight(self):
        return torch.concat([self.se3_refine[i].weight for i in range(len(self.se3_refine))], dim=0)

    def get_pose(self,opt,var,mode=None):
        # if mode=="train":
            # add the pre-generated pose perturbations
        if opt.data.dataset=="blender":
            if opt.camera.noise:
                var.pose_noise = self.pose_noise[var.timestep]
                pose = camera.pose.compose([var.pose_noise,var.pose])
            else: pose = var.pose
        else: pose = self.pose_eye
        # add learnable pose correction
        var.se3_refine = self.get_se3_refine_weight()[var.timestep]
        pose_refine = camera.lie.se3_to_SE3(var.se3_refine)
        pose = camera.pose.compose([pose_refine,pose])
        # elif mode in ["val","eval","test-optim"]:
        #     # align test pose to refined coordinate system (up to sim3)
        #     sim3 = self.sim3
        #     center = torch.zeros(1,1,3,device=opt.device)
        #     center = camera.cam2world(center,var.pose)[:,0] # [N,3]
        #     center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
        #     R_aligned = var.pose[...,:3]@self.sim3.R
        #     t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        #     pose = camera.pose(R=R_aligned,t=t_aligned)
        #     # additionally factorize the remaining pose imperfection
        #     if opt.optim.test_photo and mode!="val":
        #         pose = camera.pose.compose([var.pose_refine_test,pose])
        # else: pose = var.pose
        return pose

    def MSE_loss(self,pred,label=0):
        # label is of shape (num_images, ray_idx, 3)
        assert self.per_image_loss_weighting is not None
        loss = (pred.contiguous()-label)**2 # (num_images, ray_idx)
        for i in range(loss.shape[0]):
            loss[i] = self.per_image_loss_weighting[i] * loss[i]
        return loss.mean()

class NeRF(nerf.NeRF):

    def __init__(self,opt):
        super().__init__(opt)
        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

    def positional_encoding(self,opt,input,L): # [B,...,N]
        input_enc = super().positional_encoding(opt,input,L=L) # [B,...,2NL]
        # coarse-to-fine: smoothly mask positional encoding for BARF
        if opt.barf_c2f is not None:
            # set weights for different frequency bands
            start,end = opt.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*L
            k = torch.arange(L,dtype=torch.float32,device=opt.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,L)*weight).view(*shape)
        return input_enc
