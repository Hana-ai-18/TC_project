"""
SRC-Track v2 — Loss Functions

L_total = L_pos + w_speed*L_speed + w_regime*L_regime + w_div*L_diversity

CHANGES vs v1:
  [FIX-1] easy_thresh is now a mutable attribute (adaptive threshold hook)
  [FIX-2] L_pos: easy/hard 50/50 blend prevents hard storms dominating early
  [FIX-3] Difficulty weighting uses regime B extra weight properly
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def haversine_distance(pred, gt, eps=1e-8):
    R=6371.0; pr=torch.deg2rad(pred); gr=torch.deg2rad(gt)
    dlat=pr[...,0]-gr[...,0]; dlon=pr[...,1]-gr[...,1]
    a=(torch.sin(dlat/2)**2+torch.cos(pr[...,0])*torch.cos(gr[...,0])*torch.sin(dlon/2)**2)
    return R*2*torch.asin(torch.clamp(a,eps,1-eps).sqrt())


def compute_gt_speed(gt_traj):
    dlat=gt_traj[:,1:,0]-gt_traj[:,:-1,0]; dlon=gt_traj[:,1:,1]-gt_traj[:,:-1,1]
    lat_mid=(gt_traj[:,1:,0]+gt_traj[:,:-1,0])/2.0
    cos_lat=torch.cos(torch.deg2rad(lat_mid)).clamp(min=1e-3)
    return torch.sqrt((dlat*111.0)**2+(dlon*111.0*cos_lat)**2)


def position_loss(pred_traj, gt_traj, regime_labels, rii_values,
                  easy_mask=None, huber_delta=50.0, step_w_min=0.625,
                  step_w_max=2.0, rii_threshold=0.5, rii_weight_scale=1.5,
                  regime_b_extra=0.5):
    T=pred_traj.shape[1]; device=pred_traj.device
    dist_km=haversine_distance(pred_traj,gt_traj)
    t_idx=torch.arange(1,T+1,dtype=torch.float,device=device)
    step_w=step_w_min+(step_w_max-step_w_min)*(t_idx/T)
    diff_w=1.0+rii_weight_scale*F.relu(rii_values-rii_threshold)
    diff_w=diff_w+regime_b_extra*(regime_labels==1).float()
    weighted=dist_km*step_w.unsqueeze(0)
    huber=F.huber_loss(weighted,torch.zeros_like(weighted),delta=huber_delta,reduction='none')
    per_sample=(huber*diff_w.unsqueeze(1)).mean(dim=1)/huber_delta
    if easy_mask is None or not easy_mask.any():
        return per_sample.mean()
    hard_mask=~easy_mask
    l_easy=per_sample[easy_mask].mean() if easy_mask.any() else per_sample.mean()
    l_hard=per_sample[hard_mask].mean() if hard_mask.any() else per_sample.mean()
    return 0.5*l_easy+0.5*l_hard


def speed_loss(pred_speed, gt_traj):
    gt_speed=compute_gt_speed(gt_traj)
    T_min=min(pred_speed.shape[1],gt_speed.shape[1])
    return F.mse_loss(torch.log(pred_speed[:,:T_min]+1.0),torch.log(gt_speed[:,:T_min]+1.0))


def regime_loss(regime_logits, regime_labels, label_smoothing=0.05):
    return F.cross_entropy(regime_logits, regime_labels, label_smoothing=label_smoothing)


def diversity_loss(expert_dirs, regime_labels, threshold=-0.5):
    dir_A=expert_dirs['A']; dir_C=expert_dirs['C']
    cos_sim=(dir_A*dir_C).sum(dim=-1)
    mask_b=(regime_labels==1).float().unsqueeze(1)
    return (F.relu(cos_sim-threshold)*mask_b).mean()


class SRCTrackLoss(nn.Module):
    def __init__(self, w_speed=0.5, w_regime=0.1, w_div=0.05,
                 regime_start_epoch=16, div_start_epoch=31,
                 huber_delta=50.0, step_w_min=0.625, step_w_max=2.0,
                 rii_threshold=0.5, rii_weight_scale=1.5, regime_b_extra=0.5,
                 diversity_threshold=-0.5, label_smoothing=0.05,
                 easy_thresh=0.40):
        super().__init__()
        self.w_speed=w_speed; self.w_regime=w_regime; self.w_div=w_div
        self.regime_start=regime_start_epoch; self.div_start=div_start_epoch
        self.pos_kw=dict(huber_delta=huber_delta,step_w_min=step_w_min,
            step_w_max=step_w_max,rii_threshold=rii_threshold,
            rii_weight_scale=rii_weight_scale,regime_b_extra=regime_b_extra)
        self.diversity_threshold=diversity_threshold
        self.label_smoothing=label_smoothing
        self.easy_thresh=easy_thresh  # mutable — updated by AdaptiveThreshold

    def forward(self, outputs, batch, current_epoch):
        pred_traj=outputs['pred_traj']; pred_speed=outputs['pred_speed']
        regime_logits=outputs['regime_logits']; expert_dirs=outputs['expert_dirs']
        gt_traj=batch['gt_traj']; regime_labels=batch['regime_label']
        rii_values=batch['rii']; difficulty=batch.get('difficulty')

        easy_mask=None; easy_frac=1.0
        if difficulty is not None:
            easy_mask=(difficulty<self.easy_thresh)
            easy_frac=float(easy_mask.float().mean().item())

        l_pos=position_loss(pred_traj,gt_traj,regime_labels,rii_values,
                            easy_mask=easy_mask,**self.pos_kw)
        l_speed=speed_loss(pred_speed,gt_traj)
        l_regime=(regime_loss(regime_logits,regime_labels,self.label_smoothing)
                  if current_epoch>=self.regime_start else pred_traj.new_zeros(()))
        l_div=(diversity_loss(expert_dirs,regime_labels,self.diversity_threshold)
               if current_epoch>=self.div_start else pred_traj.new_zeros(()))

        loss=l_pos+self.w_speed*l_speed+self.w_regime*l_regime+self.w_div*l_div
        return {'loss':loss,'l_pos':l_pos.detach(),'l_speed':l_speed.detach(),
                'l_regime':l_regime.detach(),'l_div':l_div.detach(),'easy_frac':easy_frac}
