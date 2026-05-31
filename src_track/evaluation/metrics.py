"""SRC-Track v2 Evaluation Metrics — ADE/ATE/CTE/RCA/RCE/speed_bias"""
from __future__ import annotations
import numpy as np
import torch
from typing import Dict

R_EARTH = 6371.0
HORIZONS = {2:"12h",4:"24h",8:"48h",12:"72h"}
ST_TRANS = {"ADE":224.4,"ATE":213.7,"CTE":59.4,"12h":77.5,"24h":130.5,"48h":269.9,"72h":423.3}


def haversine_km(lat1,lon1,lat2,lon2):
    r=torch.deg2rad; la1=r(lat1);lo1=r(lon1);la2=r(lat2);lo2=r(lon2)
    dlat=la2-la1;dlon=lo2-lo1
    a=torch.sin(dlat/2)**2+torch.cos(la1)*torch.cos(la2)*torch.sin(dlon/2)**2
    return R_EARTH*2*torch.asin(a.clamp(1e-8,1-1e-8).sqrt())


def compute_ate_cte(pred,gt):
    B,T,_=pred.shape
    cos_lat=torch.cos(torch.deg2rad(gt[...,0])).clamp(1e-3)
    err_lat=(pred[...,0]-gt[...,0])*111.0
    err_lon=(pred[...,1]-gt[...,1])*111.0*cos_lat
    if T<2: return torch.zeros(1),torch.zeros(1)
    gty=torch.zeros(B,T,device=pred.device); gtx=torch.zeros(B,T,device=pred.device)
    gty[:,1:]=(gt[:,1:,0]-gt[:,:-1,0])*111.0
    gtx[:,1:]=(gt[:,1:,1]-gt[:,:-1,1])*111.0*cos_lat[:,1:]
    gty[:,0]=gty[:,1]; gtx[:,0]=gtx[:,1]
    mag=torch.sqrt(gty**2+gtx**2).clamp(1e-3)
    uy=gty/mag; ux=gtx/mag
    ate=(err_lat*uy+err_lon*ux).abs().mean()
    cte=(err_lat*(-ux)+err_lon*uy).abs().mean()
    return ate, cte


class MetricsAccumulator:
    def __init__(self): self.reset()

    def reset(self):
        self._dists=[]; self._ate=[]; self._cte=[]
        self._sp_pred=[]; self._sp_gt=[]
        self._reg_correct=0; self._reg_total=0
        self._reg_dists={0:[],1:[],2:[]}

    @torch.no_grad()
    def update(self, pred, gt, regime_labels=None, rii=None, regime_pred=None):
        pred=pred.float().cpu(); gt=gt.float().cpu(); B,T,_=pred.shape
        d=haversine_km(pred[...,0],pred[...,1],gt[...,0],gt[...,1])
        self._dists.append(d)
        ate,cte=compute_ate_cte(pred,gt)
        self._ate.append(ate.item()); self._cte.append(cte.item())
        if T>=2:
            def _spd(tr):
                dl=tr[:,1:,0]-tr[:,:-1,0]; dln=tr[:,1:,1]-tr[:,:-1,1]
                lat_m=(tr[:,1:,0]+tr[:,:-1,0])/2
                cl=torch.cos(torch.deg2rad(lat_m)).clamp(1e-3)
                return torch.sqrt((dl*111)**2+(dln*111*cl)**2).mean().item()
            self._sp_pred.append(_spd(pred)); self._sp_gt.append(_spd(gt))
        if regime_labels is not None and regime_pred is not None:
            rl=regime_labels.cpu(); rp=regime_pred.cpu().argmax(-1)
            self._reg_correct+=int((rl==rp).sum()); self._reg_total+=B
        if regime_labels is not None:
            rl=regime_labels.cpu(); md=d.mean(1)
            for r in [0,1,2]:
                mask=(rl==r)
                if mask.any(): self._reg_dists[r].extend(md[mask].tolist())

    def compute(self) -> Dict[str,float]:
        if not self._dists: return {}
        all_d=torch.cat(self._dists,0); T=all_d.shape[1]; ps=all_d.mean(0)
        out={"ADE":float(all_d.mean()),"FDE":float(all_d[:,-1].mean()),
             "ATE":float(np.mean(self._ate)),"CTE":float(np.mean(self._cte))}
        for step,label in HORIZONS.items():
            if step-1<T: out[label]=float(ps[step-1])
        if self._sp_pred and self._sp_gt:
            out["speed_bias"]=float(np.mean(self._sp_pred)-np.mean(self._sp_gt))
            out["mean_pred_speed"]=float(np.mean(self._sp_pred))
            out["mean_gt_speed"]=float(np.mean(self._sp_gt))
        if "ATE" in out and "CTE" in out:
            out["ate_cte_ratio"]=out["ATE"]/max(out["CTE"],1e-6)
        if self._reg_total>0:
            out["regime_acc"]=self._reg_correct/self._reg_total
        for r,rn in [(0,"A"),(1,"B"),(2,"C")]:
            if self._reg_dists[r]:
                out[f"ADE_{rn}"]=float(np.mean(self._reg_dists[r]))
        return out


def print_metrics(metrics, tag=""):
    print(f"\n{'='*65}")
    if tag: print(f"  [{tag}]")
    for k in ["ADE","12h","24h","48h","72h"]:
        v=metrics.get(k,float("nan")); ref=ST_TRANS.get(k)
        if ref:
            ok="OK" if v<ref else ".."; gap=v-ref
            print(f"  {k:>4}={v:>7.1f}km [{ok}] ST-Trans={ref} ({gap:+.1f})")
    if "ATE" in metrics and "CTE" in metrics:
        ate=metrics["ATE"]; cte=metrics["CTE"]; ratio=ate/max(cte,1e-6)
        ok="OK" if ratio<2.5 else ".."
        print(f"  ATE={ate:.1f}  CTE={cte:.1f}  ratio={ratio:.2f}x [{ok}] target <2.5x was 3.58x")
    if "speed_bias" in metrics:
        sb=metrics["speed_bias"]; ok="OK" if abs(sb)<3 else ".."
        print(f"  speed_bias={sb:+.2f} km/6h [{ok}] target |bias|<3")
    if "regime_acc" in metrics:
        ra=metrics["regime_acc"]; ok="OK" if ra>0.7 else ".."
        print(f"  regime_acc={ra:.2%} [{ok}] target >70%")
    for r,rn in [(0,"A"),(1,"B"),(2,"C")]:
        k=f"ADE_{rn}"
        if k in metrics: print(f"  ADE_Regime{rn}={metrics[k]:.1f}")
    print(f"{'='*65}\n")
