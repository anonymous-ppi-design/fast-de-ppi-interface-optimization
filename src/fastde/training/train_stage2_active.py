from __future__ import annotations
import argparse, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from fastde.data.sequence_utils import AA_TO_IDX, clean_sequence, sequence_hash, hamming_distance
from fastde.models.sparse_mutation_vae import SparseMutationVAE, VAEConfig, vae_loss
from fastde.models.trainable_head import TrainablePerformanceHead
from fastde.models.frozen_heads import RandomFrozenHeadEnsemble, RFHConfig
from fastde.generation.latent_search import decode_latents, optimize_latents
class LabelDataset(Dataset):
    def __init__(self,df,seq_col,score_col,score_center,score_scale):
        self.seqs=[clean_sequence(s) for s in df[seq_col].tolist()]; self.x=torch.tensor([[AA_TO_IDX[a] for a in s] for s in self.seqs],dtype=torch.long); score=pd.to_numeric(df[score_col],errors='coerce').fillna(0.0).values; self.y=torch.tensor(np.clip((score-score_center)/(score_scale+1e-8),-4,4),dtype=torch.float32); self.w=torch.ones(len(self.x),dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self,idx): return self.x[idx],self.y[idx],self.w[idx]
def load_vae(path,device):
    ckpt=torch.load(path,map_location=device); cfg=VAEConfig(**ckpt['vae_config']); model=SparseMutationVAE(ckpt['wt_sequence'],cfg).to(device); model.load_state_dict(ckpt['model_state_dict']); return model,ckpt['wt_sequence']
@torch.no_grad()
def encode_df(model,df,seq_col,device):
    toks=torch.tensor([[AA_TO_IDX[a] for a in clean_sequence(s)] for s in df[seq_col]],dtype=torch.long); outs=[]; model.eval()
    for start in range(0,len(toks),256):
        x=toks[start:start+256].to(device); mu,_=model.encode(x); outs.append(mu.detach().cpu().numpy())
    return np.vstack(outs)
def score_generated_by_nearest_label(seqs,label_df,seq_col,score_col,wt):
    ref_seqs=label_df[seq_col].tolist(); ref_scores=pd.to_numeric(label_df[score_col],errors='coerce').fillna(0.0).values; rows=[]
    for s in seqs:
        d=np.array([hamming_distance(s,r) for r in ref_seqs]); j=int(d.argmin()); k=hamming_distance(s,wt); rows.append({'design_sequence':s,'score':float(ref_scores[j]-0.02*d[j]-0.01*max(k-25,0)),'nearest_label_distance':int(d[j]),'mutation_count':k})
    return pd.DataFrame(rows)
def train_round(model,pool,seq_col,score_col,center,scale,cfg,device,method,head=None,rfh=None):
    dl=DataLoader(LabelDataset(pool,seq_col,score_col,center,scale),batch_size=cfg['active_training'].get('batch_size',64),shuffle=True); params=list(model.parameters())+(list(head.parameters()) if head is not None else []); opt=torch.optim.AdamW(params,lr=cfg['active_training'].get('learning_rate',3e-4),weight_decay=1e-4)
    for _ in range(cfg['active_training'].get('epochs_per_round',2)):
        model.train();
        if head is not None: head.train()
        for x,y,w in dl:
            x,y,w=x.to(device),y.to(device),w.to(device); loss_recon,_,mu=vae_loss(model,x,mask_probability=0.1,sample_weight=w); loss_score=torch.tensor(0.0,device=device)
            if method=='trainable_head': loss_score=F.mse_loss(head(mu),y)
            elif method=='frozen_rfh': pred,_=rfh.predict_torch(mu); loss_score=F.mse_loss(pred,y)
            loss=loss_recon if method=='no_head_latent' else 0.55*loss_recon+0.45*loss_score; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
def make_candidates(model,pool,seq_col,score_col,wt,cfg,device,method,head=None,rfh=None):
    n=cfg['active_training'].get('generated_per_round',128); top=pool.sort_values(score_col,ascending=False).head(max(8,int(0.2*len(pool)))); z_top=encode_df(model,top,seq_col,device); rng=np.random.default_rng(cfg.get('seed',37)); idx=rng.choice(np.arange(len(z_top)),size=n,replace=True); z0=z_top[idx]+rng.normal(0,0.35,size=(n,z_top.shape[1])).astype(np.float32)
    if method=='no_head_latent': z_opt=z0
    else:
        objective=(lambda z: head(z)) if method=='trainable_head' else (lambda z: rfh.objective_torch(z)); z_opt=optimize_latents(z0,objective,steps=cfg['active_training'].get('latent_optimization_steps',20),lr=cfg['active_training'].get('latent_optimization_lr',0.05),l2_penalty=cfg['active_training'].get('latent_l2_penalty',0.01),device=device)
    seqs=decode_latents(model,torch.tensor(z_opt,dtype=torch.float32,device=device),temperature=cfg['generation'].get('temperature',0.9),mutation_budget_min=cfg['generation'].get('mutation_budget_min',3),mutation_budget_max=cfg['generation'].get('mutation_budget_max',25)); out=[]; seen=set()
    for s in seqs:
        h=sequence_hash(s)
        if h not in seen: seen.add(h); out.append(s)
    return out
def run_method(method,base_model,label_df,seq_col,score_col,wt,cfg,device):
    model=copy.deepcopy(base_model).to(device); pool=label_df.copy(); center=float(pool[score_col].median()); scale=float(np.median(np.abs(pool[score_col]-center))) or 1.0; z_dim=model.latent_dim; head=TrainablePerformanceHead(z_dim,cfg['trainable_head'].get('hidden_dim',128),cfg['trainable_head'].get('dropout',0.1)).to(device) if method=='trainable_head' else None; recs=[]; gens=[]
    for r in range(cfg['active_training'].get('rounds',2)):
        rfh=None
        if method=='frozen_rfh':
            z=encode_df(model,pool,seq_col,device); y=np.clip((pool[score_col].values-center)/(scale+1e-8),-4,4).astype(np.float32); rfh=RandomFrozenHeadEnsemble(z_dim,RFHConfig(n_heads=cfg['rfh'].get('n_heads',8),feature_dim=cfg['rfh'].get('feature_dim',256),ridge_alpha=cfg['rfh'].get('ridge_alpha',1.0),bootstrap_fraction=cfg['rfh'].get('bootstrap_fraction',0.8),uncertainty_penalty=cfg['rfh'].get('uncertainty_penalty',0.05),seed=cfg.get('seed',37)+r),device=device); rfh.fit(z,y)
        train_round(model,pool,seq_col,score_col,center,scale,cfg,device,method,head,rfh); cand=make_candidates(model,pool,seq_col,score_col,wt,cfg,device,method,head,rfh); scored=score_generated_by_nearest_label(cand,label_df,seq_col,score_col,wt); scored['method']=method; scored['round']=r; gens.append(scored); top20=scored.sort_values('score',ascending=False).head(min(20,len(scored))); rec={'method':method,'round':r,'n':len(scored),'mean_score':scored['score'].mean(),'max_score':scored['score'].max(),'top20_mean_score':top20['score'].mean(),'median_mutation_count':scored['mutation_count'].median()}; recs.append(rec); print(rec); add=scored.sort_values('score',ascending=False).head(cfg['active_training'].get('add_top_per_round',32)).copy(); add[seq_col]=add['design_sequence']; add[score_col]=add['score']; pool=pd.concat([pool,add[pool.columns.intersection(add.columns)]],ignore_index=True).drop_duplicates(seq_col)
    return pd.DataFrame(recs),pd.concat(gens,ignore_index=True),model
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--vae-checkpoint',required=True); p.add_argument('--labels',required=True); p.add_argument('--out',required=True); args=p.parse_args(); cfg=yaml.safe_load(open(args.config)); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); device='cuda' if torch.cuda.is_available() else 'cpu'; base,wt=load_vae(args.vae_checkpoint,device); seq_col=cfg.get('sequence_column','design_sequence'); score_col=cfg.get('score_column','active_learning_score'); labels=pd.read_csv(args.labels); labels[seq_col]=labels[seq_col].apply(clean_sequence); labels=labels[labels[seq_col].str.len()==len(wt)].drop_duplicates(seq_col).reset_index(drop=True); all_m=[]; all_g=[]
    for method in cfg.get('methods',['no_head_latent','trainable_head','frozen_rfh']):
        m,g,model=run_method(method,base,labels,seq_col,score_col,wt,cfg,device); all_m.append(m); all_g.append(g); torch.save({'model_state_dict':model.state_dict(),'method':method,'wt_sequence':wt},out/f'{method}_final_model.pt')
    metrics=pd.concat(all_m,ignore_index=True); gen=pd.concat(all_g,ignore_index=True); metrics.to_csv(out/'active_round_metrics.csv',index=False); gen.to_csv(out/'generated_candidates_by_round.csv',index=False); summary=metrics.groupby('method').agg(final_mean_score=('mean_score','last'),final_peak_score=('max_score','last'),final_top20_mean=('top20_mean_score','last')).reset_index(); summary.to_csv(out/'method_summary.csv',index=False); print(summary)
if __name__=='__main__': main()
