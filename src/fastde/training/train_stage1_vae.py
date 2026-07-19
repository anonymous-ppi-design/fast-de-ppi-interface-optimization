from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import Dataset, DataLoader
from fastde.data.sequence_utils import AA_TO_IDX, clean_sequence
from fastde.models.sparse_mutation_vae import SparseMutationVAE, VAEConfig, vae_loss
class SequenceDataset(Dataset):
    def __init__(self,df,seq_col): self.seqs=[clean_sequence(s) for s in df[seq_col].tolist()]; self.tokens=torch.tensor([[AA_TO_IDX[a] for a in s] for s in self.seqs],dtype=torch.long)
    def __len__(self): return len(self.tokens)
    def __getitem__(self,idx): return self.tokens[idx]
def infer_wt(df,seq_col):
    if 'pretrain_source' in df.columns:
        wt=df[df['pretrain_source'].astype(str).str.contains('WT',case=False,na=False)]
        if len(wt): return clean_sequence(wt.iloc[0][seq_col])
    if 'mutation_count' in df.columns:
        wt=df[pd.to_numeric(df['mutation_count'],errors='coerce').fillna(999)==0]
        if len(wt): return clean_sequence(wt.iloc[0][seq_col])
    return clean_sequence(df.iloc[0][seq_col])
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--input',required=True); p.add_argument('--out',required=True); args=p.parse_args(); cfg=yaml.safe_load(open(args.config)); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    seq_col=cfg.get('sequence_column','design_sequence'); df=pd.read_csv(args.input); df[seq_col]=df[seq_col].apply(clean_sequence); df=df[df[seq_col].str.len()>0].drop_duplicates(seq_col).reset_index(drop=True); wt=infer_wt(df,seq_col); L=len(wt); df=df[df[seq_col].str.len()==L].reset_index(drop=True)
    mc=cfg.get('model',{}); tc=cfg.get('training',{}); vae_cfg=VAEConfig(length=L,latent_dim=mc.get('latent_dim',64),hidden_dim=mc.get('hidden_dim',128),num_layers=mc.get('num_layers',2),dropout=mc.get('dropout',0.1),kl_beta=tc.get('kl_beta',1e-4),mutation_gate_weight=tc.get('mutation_gate_weight',0.25)); device='cuda' if tc.get('device','auto')=='auto' and torch.cuda.is_available() else 'cpu'; model=SparseMutationVAE(wt,vae_cfg).to(device)
    dl=DataLoader(SequenceDataset(df,seq_col),batch_size=tc.get('batch_size',64),shuffle=True); opt=torch.optim.AdamW(model.parameters(),lr=tc.get('learning_rate',1e-3),weight_decay=tc.get('weight_decay',1e-4)); hist=[]
    for epoch in range(1,tc.get('epochs',2)+1):
        losses=[]; model.train()
        for x in dl:
            x=x.to(device); loss,_,_=vae_loss(model,x,mask_probability=tc.get('mask_probability',0.15)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        row={'epoch':epoch,'loss':float(np.mean(losses))}; hist.append(row); print(row)
    torch.save({'model_state_dict':model.state_dict(),'vae_config':vae_cfg.__dict__,'wt_sequence':wt,'seq_col':seq_col},out/'stage1_vae.pt'); pd.DataFrame(hist).to_csv(out/'stage1_training_history.csv',index=False); print('Saved Stage-1 VAE:', out/'stage1_vae.pt')
if __name__=='__main__': main()
