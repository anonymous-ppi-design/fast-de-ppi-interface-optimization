from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
def positive_part(x): return np.maximum(pd.to_numeric(x,errors='coerce').fillna(0.0).astype(float),0.0)
def compute_active_score(df, ddg_binding_col='ddG_binding_vs_WT', ddg_stability_col='ddG_stability_vs_WT', clash_col='clash_increase_vs_WT', mutation_count_col='mutation_count', binding_weight=1.0, stability_penalty_weight=0.5, clash_penalty_weight=0.3, mutation_penalty_weight=0.05):
    return (-binding_weight*pd.to_numeric(df[ddg_binding_col],errors='coerce').fillna(0.0) - stability_penalty_weight*positive_part(df[ddg_stability_col]) - clash_penalty_weight*positive_part(df[clash_col]) - mutation_penalty_weight*pd.to_numeric(df[mutation_count_col],errors='coerce').fillna(0.0))
def wt_center_score(df, score_col='active_learning_score_raw', is_wt_col='is_WT'):
    mask=df[is_wt_col].astype(str).str.lower().isin(['true','1']) if is_wt_col in df.columns else pd.Series(False,index=df.index); wt=float(df.loc[mask,score_col].iloc[0]) if mask.any() else 0.0; return df[score_col]-wt
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--config'); p.add_argument('--out',required=True); args=p.parse_args(); cfg={}
    if args.config:
        with open(args.config) as f: cfg=yaml.safe_load(f) or {}
    cols=cfg.get('columns',{}); sc=cfg.get('score',{}); df=pd.read_csv(args.input)
    df['active_learning_score_raw']=compute_active_score(df, cols.get('ddg_binding','ddG_binding_vs_WT'), cols.get('ddg_stability','ddG_stability_vs_WT'), cols.get('clash_increase','clash_increase_vs_WT'), cols.get('mutation_count','mutation_count'), sc.get('binding_weight',1.0), sc.get('stability_penalty_weight',0.5), sc.get('clash_penalty_weight',0.3), sc.get('mutation_penalty_weight',0.05))
    df['active_learning_score']=wt_center_score(df,'active_learning_score_raw',cols.get('is_wt','is_WT')) if sc.get('center_wt_to_zero',True) else df['active_learning_score_raw']
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.out,index=False); print('Saved scored table:',args.out); print(df[['active_learning_score']].describe())
if __name__=='__main__': main()
