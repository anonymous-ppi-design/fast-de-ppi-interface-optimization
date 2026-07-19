from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
REQUIRED_COLUMNS=['candidate_id','design_sequence','mutation_count','ddG_binding_vs_WT','ddG_stability_vs_WT','clash_increase_vs_WT']
def validate_foldx_table(df): return [c for c in REQUIRED_COLUMNS if c not in df.columns]
def read_foldx_table(path):
    df=pd.read_csv(path); missing=validate_foldx_table(df)
    if missing: raise ValueError(f'FoldX pseudo-label table is missing columns: {missing}')
    return df
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--out',required=True); args=p.parse_args(); df=read_foldx_table(args.input); Path(args.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.out,index=False); print(f'Validated FoldX table saved to {args.out}')
if __name__=='__main__': main()
