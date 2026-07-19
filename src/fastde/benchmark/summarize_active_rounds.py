import argparse
from pathlib import Path
import pandas as pd
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--out',required=True); args=p.parse_args(); df=pd.read_csv(args.input); rows=[]
    for method,sub in df.groupby('method'):
        sub=sub.sort_values('round'); rows.append({'method':method,'initial_mean':sub['mean_score'].iloc[0],'final_mean':sub['mean_score'].iloc[-1],'initial_peak':sub['max_score'].iloc[0],'final_peak':sub['max_score'].iloc[-1],'rounds':len(sub)})
    out=pd.DataFrame(rows); Path(args.out).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.out,index=False); print(out)
if __name__=='__main__': main()
