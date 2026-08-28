import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests
import scanpy as sc

REPO_ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-h5ad',type=Path,default=REPO_ROOT/'results/GSE136103_atlas/GSE136103_human_liver_cell_state_labels_v1.h5ad')
parser.add_argument('--output-dir',type=Path,default=REPO_ROOT/'results/GSE136103_atlas')
args=parser.parse_args()
OUT=args.output_dir
OUT.mkdir(parents=True,exist_ok=True)
adata=sc.read_h5ad(args.input_h5ad)
X=adata.layers['counts'] if 'counts' in adata.layers else adata.X
X=X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)
symbols=adata.var['gene_symbol'].astype(str).values if 'gene_symbol' in adata.var.columns else adata.var_names.astype(str).values
rows=[]; meta=[]
for state in sorted(adata.obs['cell_state_v1'].astype(str).unique()):
    obs_s=adata.obs[adata.obs['cell_state_v1'].astype(str).values==state]
    for donor in sorted(obs_s['donor'].astype(str).unique()):
        idx=np.where((adata.obs['cell_state_v1'].astype(str).values==state) & (adata.obs['donor'].astype(str).values==donor))[0]
        if len(idx)<20:
            continue
        summed=np.asarray(X[idx].sum(axis=0)).ravel()
        rows.append(summed)
        disease=adata.obs.iloc[idx[0]]['disease']
        meta.append({'cell_state_v1':state,'donor':donor,'disease':disease,'n_cells':len(idx),'library_size':float(summed.sum())})
counts=np.vstack(rows)
meta=pd.DataFrame(meta)
lib=counts.sum(axis=1)
logcpm=np.log2((counts/lib[:,None])*1e6+1)
all_de=[]; sig_lines=[]
for state in sorted(meta['cell_state_v1'].unique()):
    mask=(meta['cell_state_v1']==state).values
    m=meta[mask].reset_index(drop=True)
    mat=logcpm[mask]
    cirr=(m['disease']=='cirrhotic').values
    heal=(m['disease']=='healthy').values
    if cirr.sum()<3 or heal.sum()<3:
        continue
    mean_c=mat[cirr].mean(axis=0); mean_h=mat[heal].mean(axis=0)
    lfc=mean_c-mean_h
    t,p=stats.ttest_ind(mat[cirr],mat[heal],axis=0,equal_var=False,nan_policy='omit')
    p=np.where(np.isfinite(p),p,1.0)
    fdr=multipletests(p,method='fdr_bh')[1]
    df=pd.DataFrame({'cell_state_v1':state,'gene_id':adata.var_names.astype(str),'gene_symbol':symbols,'log2fc_cirrhotic_vs_healthy':lfc,'mean_logcpm_cirrhotic':mean_c,'mean_logcpm_healthy':mean_h,'t_stat':t,'p_value':p,'fdr_bh':fdr,'n_cirrhotic_donors':int(cirr.sum()),'n_healthy_donors':int(heal.sum()),'n_cirrhotic_cells':int(m.loc[cirr,'n_cells'].sum()),'n_healthy_cells':int(m.loc[heal,'n_cells'].sum())})
    df=df.sort_values(['fdr_bh','log2fc_cirrhotic_vs_healthy'],ascending=[True,False])
    all_de.append(df)
    up=df[(df['log2fc_cirrhotic_vs_healthy']>0.25)&(df['mean_logcpm_cirrhotic']>1)].head(80)
    genes=[g for g in up['gene_symbol'].astype(str).tolist() if g and g.lower()!='nan']
    if genes:
        sig_lines.append(state.replace(' ','_')+'_CIRRHOTIC_UP\tGSE136103 cell-state donor-pseudobulk cirrhotic-up\t'+'\t'.join(genes))
res=pd.concat(all_de,ignore_index=True)
res.to_csv(OUT/'GSE136103_cell_state_pseudobulk_cirrhotic_vs_healthy.tsv',sep='\t',index=False)
meta.to_csv(OUT/'GSE136103_cell_state_pseudobulk_meta.tsv',sep='\t',index=False)
(OUT/'GSE136103_cell_state_cirrhotic_up_signatures_v1.gmt').write_text('\n'.join(sig_lines)+'\n')
summary=res.groupby('cell_state_v1').agg(n_tested=('gene_id','size'),top_gene=('gene_symbol','first'),top_log2fc=('log2fc_cirrhotic_vs_healthy','first'),top_fdr=('fdr_bh','first')).reset_index()
summary.to_csv(OUT/'GSE136103_cell_state_pseudobulk_de_summary.tsv',sep='\t',index=False)
js={'cell_states_tested':int(summary.shape[0]),'pseudobulk_profiles':int(meta.shape[0]),'signature_lines':len(sig_lines)}
(OUT/'GSE136103_cell_state_pseudobulk_de.summary.json').write_text(json.dumps(js,indent=2,sort_keys=True)+'\n')
print(summary.to_string(index=False))
print('SUMMARY', json.dumps(js, sort_keys=True))
