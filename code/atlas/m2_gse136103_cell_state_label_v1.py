import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
import scanpy as sc

REPO_ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-h5ad',type=Path,default=REPO_ROOT/'results/GSE136103_atlas/GSE136103_human_liver_scanpy_clustered_v1.h5ad')
parser.add_argument('--output-dir',type=Path,default=REPO_ROOT/'results/GSE136103_atlas')
args=parser.parse_args()
OUT=args.output_dir
ADATA=args.input_h5ad
OUT.mkdir(parents=True,exist_ok=True)
adata=sc.read_h5ad(ADATA)
marker_sets={
 'Hepatocyte':['ALB','APOA1','APOB','TTR','TF','CYP3A4','CYP2E1','HP'],
 'Cholangiocyte':['KRT19','KRT7','EPCAM','SOX9','MUC1','CFTR'],
 'Endothelial':['PECAM1','VWF','KDR','FLT1','ENG','RAMP2','PLVAP','CLEC4G','STAB2'],
 'Lymphatic_endothelial':['PROX1','PDPN','LYVE1','CCL21','FLT4'],
 'HSC_Myofibroblast':['COL1A1','COL1A2','COL3A1','COL6A3','DCN','LUM','RGS5','ACTA2','TAGLN','PDGFRB','COL15A1'],
 'Macrophage_Kupffer':['CD68','MARCO','C1QA','C1QB','C1QC','LST1','FCGR3A','TYROBP','TREM2','SPP1'],
 'Monocyte_inflammatory':['S100A8','S100A9','FCN1','LYZ','VCAN','LST1','IL1B'],
 'T_NK':['CD3D','CD3E','TRAC','NKG7','GNLY','PRF1','GZMB','IL7R'],
 'B_cell':['MS4A1','CD79A','CD79B','BANK1','CD74'],
 'Plasma_cell':['MZB1','JCHAIN','XBP1','IGHG1','IGKC','SDC1'],
 'Mast_cell':['TPSAB1','TPSB2','CPA3','KIT','MS4A2'],
 'Cycling':['MKI67','TOP2A','STMN1','HMGB2','UBE2C'],
 'Erythroid':['HBB','HBA1','HBA2','ALAS2'],
}
# Gene symbol to var index map; duplicates collapsed by first occurrence.
symbols=adata.var['gene_symbol'].astype(str) if 'gene_symbol' in adata.var.columns else pd.Series(adata.var_names,index=adata.var_names)
sym_to_ids={}
for gid,sym in symbols.items():
    sym_to_ids.setdefault(sym.upper(), gid)
X=adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
clusters=sorted(adata.obs['leiden_r08'].astype(str).unique(), key=lambda x:int(x) if x.isdigit() else x)
rows=[]
for cl in clusters:
    mask=(adata.obs['leiden_r08'].astype(str).values==cl)
    for label,markers in marker_sets.items():
        ids=[sym_to_ids[m.upper()] for m in markers if m.upper() in sym_to_ids]
        if ids:
            idx=[adata.var_names.get_loc(i) for i in ids]
            score=float(np.asarray(X[mask][:,idx].mean(axis=1)).ravel().mean())
            frac=float(np.asarray((X[mask][:,idx]>0).mean(axis=0)).ravel().mean())
        else:
            score=np.nan; frac=np.nan
        rows.append({'cluster':cl,'marker_label':label,'n_markers_present':len(ids),'mean_logexpr_score':score,'mean_marker_fraction':frac,'markers_present':';'.join([symbols.loc[i] for i in ids])})
scores=pd.DataFrame(rows)
scores.to_csv(OUT/'GSE136103_cluster_marker_panel_scores_v1.tsv',sep='\t',index=False)
# Assign top label by mean score; require at least two markers except rare/erythroid panels where one strong marker can dominate.
assign=[]
comp=pd.read_csv(OUT/'GSE136103_leiden_r08_cluster_summary.tsv',sep='\t')
if 'leiden_r08' in comp.columns:
    comp['leiden_r08']=comp['leiden_r08'].astype(str)
if 'cluster' in comp.columns:
    comp['cluster']=comp['cluster'].astype(str)
for cl in clusters:
    s=scores[(scores.cluster==cl) & (scores.n_markers_present>0)].sort_values(['mean_logexpr_score','mean_marker_fraction'],ascending=False)
    top=s.iloc[0]
    second=s.iloc[1] if len(s)>1 else top
    assigned=top['marker_label']
    confidence='medium'
    if top['mean_logexpr_score']>=1.0 and (top['mean_logexpr_score']-second['mean_logexpr_score'])>=0.20:
        confidence='high'
    elif top['mean_logexpr_score']<0.35:
        confidence='low'
        assigned='Unresolved_'+str(assigned)
    row={'cluster':cl,'cell_state_v1':assigned,'label_confidence':confidence,'top_score':top['mean_logexpr_score'],'second_label':second['marker_label'],'second_score':second['mean_logexpr_score']}
    assign.append(row)
assign=pd.DataFrame(assign)
if 'leiden_r08' in comp.columns:
    assign=assign.merge(comp, how='left', left_on='cluster', right_on='leiden_r08').drop(columns=['leiden_r08'])
else:
    assign=assign.merge(comp, how='left', on='cluster')
assign['fibrotic_niche_candidate']=(assign['pct_cirrhotic'].fillna(0)>=60) & (assign['cell_state_v1'].str.contains('HSC|Macrophage|Endothelial|Cholangiocyte|Lymphatic',regex=True))
assign.to_csv(OUT/'cell_state_labels.tsv',sep='\t',index=False)
assign[assign['fibrotic_niche_candidate']].to_csv(OUT/'fibrotic_niche_clusters_v1.tsv',sep='\t',index=False)
label_map=assign.set_index('cluster')['cell_state_v1'].to_dict()
conf_map=assign.set_index('cluster')['label_confidence'].to_dict()
adata.obs['cell_state_v1']=adata.obs['leiden_r08'].astype(str).map(label_map).astype('category')
adata.obs['cell_state_v1_confidence']=adata.obs['leiden_r08'].astype(str).map(conf_map).astype('category')
adata.uns['cell_state_label_note']='Rule-based marker panel v1 from canonical liver/immune markers; inspect before manuscript-grade cell type claims.'
adata.write_h5ad(OUT/'GSE136103_human_liver_cell_state_labels_v1.h5ad',compression='gzip')
summary={'clusters':len(assign),'fibrotic_niche_candidate_clusters':int(assign['fibrotic_niche_candidate'].sum()),'cell_states':sorted(assign['cell_state_v1'].unique().tolist())}
(OUT/'cell_state_labels.summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(assign.sort_values('cluster').to_string(index=False))
print('SUMMARY', json.dumps(summary, sort_keys=True))
