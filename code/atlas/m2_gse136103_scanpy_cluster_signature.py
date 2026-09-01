import argparse
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy import sparse, stats
import scanpy as sc
import harmonypy as hm
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-h5ad', type=Path, default=REPO_ROOT/'results/GSE136103_atlas/GSE136103_human_liver_filtered_qc_v1.h5ad')
parser.add_argument('--output-dir', type=Path, default=REPO_ROOT/'results/GSE136103_atlas')
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()
sc.settings.n_jobs = args.threads
OUT = args.output_dir
IN = args.input_h5ad
OUT.mkdir(parents=True, exist_ok=True)
print('READ', IN, flush=True)
t0 = time.time()
adata = sc.read_h5ad(IN)
print(f'LOADED cells={adata.n_obs} genes={adata.n_vars} sec={time.time()-t0:.1f}', flush=True)

# Donor-level pseudobulk disease signature from raw counts before normalization.
print('PSEUDOBULK_START', flush=True)
Xraw = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
donors = adata.obs[['donor','disease']].drop_duplicates().sort_values(['disease','donor'])
rows = []
labels = []
for _, row in donors.iterrows():
    mask = (adata.obs['donor'].values == row['donor'])
    s = np.asarray(Xraw[mask].sum(axis=0)).ravel()
    rows.append(s)
    labels.append((row['donor'], row['disease'], int(mask.sum())))
counts = np.vstack(rows)
lib = counts.sum(axis=1)
logcpm = np.log2((counts / lib[:, None]) * 1e6 + 1)
disease_arr = np.array([x[1] for x in labels])
cirr = disease_arr == 'cirrhotic'
healthy = disease_arr == 'healthy'
mean_c = logcpm[cirr].mean(axis=0)
mean_h = logcpm[healthy].mean(axis=0)
logfc = mean_c - mean_h
tstat, pval = stats.ttest_ind(logcpm[cirr], logcpm[healthy], axis=0, equal_var=False, nan_policy='omit')
pval = np.where(np.isfinite(pval), pval, 1.0)
fdr = multipletests(pval, method='fdr_bh')[1]
var = adata.var.copy()
gene_symbol = var['gene_symbol'].astype(str).values if 'gene_symbol' in var.columns else var.index.astype(str).values
sig = pd.DataFrame({
    'gene_id': adata.var_names.astype(str),
    'gene_symbol': gene_symbol,
    'mean_logcpm_cirrhotic': mean_c,
    'mean_logcpm_healthy': mean_h,
    'log2fc_cirrhotic_vs_healthy': logfc,
    't_stat': tstat,
    'p_value': pval,
    'fdr_bh': fdr,
})
sig = sig.sort_values(['fdr_bh','log2fc_cirrhotic_vs_healthy'], ascending=[True, False])
sig.to_csv(OUT/'GSE136103_donor_pseudobulk_cirrhotic_vs_healthy.tsv', sep='\t', index=False)
up = sig[(sig['log2fc_cirrhotic_vs_healthy'] > 0.25) & (sig['mean_logcpm_cirrhotic'] > 1)].copy()
if len(up) < 50:
    up = sig[sig['log2fc_cirrhotic_vs_healthy'] > 0].copy()
up = up.sort_values(['fdr_bh','log2fc_cirrhotic_vs_healthy'], ascending=[True, False]).head(150)
up.to_csv(OUT/'fibrotic_niche_signature_v1_genes.tsv', sep='\t', index=False)
genes_for_gmt = [g for g in up['gene_symbol'].astype(str).tolist() if g and g.lower() != 'nan']
(OUT/'fibrotic_niche_signature_v1.gmt').write_text('FIBROTIC_NICHE_GSE136103_V1\tGSE136103 donor-pseudobulk cirrhotic-up human liver signature\t' + '\t'.join(genes_for_gmt) + '\n')
pb_meta = pd.DataFrame(labels, columns=['donor','disease','n_cells'])
pb_meta.to_csv(OUT/'GSE136103_donor_pseudobulk_meta.tsv', sep='\t', index=False)
print(f'PSEUDOBULK_DONE up_genes={len(up)} gmt_genes={len(genes_for_gmt)}', flush=True)

# Standard single-cell embedding and clustering.
print('SCANPY_NORMALIZE', flush=True)
adata.layers['counts'] = Xraw.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
print('HVG', flush=True)
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000, batch_key='sample_prefix', layer='counts')
adata_hvg = adata[:, adata.var['highly_variable']].copy()
print(f'HVG_SELECTED {adata_hvg.n_vars}', flush=True)
sc.pp.scale(adata_hvg, max_value=10)
print('PCA', flush=True)
sc.tl.pca(adata_hvg, n_comps=50, svd_solver='arpack')
print('HARMONY', flush=True)
ho = hm.run_harmony(adata_hvg.obsm['X_pca'], adata_hvg.obs, ['sample_prefix'], max_iter_harmony=20)
Z = ho.Z_corr
if Z.shape[0] == adata_hvg.n_obs:
    adata_hvg.obsm['X_pca_harmony'] = Z
elif Z.shape[1] == adata_hvg.n_obs:
    adata_hvg.obsm['X_pca_harmony'] = Z.T
else:
    raise ValueError(f'Harmony output shape {Z.shape} incompatible with n_obs={adata_hvg.n_obs}')
print(f'HARMONY_SHAPE {adata_hvg.obsm["X_pca_harmony"].shape}', flush=True)
print('NEIGHBORS_UMAP_LEIDEN', flush=True)
sc.pp.neighbors(adata_hvg, n_neighbors=20, n_pcs=40, use_rep='X_pca_harmony')
sc.tl.umap(adata_hvg, min_dist=0.35, spread=1.0)
sc.tl.leiden(adata_hvg, resolution=0.8, flavor='igraph', n_iterations=2, key_added='leiden_r08')
# Copy clustering and embeddings back into full object, keeping memory moderate.
adata.obs['leiden_r08'] = adata_hvg.obs['leiden_r08'].astype(str).values
adata.obsm['X_pca_harmony'] = adata_hvg.obsm['X_pca_harmony']
adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
adata.uns['neighbors'] = adata_hvg.uns['neighbors']
adata.obsp['distances'] = adata_hvg.obsp['distances']
adata.obsp['connectivities'] = adata_hvg.obsp['connectivities']
adata.uns['leiden'] = adata_hvg.uns.get('leiden', {})
adata.uns['m2_signature_note'] = 'fibrotic_niche_signature_v1.gmt from donor-level pseudobulk cirrhotic vs healthy, not yet cell-type refined.'
print('COMPOSITION', flush=True)
comp = adata.obs.groupby(['leiden_r08','disease','sort_fraction'], observed=True).size().reset_index(name='n_cells')
comp.to_csv(OUT/'GSE136103_leiden_r08_composition.tsv', sep='\t', index=False)
cluster_summary = adata.obs.groupby('leiden_r08', observed=True).agg(
    n_cells=('disease','size'),
    pct_cirrhotic=('disease', lambda x: float((x=='cirrhotic').mean()*100)),
    n_donors=('donor', lambda x: int(pd.Series(x).nunique())),
).reset_index()
cluster_summary.to_csv(OUT/'GSE136103_leiden_r08_cluster_summary.tsv', sep='\t', index=False)
print('MARKERS', flush=True)
sc.tl.rank_genes_groups(adata, groupby='leiden_r08', method='wilcoxon', pts=True, n_genes=80)
marker_rows = []
rg = adata.uns['rank_genes_groups']
groups = rg['names'].dtype.names
symbol_map = adata.var['gene_symbol'].astype(str).to_dict() if 'gene_symbol' in adata.var.columns else {}
for g in groups:
    names = rg['names'][g]
    for rank, gene_id in enumerate(names, 1):
        marker_rows.append({
            'cluster': g, 'rank': rank, 'gene_id': gene_id,
            'gene_symbol': symbol_map.get(gene_id, gene_id),
            'score': float(rg['scores'][g][rank-1]),
            'logfoldchanges': float(rg['logfoldchanges'][g][rank-1]) if 'logfoldchanges' in rg else np.nan,
            'pvals_adj': float(rg['pvals_adj'][g][rank-1]) if 'pvals_adj' in rg else np.nan,
        })
pd.DataFrame(marker_rows).to_csv(OUT/'GSE136103_leiden_r08_markers_top80.tsv', sep='\t', index=False)
print('WRITE_H5AD', flush=True)
adata.write_h5ad(OUT/'GSE136103_human_liver_scanpy_clustered_v1.h5ad', compression='gzip')
summary = {
    'input': str(IN),
    'cells': int(adata.n_obs),
    'genes': int(adata.n_vars),
    'hvg': int(adata_hvg.n_vars),
    'clusters_leiden_r08': int(adata.obs['leiden_r08'].nunique()),
    'signature_genes': int(len(genes_for_gmt)),
    'elapsed_sec': round(time.time()-t0, 1),
}
(OUT/'GSE136103_scanpy_cluster_signature_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print('DONE', json.dumps(summary, sort_keys=True), flush=True)
