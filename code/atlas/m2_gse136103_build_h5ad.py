import argparse
from pathlib import Path
import gzip, re, csv, json, time
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
import anndata as ad

REPO_ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw-dir', type=Path, default=REPO_ROOT/'data/raw_external/GSE136103')
parser.add_argument('--output-dir', type=Path, default=REPO_ROOT/'results/GSE136103_atlas')
args = parser.parse_args()
RAW = args.raw_dir
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)

def parse_prefix(prefix: str):
    parts = prefix.split('_')
    gsm = parts[0]
    rest = parts[1:]
    meta = {'gsm': gsm, 'sample_prefix': prefix, 'dataset': 'GSE136103'}
    if len(rest) >= 2 and re.match(r'^(healthy|cirrhotic)\d+$', rest[0]):
        donor = rest[0]
        disease = 'healthy' if donor.startswith('healthy') else 'cirrhotic'
        meta.update({
            'analysis_class': 'human_liver',
            'species': 'human',
            'tissue': 'liver',
            'disease': disease,
            'donor': donor,
            'donor_num': re.sub(r'^(healthy|cirrhotic)', '', donor),
            'sort_fraction': '_'.join(rest[1:]),
            'include_m2_primary': True,
        })
    elif len(rest) == 1 and re.match(r'^blood\d+$', rest[0]):
        meta.update({
            'analysis_class': 'human_blood_reference',
            'species': 'human',
            'tissue': 'blood',
            'disease': 'reference_blood',
            'donor': rest[0],
            'donor_num': re.sub(r'^blood', '', rest[0]),
            'sort_fraction': 'unfractionated',
            'include_m2_primary': False,
        })
    elif len(rest) >= 2 and rest[0] == 'mouse':
        meta.update({
            'analysis_class': 'mouse_liver_reference',
            'species': 'mouse',
            'tissue': 'liver',
            'disease': 'mouse_' + '_'.join(rest[1:]),
            'donor': 'mouse_' + '_'.join(rest[1:]),
            'donor_num': '',
            'sort_fraction': 'unspecified',
            'include_m2_primary': False,
        })
    else:
        meta.update({
            'analysis_class': 'unknown', 'species': 'unknown', 'tissue': 'unknown',
            'disease': 'unknown', 'donor': '_'.join(rest), 'donor_num': '',
            'sort_fraction': 'unknown', 'include_m2_primary': False,
        })
    return meta

def count_lines_gz(path: Path):
    n = 0
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for _ in fh:
            n += 1
    return n

triplets = {}
for f in RAW.glob('*.gz'):
    name = f.name
    for kind, suffix in [('barcodes','_barcodes.tsv.gz'), ('genes','_genes.tsv.gz'), ('matrix','_matrix.mtx.gz')]:
        if name.endswith(suffix):
            prefix = name[:-len(suffix)]
            triplets.setdefault(prefix, {})[kind] = f
            break

rows = []
for prefix, d in sorted(triplets.items()):
    meta = parse_prefix(prefix)
    missing = sorted(set(['barcodes','genes','matrix']) - set(d))
    n_b = count_lines_gz(d['barcodes']) if 'barcodes' in d else None
    n_g = count_lines_gz(d['genes']) if 'genes' in d else None
    m_rows = m_cols = m_nnz = None
    if 'matrix' in d:
        with gzip.open(d['matrix'], 'rt', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.startswith('%'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    m_rows, m_cols, m_nnz = map(int, parts[:3]); break
    row = {**meta,
        'barcodes_file': d.get('barcodes', Path('')).name,
        'genes_file': d.get('genes', Path('')).name,
        'matrix_file': d.get('matrix', Path('')).name,
        'missing_files': ','.join(missing),
        'n_barcodes': n_b, 'n_genes_file_lines': n_g,
        'matrix_n_genes': m_rows, 'matrix_n_cells': m_cols, 'matrix_nnz': m_nnz,
        'genes_match': bool(n_g == m_rows) if n_g is not None and m_rows is not None else False,
        'barcodes_match': bool(n_b == m_cols) if n_b is not None and m_cols is not None else False,
    }
    rows.append(row)

pd.DataFrame(rows).to_csv(OUT/'GSE136103_raw_sample_audit_all26.tsv', sep='\t', index=False)
summary = {
    'n_samples_total': len(rows),
    'n_primary_human_liver_samples': sum(r['include_m2_primary'] for r in rows),
    'n_human_blood_reference_samples': sum(r['analysis_class']=='human_blood_reference' for r in rows),
    'n_mouse_reference_samples': sum(r['analysis_class']=='mouse_liver_reference' for r in rows),
    'primary_cells_by_barcodes': sum(int(r['n_barcodes']) for r in rows if r['include_m2_primary']),
    'all_triplets_complete': all(not r['missing_files'] for r in rows),
    'all_dimensions_match': all(r['genes_match'] and r['barcodes_match'] for r in rows),
}
(OUT/'GSE136103_raw_sample_audit_all26.summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
print('AUDIT_SUMMARY', json.dumps(summary, sort_keys=True))

primary = [r for r in rows if r['include_m2_primary']]
adatas = []
for i, r in enumerate(primary, 1):
    t0 = time.time()
    prefix = r['sample_prefix']
    print(f'READ_SAMPLE {i}/{len(primary)} {prefix}', flush=True)
    genes_path = RAW / r['genes_file']
    barcodes_path = RAW / r['barcodes_file']
    matrix_path = RAW / r['matrix_file']
    genes = pd.read_csv(genes_path, sep='\t', header=None, names=['gene_id','gene_symbol'])
    barcodes = pd.read_csv(barcodes_path, sep='\t', header=None, names=['barcode'])
    X = mmread(str(matrix_path)).tocsr().T.tocsr()
    obs = pd.DataFrame({
        'barcode': barcodes['barcode'].astype(str).values,
        'sample_prefix': prefix,
        'gsm': r['gsm'],
        'donor': r['donor'],
        'donor_num': r['donor_num'],
        'disease': r['disease'],
        'sort_fraction': r['sort_fraction'],
        'dataset': 'GSE136103',
    })
    obs.index = [f'{prefix}:{bc}' for bc in obs['barcode']]
    var = genes.copy()
    var.index = var['gene_id'].astype(str)
    a = ad.AnnData(X=X, obs=obs, var=var)
    adatas.append(a)
    print(f'DONE_SAMPLE {prefix} cells={a.n_obs} genes={a.n_vars} nnz={a.X.nnz} sec={time.time()-t0:.1f}', flush=True)

print('CONCAT_START', flush=True)
adata = ad.concat(adatas, join='inner', label='sample_prefix_from_concat', keys=[a.obs['sample_prefix'][0] for a in adatas], merge='same')
adata.var['mt'] = adata.var['gene_symbol'].astype(str).str.upper().str.startswith('MT-').values
X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
adata.obs['n_counts'] = np.asarray(X.sum(axis=1)).ravel()
adata.obs['n_genes_by_counts'] = np.asarray((X > 0).sum(axis=1)).ravel()
mt_mask = adata.var['mt'].values
if mt_mask.any():
    mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel()
    adata.obs['pct_counts_mt'] = np.divide(mt_counts, adata.obs['n_counts'].values, out=np.zeros(adata.n_obs), where=adata.obs['n_counts'].values>0) * 100
else:
    adata.obs['pct_counts_mt'] = 0.0
adata.uns['build_note'] = 'GSE136103 human liver primary samples only; blood and mouse reference samples excluded from this h5ad and retained in audit table.'
adata.uns['raw_audit_summary'] = summary
raw_path = OUT/'GSE136103_human_liver_raw_counts_qc.h5ad'
adata.write_h5ad(raw_path, compression='gzip')
qc = adata.obs.groupby(['disease','sort_fraction'], observed=True).agg(
    n_cells=('n_counts','size'),
    median_counts=('n_counts','median'),
    median_genes=('n_genes_by_counts','median'),
    median_pct_mt=('pct_counts_mt','median'),
).reset_index()
qc.to_csv(OUT/'GSE136103_human_liver_qc_by_group.tsv', sep='\t', index=False)
filter_mask = (adata.obs['n_genes_by_counts'] >= 200) & (adata.obs['n_genes_by_counts'] <= 6000) & (adata.obs['pct_counts_mt'] <= 25)
adata_f = adata[filter_mask].copy()
adata_f.uns['filter_rule'] = 'n_genes_by_counts>=200, <=6000, pct_counts_mt<=25; thresholds are first-pass for downstream review.'
filtered_path = OUT/'GSE136103_human_liver_filtered_qc_v1.h5ad'
adata_f.write_h5ad(filtered_path, compression='gzip')
filter_summary = {
    'raw_cells': int(adata.n_obs), 'raw_genes': int(adata.n_vars),
    'filtered_cells': int(adata_f.n_obs), 'filtered_genes': int(adata_f.n_vars),
    'retained_fraction': float(adata_f.n_obs / adata.n_obs),
    'output_raw_h5ad': str(raw_path), 'output_filtered_h5ad': str(filtered_path),
}
(OUT/'GSE136103_human_liver_filter_summary.json').write_text(json.dumps(filter_summary, indent=2, sort_keys=True)+'\n')
print('FILTER_SUMMARY', json.dumps(filter_summary, sort_keys=True))
