# Workflow entry points

All commands are run from the repository root. Public raw datasets are intentionally omitted and should be placed under `data/raw_external/` or supplied through explicit command-line paths.

## Environment and validation

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python environment/requirements-lock.txt
.venv/bin/python scripts/validate_release.py --rebuild
```

## GSE295293 mouse progression and withdrawal

```bash
.venv/bin/python code/mouse/GSE295293_prepare_pseudobulk.py \
  --archive data/raw_external/GSE295293/GSE295293_RAW.tar \
  --work-dir work/GSE295293 \
  --output-dir results/GSE295293_dynamic_regression

Rscript code/mouse/GSE295293_dynamic_pseudobulk.R \
  --input-dir work/GSE295293 \
  --output-dir results/GSE295293_dynamic_regression

.venv/bin/python code/mouse/mouse_resolver_lock_trajectory.py \
  --counts work/GSE295293/pseudobulk_counts.tsv.gz \
  --metadata work/GSE295293/pseudobulk_metadata.tsv \
  --pooled-trajectories results/GSE305331_state_trajectories/state_module_trajectories.tsv \
  --output-dir results/S13_mouse_resolver_retention_trajectory
```

The inferential unit for GSE295293 is the mouse. GSE305331 contains one pooled library per condition and supports descriptive timing only.

## GSE244832 human HSC donor pseudobulk

```bash
.venv/bin/python code/human_hsc/GSE244832_prepare_hsc_map.py \
  --source-dir data/raw_external/GSE244832 \
  --output-dir work/GSE244832_hsc_pseudobulk

c++ -O3 -std=c++17 code/human_hsc/GSE244832_stream_hsc_pseudobulk.cpp -o work/stream_hsc
gzip -cd data/raw_external/GSE244832/hLIVER_counts.mtx.gz | work/stream_hsc \
  work/GSE244832_hsc_pseudobulk/cell_map.tsv \
  work/GSE244832_hsc_pseudobulk/genes.tsv \
  work/GSE244832_hsc_pseudobulk/hsc_pseudobulk_counts.tsv \
  work/GSE244832_hsc_pseudobulk/stream_audit.tsv

Rscript code/human_hsc/GSE244832_hsc_donor_edger.R \
  --input-dir work/GSE244832_hsc_pseudobulk \
  --output-dir results/GSE244832_hsc_donor
```

## GSE136103 atlas

```bash
.venv/bin/python code/atlas/m2_gse136103_build_h5ad.py \
  --raw-dir data/raw_external/GSE136103 \
  --output-dir results/GSE136103_atlas

.venv/bin/python code/atlas/m2_gse136103_scanpy_cluster_signature.py \
  --input-h5ad results/GSE136103_atlas/GSE136103_human_liver_filtered_qc_v1.h5ad \
  --output-dir results/GSE136103_atlas \
  --threads 8

.venv/bin/python code/atlas/m2_gse136103_cell_state_label_v1.py \
  --input-h5ad results/GSE136103_atlas/GSE136103_human_liver_scanpy_clustered_v1.h5ad \
  --output-dir results/GSE136103_atlas

.venv/bin/python code/atlas/m2_gse136103_cell_state_pseudobulk_de.py \
  --input-h5ad results/GSE136103_atlas/GSE136103_human_liver_cell_state_labels_v1.h5ad \
  --output-dir results/GSE136103_atlas
```

Cell-level embeddings are descriptive. Disease comparisons aggregate by donor.

## UQ Visium and spatial retention domains

```bash
.venv/bin/python code/spatial/UQ_Visium_component_audit.py --help
.venv/bin/python code/spatial/UQ_Visium_resolver_window_analysis.py --help

.venv/bin/python code/spatial/uq_phase_basin_control.py \
  --input data/UQ_spatial_derived/S10_UQ_spatial_proteolytic_lock/UQ_proteolytic_lock_spot_scores.parquet \
  --output-dir results/S11_UQ_static_component_control

.venv/bin/python code/spatial/spatial_domain_wall_resolver_window.py \
  --input results/S11_UQ_static_component_control/phase_spot_states.parquet \
  --output-dir results/S14_spatial_retention_domains

.venv/bin/python code/spatial/UQ_Visium_component_audit.py \
  --visium-root data/raw_external/UQ_Visium \
  --results-dir data/UQ_spatial_mapping_derived \
  --figures-dir results/S15_UQ_biopsy_mapping_sensitivity/geometry

.venv/bin/python code/spatial/UQ_biopsy_mapping_sensitivity.py \
  --phase-spots data/UQ_spatial_derived/S11_UQ_phase_basin_control/phase_spot_states.parquet \
  --component-map data/UQ_spatial_mapping_derived/UQ_Visium_spot_component_map.parquet \
  --biopsy-map docs/methods_audit/machine_readable/UQ_Biopsy_Reconstruction_Map.tsv \
  --output-dir results/S15_UQ_biopsy_mapping_sensitivity
```

The two-component model is a static distributional model. Retention-domain topology uses the posterior threshold and spatial adjacency rules recorded in panel provenance. The mapping audit assigns connected components from array coordinates only; score columns do not enter biopsy assignment. Its stress tests remove fragment rescues, split geometric components, exclude inferred or merged fields, and leave out each array.

## LINCS order-aware nomination

```bash
.venv/bin/python code/perturbation/sequential_basin_switch_drugs.py \
  --input data/LINCS_derived/LINCS_name_collapsed_hard_gated.tsv \
  --output-dir results/S12_order_aware_drug_nomination
```

The asymmetric score nominates an equal-exposure factorial experiment. scGPT and CellOracle use dedicated environments documented under `environment/` and are excluded from the core lock.

## Evidence-coded mechanism and main SVG plates

```bash
.venv/bin/python code/figures/build_mechanism_circuit.py
.venv/bin/python code/figures/rebuild_main_figure_plates.py
```

The mechanism circuit reads `data/mechanism_summary/mechanism_edge_evidence.tsv`. Solid links are literature-established, dashed links are project-supported associations, and dotted links are candidates for direct testing. The plate builder applies the manuscript terminology and composes Figure 7 from the four editable source panels.
