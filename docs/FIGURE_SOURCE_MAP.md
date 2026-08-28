# Figure-to-source map

Panel-level details are machine-readable in `PANEL_PROVENANCE.tsv`. The accompanying audit workbook records sample sizes, biological units, transformations, uncertainty, scripts, source outputs, and the correction made after peer review.

| Figure | Principal content | Public files | Inference boundary |
|---|---|---|---|
| 1 | Clinical cohort, CAP-LSM states, validation, longitudinal transitions, serum proxy | `data/clinical_aggregate/*`; `code/governed_clinical_transparency/rebuild_final_clinical_visuals.py` | Patient-level inputs are governed; the released serum association is exploratory and univariable |
| 2 | Replicated withdrawal geometry and descriptive resolver timing | `data/mouse_pseudobulk_derived/*`; `data/UQ_spatial_derived/S13_mouse_resolver_lock_trajectory/*` | GSE295293 uses nine mice; GSE305331 uses seven pooled condition libraries and has no replicate-level inference |
| 3 | Functional persistence atlas and categorical cross-compartment evidence | `data/figure_reconstruction/*`; public accessions in `RAW_DATA_ACCESSIONS.md` | Cell-level displays are descriptive; donor or animal is the biological unit in supporting analyses; no shared numeric evidence scale is asserted |
| 4 | HSC-retention coupling to human scar and descriptive CODEX concordance | `data/UQ_resolver_source_inputs/*`; `data/UQ_spatial_derived/S10_UQ_spatial_proteolytic_lock/*`; `data/figure_reconstruction/UQ_CODEX_scar_enrichment.tsv` | Visium stage tests aggregate to 32 patient clusters; CODEX contains one section per displayed stage |
| 5 | Retention-domain coalescence and repair-interface contraction | `data/UQ_spatial_derived/S11_UQ_phase_basin_control/*`; `data/UQ_spatial_derived/S14_spatial_domain_wall/*` | Display posterior 0.67 and topology threshold 0.50 have distinct roles; patient clusters are the stage-level unit |
| 6 | Static two-component spatial distribution and incomplete mouse return | `data/UQ_spatial_derived/S11_UQ_phase_basin_control/*`; `data/UQ_spatial_derived/S13_mouse_resolver_lock_trajectory/*` | The GMM is static; no physical potential, flow field, or hysteresis parameter was estimated |
| 7 | Evidence-coded candidate mechanism, in-silico perturbation, compound screen, and order-aware nomination | `data/mechanism_summary/mechanism_edge_evidence.tsv`; `data/LINCS_derived/*`; `data/UQ_spatial_derived/S12_sequential_basin_switch_drugs/*`; `figures/source_panels/*` | Edge styles distinguish established literature, project-supported association, and directly testable candidate links; the asymmetric scoring rule ranks role assignments and is not an observed treatment-order or efficacy estimate |

## Revised figure formats

All seven main figures are provided in PNG and PDF. Figures 1, 2, 5, and 7 also have editable SVG plates. The corrected analytical panels for Figures 3, 4, 6, and 7 are separately provided as editable SVG, PDF, and PNG under `figures/reviewer_revision/`; the four Figure 7 source panels are under `figures/source_panels/`.

## Numerical anchors

- Adult index cohort: 8,142 patients.
- Longitudinal state-flow cohort: 772 patients.
- Human UQ spatial analysis: 13,585 spots, 33 biopsy fragments, 32 patient clusters, and 589 spatial blocks.
- One- versus two-component BIC gain: 141.012.
- Largest retention-domain stage slope: +0.0960 per stage.
- Cross-state interface stage slope: -0.0287 per stage.
- Mouse withdrawal residual distance: 69.8% of peak-fibrosis displacement.
- Ezetimibe-to-verteporfin versus reverse order-aware scores: 0.892 versus 0.504; computational nomination only.
