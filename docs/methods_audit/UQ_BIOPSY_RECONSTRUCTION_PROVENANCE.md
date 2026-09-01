# UQ biopsy reconstruction provenance

## Source record

- The source paper reports 33 Visium liver biopsies from 32 patients, including two synchronous cores from one patient.
- The published Supplementary Figure S1E displays all biopsy-shaped spot fields coloured by fibrosis stage. The local source is `mmc1.pdf`, page 14, with the panel legend on page 15.
- The source `QC and normalisation` script loads eight arrays and reads one per-spot `*_samples.csv` file per array before any module scores are calculated. It then assigns the published fibrosis stages to 33 named biopsy samples.
- The source `Integration` script independently lists the same 33 patient labels and fibrosis factors and composes the 2 x 4 array layout used in Supplementary Figure S1E.
- The archived reconstruction table contains 33 biopsy entries, 32 patient clusters, 14 early, 5 intermediate, and 14 late biopsies, matching the paper.

## Reproducible geometry

`code/spatial/UQ_Visium_component_audit.py` constructs connected components from `array_row` and `array_col` only. It does not read expression values or module scores. The current rerun recovered 43 components across the eight published arrays.

`code/spatial/UQ_biopsy_mapping_sensitivity.py` compared this geometry-only reconstruction with the archived component and biopsy assignments. It recovered 13,539 analysis spots directly from connected components with 100% component-label agreement. Forty-six spots in small disconnected fragments had been assigned to the nearest registered component. The script excludes those spots in a dedicated sensitivity analysis.

## Registration evidence

The component-to-biopsy registration was transcribed from the published Supplementary Figure S1E before the project generated ECM, retention, resolver, repair, or locked-state scores. The registration table is preserved at `docs/methods_audit/machine_readable/UQ_Biopsy_Reconstruction_Map.tsv`.

The archived filesystem timeline is consistent with that sequence: the component geometry output was completed at 19:54:16, the biopsy map at 20:04:22, and the module-score output began at 20:10:18 on 2026-07-11. These timestamps support temporal separation; they are not a substitute for a prospective blinding log.

The source study's original per-spot `*_samples.csv` files are referenced by its public code but are absent from the local data archive. The original operator's strict score blinding was not documented. Therefore the historical record supports temporal separation from score generation, while it does not prove strict operator blinding.

## Stress tests

The mapping sensitivity analysis reruns spatial topology after:

1. removing all nearest-component fragment rescues;
2. splitting every geometric component into an independent analysis unit;
3. combining component splitting with fragment removal;
4. excluding the inferred synchronous pair;
5. excluding every biopsy assembled from multiple components; and
6. leaving out each Visium array in turn.

The largest retention-domain effect and interface-compression effect retained their expected directions in all 12 scenario-metric tests and all 16 leave-one-array-out tests. Full estimates are in `results/S15_UQ_biopsy_mapping_sensitivity/`.

## Manuscript wording

Use the following wording:

> Biopsy fields were registered from the published spatial layout before module-score generation. We independently reconstructed hex-grid components using array coordinates alone and verified 100% agreement among directly connected spots. Mapping sensitivity analyses that removed rescued fragments, split geometric components, excluded inferred or merged fields, and left out each array preserved both retention-domain expansion and interface compression. Strict blinding of the historical registration operator was not documented.
