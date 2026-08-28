# UQ biopsy reconstruction and mapping sensitivity audit

## Decision

The spatial-retention result is robust to the principal reconstruction choices tested here. The largest retention-domain effect and interface-compression effect retained their expected directions in 12/12 prespecified mapping scenarios and 16/16 leave-one-array-out tests.

## Score-independent reconstruction

The audit recovered 13,539 spots directly from Visium hex-grid connected components. All directly recovered spots matched their archived component labels (100.000%). A further 46 spots belonged to small disconnected fragments and had previously been assigned to the nearest registered component. Removing those rescued spots did not reverse either key result. Component construction and registration checking used array identity, barcodes, spatial coordinates, and the archived biopsy table. ECM, HSC-retention, endothelial/macrophage resolver, repair, and fitted-state scores were excluded from assignment.

The archived map contains 7 biopsies assembled from more than one geometric component and 2 biopsy entries from one inferred synchronous pair. Splitting every component into its own analysis unit, excluding all multicomponent biopsies, and excluding the inferred pair each preserved the key effect directions.

## Key estimates

Registered-biopsy analysis:

- Largest retention domain per stage: beta = 0.0960; within-array permutation q = 0.0004.
- Resolver-retention interface per stage: beta = -0.0287; within-array permutation q = 0.0002.

Geometry-component analysis with rescued fragments removed:

- Largest retention domain per stage: beta = 0.0722; within-array permutation q = 0.0154.
- Resolver-retention interface per stage: beta = -0.0227; within-array permutation q = 0.0325.

## Audit boundary

The independent computational reconstruction is score-blind by construction. The original operator's strict blinding to ECM, retention, and repair scores is not documented in the historical archive. The defensible manuscript wording is that the reconstruction table predates score generation and that an independent score-excluded reconstruction plus mapping sensitivity analysis reproduced the conclusions. The original per-spot `*_samples.csv` files referenced by the source study code are not present in the local archive and remain the only unresolved provenance item.
