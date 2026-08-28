# Privacy and governance boundary

## Public content

The clinical files in this release contain aggregate counts, coefficients, confidence intervals, performance metrics, and summary statistics. They contain no row-level patient records, encounter dates, dates of birth, medical-record numbers, names, contact details, addresses, or hashed identifiers.

The public-data molecular files contain derived measurements indexed by public repository samples, biopsies, arrays, spatial spots, perturbations, or model iterations. These identifiers trace to already public scientific resources.

The figure directories contain publication composites generated from the released aggregate/derived results and public scientific resources. The AI-generated graphical-abstract concept produced during internal design exploration is excluded.

## Withheld content

The following content remains under institutional governance and is not included:

- Patient-level elastography, laboratory, diagnostic, and longitudinal records.
- The verified clinical linkage key and all direct or transformed identifiers.
- The one-record flow-cytometry to elastography linkage table.
- Ethics approvals, continuing-review documents, consent records, and author contact files.
- Internal server addresses, credentials, proxy subscriptions, and local infrastructure notes.

The governed workflow scripts retain generic references to controlled input classes and transformations for auditability. They contain no controlled input, linkage key, hashing salt value, or participant record.

## Disclosure control

Clinical output was restricted to aggregate tables. The smallest contextual group in the released flow summaries contains 10 patients. A single linked flow record was excluded even though its identifiers had been hashed because the combination of one individual and detailed measurements remained disclosure-prone.

Zenodo does not anonymize files on behalf of depositors. This package therefore contains only the pre-screened open subset. Any future release of participant-level information requires an institutional data-access mechanism rather than an unrestricted Zenodo upload.
