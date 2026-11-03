# SEGAN local-ramping sensitivity audit — 2026-11-03

This directory currently documents a **numerical audit**, not the complete public reproduction-code archive.

- **Published software and source-code archive:** [Zenodo DOI 10.5281/zenodo.23685364](https://doi.org/10.5281/zenodo.23685364) (the archived ZIP is the authoritative public package until its contents are migrated and version-pinned here).
- **Original experiment outputs:** 72 case JSON files, CBC logs, `case_metrics.csv`, and `primary_11_windows_summary.csv` reside in the researcher's restricted [Drive folder](https://drive.google.com/drive/folders/17oJknVKt6PWx0lMjRALBh5CS6GMChivk). These are not public redistribution materials.
- **Numerical discrepancies and solver-status caveats:** [NUMERICAL_AUDIT_2026-11-03.md](NUMERICAL_AUDIT_2026-11-03.md).
- **Revision decision:** Do not treat the existing 2026-10-28 paper sensitivity Table 4 as fully reconciled with frozen original 300-second results. Preserve 900-second targeted reruns as separate verification experiments.

The Zenodo record currently labels itself 'Version v2' while the description says 'Version 1.0.0' and the ZIP filename ends in 'v1.1.0'. These labels should be reconciled in a new Zenodo version and the outdated 'Reserved DOI pending publication' note should be removed.

**Data-use restriction:** No raw or derived confidential Iraqi SCADA hourly inputs should be pushed to this public repository.
