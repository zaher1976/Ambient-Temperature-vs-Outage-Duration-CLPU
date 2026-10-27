# New Zenodo software deposit: publication checklist

This code-only archive represents a distinct paper/experiment from the older temperature-versus-duration CLPU record. **Do not create a new version of the older record** unless the manuscripts and scholarly objects are demonstrably the same. Create a **new upload** for this SEGAN sensitivity package.

1. Review `README.md`, `RELEASE_MANIFEST_SHA256.json`, and the contents of this ZIP. There must be no raw SCADA data, private tokens or protected hourly `multiwindow_inputs_frozen.npz`.
2. Sign into https://zenodo.org and select **New upload**. Upload the public code-only ZIP.
3. Resource type **Software**; Title: `Fairness-Constrained CLPU-Aware Load Management: SEGAN Local-Ramping CBC Sensitivity Reproduction Code`; Creator: `Raham, Zaher Fadhil`; publication year **2026**; version **v1.0.0**; license **MIT** (for code). Set public visibility only after verifying institutional disclosure rules.
4. Description: *Source code, synthetic first-hour restoration test, provenance manifests and aggregate sensitivity checks for 72 CBC MILP cases, including six 900-second re-solves. Raw SCADA and frozen hourly optimization arrays are not included due to data-use restrictions.*
5. Related identifier: GitHub commit/PR URL for the SEGAN module (after final public release). Reference the archived publication once public. Do not insert an unissued DOI.
6. Publish to obtain a **new version-specific DOI**. Add the DOI to the manuscript, release metadata and public GitHub README, then create a release that matches the deposited source.

Zenodo generally preserves published files; use a new record/version for file changes. A GitHub repository containing an older `.zenodo.json` may archive that older publication's metadata when released, so a separate new Zenodo deposit avoids DOI conflation.
