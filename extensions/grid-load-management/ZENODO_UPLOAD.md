# How to deposit the general, journal-independent v1.1.0 software release

The public research software is titled **Fairness-Constrained CLPU-Aware Grid Load Management: CBC MILP Reproducibility Code and Synthetic Examples**. No journal name is part of the software title or codebase. This release adds public synthetic demonstration inputs, a generator, CBC execution scripts and GitHub Actions tests to the previous public code release.

1. Log in to https://zenodo.org/ and open the previously published software record (version 1.0.0; depositor-supplied DOI 10.5281/zenodo.23683578).
2. After confirming that record is **published** and that this is the same research software, select **New version**. Do not overwrite the immutable 1.0.0 files.
3. Upload the complete file named `Grid_Load_Management_Public_Code_v1.1.0.zip` as the updated software archive.
4. Set resource type to **Software**, the exact title above, version **1.1.0**, code license **MIT**, author **Raham, Zaher Fadhil**, and connect ORCID 0009-0001-4728-806X.
5. Explain that the 24-hour and small test inputs are **entirely synthetic**. Raw Iraqi SCADA data and the frozen demand/supply input bundle are not distributed and the synthetic examples cannot reproduce the published experimental percentages.
6. Link the updated GitHub module at https://github.com/zaher1976/Ambient-Temperature-vs-Outage-Duration-CLPU/tree/main/extensions/grid-load-management and cite the new version's exact commit.
7. Publish the new version to obtain its distinct version DOI. Only then add that specific DOI to CITATION.cff and the journal submission materials.

**Release integrity:** `RELEASE_MANIFEST_SHA256.json` lists every regular file except itself. Check SHA256 before publication. Keep the ZIP unchanged once deposited. The previous v1.0.0 DOI identifies the older archived package; it is not the v1.1.0 version DOI.
