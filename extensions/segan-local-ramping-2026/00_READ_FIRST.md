# VERIFIED correction — numerical audit (2026-11-03)

**The manuscript's sensitivity table is numerically consistent.** The earlier `NUMERICAL_AUDIT_2026-11-03.md` contains an obsolete claim of 16 differences. **Do not cite that old discrepancy claim.** It arose by comparing the manuscript's *300-second original cases plus six 900-second reruns* with the 300-second original cases alone.

## Correct audit

- **72** original case files: W00–W11 at six local-ramping multipliers.
- Replace exactly six original time-limited cases with their 900-second re-solves: `W07_m0`, `W07_m0p25`, `W07_m0p5`, `W07_m1`, `W07_m2`, `W08_m0p5`.
- Use only **W01–W11** for means. W00 is supplementary.
- Independent recomputation from all 72 correctly selected solution arrays reproduced local and aggregate served-power total variation, served energy, and unserved energy.
- State transitions matched reconnection indicators at the first hour and later hours.
- **All sensitivity-table operational metrics agree with the manuscript to two decimal places.**
- The ratio-of-means local TV reduction is **85.51%**, mean added unserved energy **6.90 MWh**, and mean aggregate TV increase **18.80%**.
- All 72 appropriately merged cases have CBC-reported-optimal and feasible-checked statuses. CBC's label is not proof of exact global zero-gap optimality.
- Recorded Jain and CLPU metrics were compared with the saved diagnostics, but this check does not claim full independent reconstruction from the frozen demand arrays.

## Files and release policy

- [Verified six-weight aggregate table](TABLE4_VERIFIED_2026-11-03.csv)
- [Detailed correction](CORRECTION_2026-11-03.md)
- [Original 300s results](https://drive.google.com/drive/folders/17oJknVKt6PWx0lMjRALBh5CS6GMChivk)
- [Six reruns](https://drive.google.com/drive/folders/141zYFd7QvvfyR7TVy8-qAzFKh79IR8dS)
- [Published public code snapshot on Zenodo](https://doi.org/10.5281/zenodo.23685364)

Original operating inputs and case-level model outputs remain in restricted Drive folders. Public releases must not include these files without authorization.
