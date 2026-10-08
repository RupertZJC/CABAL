# Release validation

The public artifact was checked on 2026-10-08 with Python 3.10.11,
NumPy 2.2.6, and SciPy 1.15.3.

The released RQ1 and RQ2 scripts reproduce the seven-run saved summaries within
1e-12. The matcher reproduces every saved assignment pair in all 21 scenarios.
All eight detector methods reproduce the fixed-triplet reference results;
detected and baseline reviewer sets agree under the released IDs.

The private export audit verified that all **111,950 retained numerical fields**
match the source records. Original profile identifiers, researcher-background
fields, free-text rationales/reviews, credentials, and private workspace paths
are absent from the released tree. Original profiles and the random identity
mapping are not distributed. Four profile-publication references in synthetic
submission text were redacted across the two paper files.

`export_validation.json` records the checks and `data_manifest.json` lists
SHA-256 checksums of the released data files. `python scripts/reproduce.py`
recomputes the outcome and detector results. This validates analysis of the
saved numerical observations; it does not replay hosted-model calls or
recollect the original researcher snapshot.
