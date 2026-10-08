# Data

The release contains numerical outputs for seven matched runs, each with an
honest, collusion-20, and collusion-50 world. The same synthetic submissions,
reference-quality assessments, affinity scores, and author relations underpin
all worlds.

## Files

| Content | Path |
|---|---|
| Synthetic submissions | `config/papers/papers_100.json` |
| Submissions and fixed reference-quality scores | `config/papers/papers_100_graded.json` |
| Reviewer-paper affinity | `config/affinity/affinity_100.json` |
| First seed-42 run | `config/simulation/{baseline,collusion-20,collusion-50}/` |
| Other six runs | `config/simulation/runs/{seed42_002,seed42_003,seed0921,seed1125,seed2024,seed7777}/` |
| Detector reference results | `baseline_detection/results/` |
| Experimental settings and release description | `config/experiment.json`, `config/release.json` |

Each scenario contains:

- `personas.json`: behavior, strategy, ring membership, peers, and target papers.
- `bids.json`: reviewer-paper bid labels and numerical values.
- `assignments.json`: assignment pairs and their matching metrics.
- `review_scores.json`: numerical evaluations and aggregate review statistics.
- `paper_level_metrics.json`: paper-level summaries and target indicators.

The first seed-42 run is labeled `seed42_001` in analysis reports but resides
directly under `config/simulation/`. The `runs/seed42_001/` directory contains
its diagnostic metric report, not another set of simulation outputs.

## De-identification

The public data contains no original researcher profile snapshot or ID mapping.
Reviewer IDs are randomly reassigned to `R0001`–`R0140`; synthetic-paper IDs to
`S0001`–`S0100`. The same mapping is applied to authorship, affinity, ring peers,
targets, bids, assignments, numerical reviews, and detection sets.

Researcher areas and representative publication titles are removed from
personas. Bid rationales, review text, grader reasoning, and profile-derived
target descriptions are omitted. Synthetic submission text is checked for
exact references to profile publication titles, institutions, contact details,
and profile links. Any resulting redactions are counted in `config/release.json`.

Numerical values and input record order are retained. Consequently, the
released records support recomputation of the saved statistics without
redistributing the underlying real-world researcher portfolios. The synthetic
reviewer behavior must not be interpreted as behavior of real researchers.

## Reuse

The dataset can support paired outcome analysis and benchmarking of detectors
against known simulated ring/target labels. It does not provide personal
backgrounds for regenerating the original reviewer agents or reconstructing
the exact stochastic model calls. Published prompt templates are in `prompts/`.

The numerical dataset under `config/` and saved detector results under
`baseline_detection/results/` are licensed under CC BY 4.0. Please cite the
CABAL paper and identify modifications when redistributing derived versions.
