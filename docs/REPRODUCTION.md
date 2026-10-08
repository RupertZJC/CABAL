# Reproduction

The main entry point recomputes the saved results without LLM calls:

```bash
python scripts/reproduce.py
```

Use `--skip-detectors` for the faster seven-run outcome analysis. Detector
execution includes 30 trials per set-method/input combination. Means and
standard deviations in the paired summaries are calculated over seven runs
with population standard deviation (`ddof=0`).

Individual paired analyses:

```bash
python scripts/assignment_counterfactual.py --json outputs/assignment.json --report outputs/assignment.txt
python scripts/review_counterfactual.py --json outputs/review.json --report outputs/review.txt
```

Individual detector analyses:

```bash
python baseline_detection/detect_baselines.py --unbid -1 --json outputs/ranking.json
python baseline_detection/detect_ring_baselines.py --num-trials 30 --json outputs/detection_sets.json
```

The ranking methods use a ternary reviewer-paper matrix with unbid entries
encoded as -1. OQC, TellTail, and greedy densest use the bid-author projection;
Fraudar uses the reviewer-paper matrix. Threshold 1 includes positive bids,
while threshold 2 retains only Very High bids. These input views test different
signals and should be reported separately.

Detector regression checks:

```bash
pip install pytest
python -m pytest baseline_detection/test_regression.py -q
```

The detector ports use the saved fixed triplet. The greedy densest method is
a Charikar approximation of the upstream exact-solver variant. Authors with
multiple synthetic papers, ranking ties, random starts, and the TellTail
heuristic are handled as documented in the implementation. Findings concern
these evaluated detector/input combinations, not general undetectability.

For a newly collected profile snapshot:

```bash
python scripts/collect_profiles_s2.py --areas cs.LG cs.CV cs.CL cs.AI --num-authors 35 --output outputs/new_profiles.json
```

Set `S2_API_KEY` in the environment. This command calls Semantic Scholar and
produces researcher profiles locally; these outputs are not part of the
distributed dataset. The example requests balanced counts; the paper's saved
profile snapshot had realized area counts 37/36/34/33.
