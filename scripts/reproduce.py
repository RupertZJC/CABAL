#!/usr/bin/env python3
"""Recompute the released CABAL analyses; no API/model calls are made."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from assignment_counterfactual import DEFAULT_LABELS, default_run_dirs
from metrics import compute_all
from run_matcher import build_coi_pairs, greedy_match


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=1e-12, atol=1e-12))
    return a == b


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--skip-detectors", action="store_true")
    args = parser.parse_args()
    out = (ROOT / args.out_dir).resolve()
    if out == ROOT or out.is_relative_to(ROOT / "config") or out.is_relative_to(ROOT / "baseline_detection"):
        parser.error("Choose a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    for name in ["assignment_counterfactual", "review_counterfactual"]:
        run("scripts/" + name + ".py", "--json", out / (name + ".json"), "--report", out / (name + ".txt"))
        assert equivalent(read(out / (name + ".json")), read(ROOT / "config/simulation" / (name + ".json"))), name

    scenarios = ["baseline", "collusion-20", "collusion-50"]
    papers = read(ROOT / "config/papers/papers_100_graded.json")
    affinity = {(r["paper_id"], r["reviewer_id"]): r["score"]
                for r in read(ROOT / "config/affinity/affinity_100.json")["scores"]}
    # Preserve the conference's released reviewer order rather than sorting reassigned IDs.
    reviewers = [p["reviewer_id"] for p in read(ROOT / "config/simulation/baseline/personas.json")]
    authors = {r for p in papers for r in p["author_profile_ids"]}
    minimum = {r: 3 if r in authors else 0 for r in reviewers}
    coi = build_coi_pairs(papers)
    global_rows = {s: [] for s in scenarios}
    matcher_checks = []
    for label, directory in zip(DEFAULT_LABELS, default_run_dirs(ROOT / "config/simulation")):
        report = compute_all(scenarios, directory, papers)
        for scenario in scenarios:
            m0 = report["metrics"]["m0_correlations"][scenario]["grader~review"]
            m1 = report["metrics"]["m1_distribution"][scenario]
            global_rows[scenario].append({"mean_review_score": m1["mean"], "pct_above_5_5": m1["pct_above_5.5"],
                "pearson": m0["pearson"], "spearman": m0["spearman"], "top30_overlap": m1["accept_overlap"]})
            bids = {(r["paper_id"], r["reviewer_id"]): r["score"]
                    for r in read(directory / scenario / "bids.json")["bids"]}
            fresh = greedy_match(papers, reviewers, affinity, bids, coi, min_per_reviewer=minimum)
            saved = read(directory / scenario / "assignments.json")["assignments"]
            pair = lambda rows: {(r["paper_id"], r["reviewer_id"]) for r in rows}
            assert pair(fresh) == pair(saved), label + "/" + scenario
            matcher_checks.append({"run": label, "scenario": scenario, "matches_saved_assignments": True})
    summary = {s: {key: {"mean": float(np.mean([r[key] for r in rows])),
                         "std": float(np.std([r[key] for r in rows]))} for key in rows[0]}
               for s, rows in global_rows.items()}
    result = {"conference_wide": summary, "matcher_checks": matcher_checks}
    (out / "conference_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not args.skip_detectors:
        run("baseline_detection/detect_baselines.py", "--unbid", "-1", "--json", out / "ranking.json")
        assert equivalent(read(out / "ranking.json"), read(ROOT / "baseline_detection/results/detect_baseline_report.json"))
        run("baseline_detection/detect_ring_baselines.py", "--num-trials", "30", "--json", out / "detection_sets.json")
        fresh = read(out / "detection_sets.json")
        saved = read(ROOT / "baseline_detection/results/detect_ring_report.json")
        assert equivalent(fresh["results"], saved["results"])
    print("Paired analyses and all 21 matcher assignment sets match the released records.")
    print("Reports:", out)


if __name__ == "__main__":
    main()
