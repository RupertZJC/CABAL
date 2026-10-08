#!/usr/bin/env python3
"""Strict own-ring review effects for the ColluBid experiments.

The analysis maps designated reviewer--paper target pairs from each collusive
run to its matched all-honest baseline.  It separates (i) the direct score gap
between assigned own-ring reviewers and honest co-reviewers on the same paper,
(ii) the paper-level counterfactual score change, and (iii) score changes for
captured versus uncaptured target papers.  No LLM calls are made.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from assignment_counterfactual import (
    DEFAULT_LABELS,
    default_run_dirs,
    load_json,
    target_structure,
    unwrap,
)


def load_scenario(run_dir: Path, scenario: str) -> dict:
    d = run_dir / scenario
    return {
        "personas": load_json(d / "personas.json"),
        "assignments": unwrap(
            load_json(d / "assignments.json"), "assignments"
        ),
        "scores": unwrap(load_json(d / "review_scores.json"), "scores"),
    }


def paper_means(scores: list[dict]) -> dict[str, float]:
    by_paper: dict[str, list[float]] = defaultdict(list)
    for row in scores:
        by_paper[row["paper_id"]].append(float(row["score"]))
    return {pid: float(np.mean(values)) for pid, values in by_paper.items()}


def summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values)) if values else None,
    }


def analyze_scenario(run_dir: Path, scenario: str) -> dict:
    baseline = load_scenario(run_dir, "baseline")
    attack = load_scenario(run_dir, scenario)
    target_pairs, reviewers_by_paper = target_structure(attack["personas"])
    target_papers = sorted(reviewers_by_paper)

    behavior = {
        p["reviewer_id"]: p.get("behavior_type", "honest")
        for p in attack["personas"]
    }
    stealth = {
        p["reviewer_id"]: p.get("stealth")
        for p in attack["personas"]
        if p.get("behavior_type") == "collusion"
    }
    assigned = {
        (x["reviewer_id"], x["paper_id"]) for x in attack["assignments"]
    }
    score_of = {
        (x["reviewer_id"], x["paper_id"]): float(x["score"])
        for x in attack["scores"]
    }
    reviews_by_paper: dict[str, list[dict]] = defaultdict(list)
    for row in attack["scores"]:
        reviews_by_paper[row["paper_id"]].append(row)

    # Equal-paper-weighted direct effect: for each captured target with both an
    # assigned designated reviewer and an honest co-reviewer, compare their
    # within-paper means before averaging across papers.
    own_paper_means, honest_paper_means, within_paper_gaps = [], [], []
    strategy_scores: dict[str, list[float]] = defaultdict(list)
    captured: dict[str, bool] = {}
    own_slot_count: dict[str, int] = {}
    for pid in target_papers:
        designated = reviewers_by_paper[pid]
        own_rids = [rid for rid in designated if (rid, pid) in assigned]
        captured[pid] = bool(own_rids)
        own_slot_count[pid] = len(own_rids)
        own_scores = [score_of[(rid, pid)] for rid in own_rids]
        honest_scores = [
            float(row["score"])
            for row in reviews_by_paper[pid]
            if behavior.get(row["reviewer_id"], "honest") == "honest"
        ]
        if own_scores and honest_scores:
            own_mean = float(np.mean(own_scores))
            honest_mean = float(np.mean(honest_scores))
            own_paper_means.append(own_mean)
            honest_paper_means.append(honest_mean)
            within_paper_gaps.append(own_mean - honest_mean)
        for rid in own_rids:
            strategy_scores[stealth.get(rid, "unknown")].append(
                score_of[(rid, pid)]
            )

    base_mean = paper_means(baseline["scores"])
    attack_mean = paper_means(attack["scores"])
    paper_deltas = {
        pid: attack_mean[pid] - base_mean[pid] for pid in target_papers
    }
    all_delta = list(paper_deltas.values())
    captured_delta = [paper_deltas[p] for p in target_papers if captured[p]]
    uncaptured_delta = [paper_deltas[p] for p in target_papers if not captured[p]]
    capture_gap = (
        float(np.mean(captured_delta) - np.mean(uncaptured_delta))
        if captured_delta and uncaptured_delta else None
    )

    # Descriptive dose response by number of assigned designated reviewers.
    by_slots: dict[str, list[float]] = defaultdict(list)
    for pid in target_papers:
        by_slots[str(own_slot_count[pid])].append(paper_deltas[pid])

    upward_crossings = sum(
        base_mean[p] < 6 <= attack_mean[p] for p in target_papers
    )
    downward_crossings = sum(
        attack_mean[p] < 6 <= base_mean[p] for p in target_papers
    )

    return {
        "n_target_papers": len(target_papers),
        "n_captured_papers": sum(captured.values()),
        "direct_own_ring_vs_honest": {
            "n_papers": len(within_paper_gaps),
            "own_ring_mean": float(np.mean(own_paper_means)),
            "honest_coreviewer_mean": float(np.mean(honest_paper_means)),
            "within_paper_gap": float(np.mean(within_paper_gaps)),
        },
        "paper_score_change_vs_baseline": {
            "all_targets": summary(all_delta),
            "captured_targets": summary(captured_delta),
            "uncaptured_targets": summary(uncaptured_delta),
            "captured_minus_uncaptured": capture_gap,
            "by_own_ring_slots": {
                slots: summary(values) for slots, values in sorted(by_slots.items())
            },
        },
        "strategy": {
            name: summary(values) for name, values in strategy_scores.items()
        },
        "threshold_crossings": {
            "upward_across_6": upward_crossings,
            "downward_across_6": downward_crossings,
        },
    }


def summarize_runs(per_run: dict, scenarios: list[str]) -> dict:
    paths = {
        "direct_own_mean": ("direct_own_ring_vs_honest", "own_ring_mean"),
        "direct_honest_mean": (
            "direct_own_ring_vs_honest", "honest_coreviewer_mean"
        ),
        "direct_gap": ("direct_own_ring_vs_honest", "within_paper_gap"),
        "all_target_delta": (
            "paper_score_change_vs_baseline", "all_targets", "mean"
        ),
        "captured_delta": (
            "paper_score_change_vs_baseline", "captured_targets", "mean"
        ),
        "uncaptured_delta": (
            "paper_score_change_vs_baseline", "uncaptured_targets", "mean"
        ),
        "capture_gap": (
            "paper_score_change_vs_baseline", "captured_minus_uncaptured"
        ),
        "aggressive_mean": ("strategy", "aggressive", "mean"),
        "subtle_mean": ("strategy", "subtle", "mean"),
    }

    def get(row: dict, path: tuple[str, ...]):
        for key in path:
            row = row[key]
        return row

    out = {}
    for scenario in scenarios:
        rows = [run[scenario] for run in per_run.values()]
        scenario_out = {}
        for name, path in paths.items():
            values = np.asarray([get(row, path) for row in rows], dtype=float)
            scenario_out[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "values": values.tolist(),
            }
        for name in ("n_target_papers", "n_captured_papers"):
            values = np.asarray([row[name] for row in rows], dtype=float)
            scenario_out[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "values": values.tolist(),
            }
        out[scenario] = scenario_out
    return out


def format_report(result: dict) -> str:
    lines = [
        "ColluBid target-review counterfactual analysis",
        "================================================",
        "Values are mean +/- population std over seven simulation runs.",
        "Direct gaps are equal-paper-weighted own-ring minus honest scores.",
        "",
    ]
    for scenario, s in result["summary"].items():
        f = lambda key: f"{s[key]['mean']:.2f} +/- {s[key]['std']:.2f}"
        lines.extend([
            scenario,
            "-" * len(scenario),
            f"target papers:             {f('n_target_papers')}",
            f"captured target papers:    {f('n_captured_papers')}",
            f"own-ring reviewer score:   {f('direct_own_mean')}",
            f"honest co-reviewer score:  {f('direct_honest_mean')}",
            f"within-paper direct gap:   {f('direct_gap')}",
            f"all-target paired uplift:  {f('all_target_delta')}",
            f"captured target uplift:    {f('captured_delta')}",
            f"uncaptured target uplift:  {f('uncaptured_delta')}",
            f"captured - uncaptured:     {f('capture_gap')}",
            f"aggressive own-target:     {f('aggressive_mean')}",
            f"subtle own-target:         {f('subtle_mean')}",
            "",
        ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-dir", default="config/simulation")
    parser.add_argument("--runs", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--scenarios", default="collusion-20,collusion-50")
    parser.add_argument(
        "--json", default="config/simulation/review_counterfactual.json"
    )
    parser.add_argument(
        "--report", default="config/simulation/review_counterfactual.txt"
    )
    args = parser.parse_args()

    sim_dir = Path(args.simulation_dir)
    run_dirs = (
        [Path(x) for x in args.runs.split(",")]
        if args.runs else default_run_dirs(sim_dir)
    )
    labels = args.labels.split(",") if args.labels else DEFAULT_LABELS
    scenarios = args.scenarios.split(",")
    if len(run_dirs) != len(labels):
        raise ValueError("--runs and --labels must have the same length")

    per_run = {
        label: {
            scenario: analyze_scenario(run_dir, scenario)
            for scenario in scenarios
        }
        for label, run_dir in zip(labels, run_dirs)
    }
    result = {
        "labels": labels,
        "scenarios": scenarios,
        "per_run": per_run,
        "summary": summarize_runs(per_run, scenarios),
    }
    report = format_report(result)
    json_path, report_path = Path(args.json), Path(args.report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved JSON: {json_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
