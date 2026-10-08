#!/usr/bin/env python3
"""Paired counterfactual analysis of collusive bidding and assignment capture.

For each collusive scenario, the script extracts the designated own-ring
reviewer--paper target pairs from personas.json and evaluates those exact pairs
in both the collusive world and its matched all-honest baseline.  This avoids
confounding attack effects with the high expertise affinity used to construct
collusion rings.

The report contains four related quantities:
  1. bid behavior on the same designated target pairs;
  2. pair-level assignment probability and its paired uplift;
  3. target-paper coverage by at least one designated own-ring reviewer;
  4. the share of target-paper review slots occupied by designated reviewers.

No LLM calls are made.  The script only reads saved simulation outputs.

Usage (defaults cover the seven runs used in the paper):
  python scripts/assignment_counterfactual.py
  python scripts/assignment_counterfactual.py \
      --json config/simulation/assignment_counterfactual.json \
      --report config/simulation/assignment_counterfactual.txt
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_LABELS = [
    "seed42_001", "seed42_002", "seed42_003", "seed0921",
    "seed1125", "seed2024", "seed7777",
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def unwrap(data, key: str) -> list[dict]:
    if isinstance(data, dict):
        return data[key]
    return data


def load_scenario(run_dir: Path, scenario: str) -> dict:
    d = run_dir / scenario
    return {
        "personas": load_json(d / "personas.json"),
        "bids": unwrap(load_json(d / "bids.json"), "bids"),
        "assignments": unwrap(
            load_json(d / "assignments.json"), "assignments"
        ),
    }


def target_structure(personas: list[dict]) -> tuple[set[tuple[str, str]], dict[str, set[str]]]:
    """Return designated (reviewer, paper) pairs and reviewers per target."""
    pairs: set[tuple[str, str]] = set()
    reviewers_by_paper: dict[str, set[str]] = defaultdict(set)
    for persona in personas:
        if persona.get("behavior_type") != "collusion":
            continue
        rid = persona["reviewer_id"]
        for pid in persona.get("target_paper_ids", []):
            pairs.add((rid, pid))
            reviewers_by_paper[pid].add(rid)
    return pairs, dict(reviewers_by_paper)


def bid_profile(scores: list[float]) -> dict:
    """Summarize ordinary bid strength separately from the -100 refusal rate."""
    n = len(scores)
    non_vl = [x for x in scores if x > -100]
    return {
        "n": n,
        "mean_excl_vl": float(np.mean(non_vl)) if non_vl else None,
        "pct_vl": float(100 * (n - len(non_vl)) / n) if n else None,
        "pct_vh": float(100 * sum(x == 2.0 for x in scores) / n) if n else None,
    }


def mcnemar_exact(collusive: list[int], baseline: list[int]) -> dict:
    """Exact two-sided McNemar test for paired binary outcomes."""
    gain = sum(c == 1 and b == 0 for c, b in zip(collusive, baseline))
    loss = sum(c == 0 and b == 1 for c, b in zip(collusive, baseline))
    discordant = gain + loss
    # Under H0 the two directions are equiprobable.  For p=.5 the exact
    # two-sided binomial p-value is twice the lower-tail probability at the
    # smaller discordant count (capped at one).  Implementing it here keeps
    # this analysis-only script independent of SciPy.
    if discordant:
        tail = sum(
            math.comb(discordant, k) for k in range(min(gain, loss) + 1)
        ) / (2 ** discordant)
        p = min(1.0, 2.0 * tail)
    else:
        p = 1.0
    return {"gain": gain, "loss": loss, "discordant": discordant, "p_value": p}


def binary_comparison(collusive: list[int], baseline: list[int]) -> dict:
    c_rate = float(np.mean(collusive)) if collusive else None
    b_rate = float(np.mean(baseline)) if baseline else None
    return {
        "n": len(collusive),
        "baseline_rate": b_rate,
        "collusive_rate": c_rate,
        "uplift": c_rate - b_rate if c_rate is not None else None,
        "mcnemar": mcnemar_exact(collusive, baseline),
    }


def analyze_scenario(run_dir: Path, scenario: str) -> dict:
    baseline = load_scenario(run_dir, "baseline")
    attack = load_scenario(run_dir, scenario)
    pairs, reviewers_by_paper = target_structure(attack["personas"])
    ordered_pairs = sorted(pairs)
    target_papers = sorted(reviewers_by_paper)

    base_bid = {(x["reviewer_id"], x["paper_id"]): x for x in baseline["bids"]}
    attack_bid = {(x["reviewer_id"], x["paper_id"]): x for x in attack["bids"]}
    base_assigned = {
        (x["reviewer_id"], x["paper_id"]) for x in baseline["assignments"]
    }
    attack_assigned = {
        (x["reviewer_id"], x["paper_id"]) for x in attack["assignments"]
    }

    missing_base_bids = [pair for pair in ordered_pairs if pair not in base_bid]
    missing_attack_bids = [pair for pair in ordered_pairs if pair not in attack_bid]
    if missing_base_bids or missing_attack_bids:
        raise ValueError(
            f"{run_dir.name}/{scenario}: designated pairs missing from bids "
            f"(baseline={len(missing_base_bids)}, attack={len(missing_attack_bids)})"
        )

    base_scores = [float(base_bid[pair]["score"]) for pair in ordered_pairs]
    attack_scores = [float(attack_bid[pair]["score"]) for pair in ordered_pairs]
    base_pair_y = [int(pair in base_assigned) for pair in ordered_pairs]
    attack_pair_y = [int(pair in attack_assigned) for pair in ordered_pairs]

    base_coverage = []
    attack_coverage = []
    for pid in target_papers:
        designated = reviewers_by_paper[pid]
        base_coverage.append(
            int(any((rid, pid) in base_assigned for rid in designated))
        )
        attack_coverage.append(
            int(any((rid, pid) in attack_assigned for rid in designated))
        )

    # Every saved run assigns the same number of slots per paper, but calculate
    # each denominator directly to keep the statistic valid for other settings.
    base_slots_by_paper: dict[str, int] = defaultdict(int)
    attack_slots_by_paper: dict[str, int] = defaultdict(int)
    for x in baseline["assignments"]:
        base_slots_by_paper[x["paper_id"]] += 1
    for x in attack["assignments"]:
        attack_slots_by_paper[x["paper_id"]] += 1
    base_total_slots = sum(base_slots_by_paper[pid] for pid in target_papers)
    attack_total_slots = sum(attack_slots_by_paper[pid] for pid in target_papers)
    base_own_slots = sum(
        (rid, pid) in base_assigned for rid, pid in ordered_pairs
    )
    attack_own_slots = sum(
        (rid, pid) in attack_assigned for rid, pid in ordered_pairs
    )
    base_slot_share = base_own_slots / base_total_slots
    attack_slot_share = attack_own_slots / attack_total_slots

    non_vl_pairs = [i for i, value in enumerate(base_scores) if value > -100]
    paired_bid_uplift_excl_base_vl = float(np.mean([
        attack_scores[i] - base_scores[i] for i in non_vl_pairs
    ])) if non_vl_pairs else None

    return {
        "n_target_pairs": len(ordered_pairs),
        "n_target_papers": len(target_papers),
        "target_pair_bids": {
            "baseline": bid_profile(base_scores),
            "collusive": bid_profile(attack_scores),
            "vh_rate_uplift_pp": (
                bid_profile(attack_scores)["pct_vh"]
                - bid_profile(base_scores)["pct_vh"]
            ),
            "mean_uplift_excl_baseline_vl": paired_bid_uplift_excl_base_vl,
        },
        "target_pair_assignment": binary_comparison(attack_pair_y, base_pair_y),
        "own_ring_paper_coverage": binary_comparison(
            attack_coverage, base_coverage
        ),
        "own_ring_slot_share": {
            "n_target_slots_baseline": base_total_slots,
            "n_target_slots_collusive": attack_total_slots,
            "baseline_n": int(base_own_slots),
            "collusive_n": int(attack_own_slots),
            "baseline_rate": float(base_slot_share),
            "collusive_rate": float(attack_slot_share),
            "uplift": float(attack_slot_share - base_slot_share),
        },
    }


def summarize(per_run: dict, scenarios: list[str]) -> dict:
    paths = {
        "bid_base_mean": ("target_pair_bids", "baseline", "mean_excl_vl"),
        "bid_base_vl": ("target_pair_bids", "baseline", "pct_vl"),
        "bid_base_vh": ("target_pair_bids", "baseline", "pct_vh"),
        "bid_attack_mean": ("target_pair_bids", "collusive", "mean_excl_vl"),
        "bid_attack_vh": ("target_pair_bids", "collusive", "pct_vh"),
        "bid_vh_uplift_pp": ("target_pair_bids", "vh_rate_uplift_pp"),
        "pair_base": ("target_pair_assignment", "baseline_rate"),
        "pair_attack": ("target_pair_assignment", "collusive_rate"),
        "pair_uplift": ("target_pair_assignment", "uplift"),
        "coverage_base": ("own_ring_paper_coverage", "baseline_rate"),
        "coverage_attack": ("own_ring_paper_coverage", "collusive_rate"),
        "coverage_uplift": ("own_ring_paper_coverage", "uplift"),
        "slots_base": ("own_ring_slot_share", "baseline_rate"),
        "slots_attack": ("own_ring_slot_share", "collusive_rate"),
        "slots_uplift": ("own_ring_slot_share", "uplift"),
    }

    def get(d: dict, path: tuple[str, ...]):
        for key in path:
            d = d[key]
        return d

    out = {}
    for scenario in scenarios:
        rows = [run[scenario] for run in per_run.values()]
        stats_out = {}
        for name, path in paths.items():
            values = np.asarray([get(row, path) for row in rows], dtype=float)
            stats_out[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "values": values.tolist(),
            }
        stats_out["n_target_pairs"] = {
            "mean": float(np.mean([x["n_target_pairs"] for x in rows])),
            "std": float(np.std([x["n_target_pairs"] for x in rows])),
            "values": [x["n_target_pairs"] for x in rows],
        }
        stats_out["n_target_papers"] = {
            "mean": float(np.mean([x["n_target_papers"] for x in rows])),
            "std": float(np.std([x["n_target_papers"] for x in rows])),
            "values": [x["n_target_papers"] for x in rows],
        }
        stats_out["pair_mcnemar_significant_runs"] = sum(
            x["target_pair_assignment"]["mcnemar"]["p_value"] < 0.05
            for x in rows
        )
        stats_out["coverage_mcnemar_significant_runs"] = sum(
            x["own_ring_paper_coverage"]["mcnemar"]["p_value"] < 0.05
            for x in rows
        )
        out[scenario] = stats_out
    return out


def format_report(result: dict) -> str:
    lines = [
        "ColluBid assignment counterfactual analysis",
        "==============================================",
        "Rates are paired on the exact own-ring reviewer--paper target pairs.",
        "Summary values are mean +/- population std over seven simulation runs.",
        "",
    ]
    for scenario, s in result["summary"].items():
        pct = lambda key: (
            f"{100*s[key]['mean']:.1f}% +/- {100*s[key]['std']:.1f}"
        )
        num = lambda key: f"{s[key]['mean']:.2f} +/- {s[key]['std']:.2f}"
        lines.extend([
            scenario,
            "-" * len(scenario),
            f"target pairs:  {num('n_target_pairs')}",
            f"target papers: {num('n_target_papers')}",
            "bids on same target pairs:",
            f"  baseline non-VL mean: {num('bid_base_mean')}",
            f"  baseline VH rate:     {num('bid_base_vh')}%",
            f"  baseline VL rate:     {num('bid_base_vl')}%",
            f"  collusive mean:       {num('bid_attack_mean')}",
            f"  collusive VH rate:    {num('bid_attack_vh')}%",
            f"  VH-rate uplift:       {num('bid_vh_uplift_pp')} pp",
            "target-pair assignment:",
            f"  baseline: {pct('pair_base')}",
            f"  attack:   {pct('pair_attack')}",
            f"  uplift:   {pct('pair_uplift')}",
            f"  McNemar p<.05 in {s['pair_mcnemar_significant_runs']}/7 runs",
            "own-ring paper coverage:",
            f"  baseline: {pct('coverage_base')}",
            f"  attack:   {pct('coverage_attack')}",
            f"  uplift:   {pct('coverage_uplift')}",
            f"  McNemar p<.05 in {s['coverage_mcnemar_significant_runs']}/7 runs",
            "own-ring target-slot share:",
            f"  baseline: {pct('slots_base')}",
            f"  attack:   {pct('slots_attack')}",
            f"  uplift:   {pct('slots_uplift')}",
            "",
        ])
    return "\n".join(lines)


def default_run_dirs(sim_dir: Path) -> list[Path]:
    return [
        sim_dir,
        sim_dir / "runs" / "seed42_002",
        sim_dir / "runs" / "seed42_003",
        sim_dir / "runs" / "seed0921",
        sim_dir / "runs" / "seed1125",
        sim_dir / "runs" / "seed2024",
        sim_dir / "runs" / "seed7777",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation-dir", default="config/simulation",
        help="Simulation root containing the first run and runs/ subdirectory.",
    )
    parser.add_argument(
        "--runs", default=None,
        help="Optional comma-separated run roots; overrides default seven runs.",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Optional comma-separated labels matching --runs.",
    )
    parser.add_argument(
        "--scenarios", default="collusion-20,collusion-50",
    )
    parser.add_argument(
        "--json", default="config/simulation/assignment_counterfactual.json",
    )
    parser.add_argument(
        "--report", default="config/simulation/assignment_counterfactual.txt",
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

    per_run = {}
    for label, run_dir in zip(labels, run_dirs):
        per_run[label] = {
            scenario: analyze_scenario(run_dir, scenario)
            for scenario in scenarios
        }
    result = {
        "labels": labels,
        "scenarios": scenarios,
        "per_run": per_run,
        "summary": summarize(per_run, scenarios),
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
