#!/usr/bin/env python3
"""Cross-run / cross-seed comparison of ColluBid M0-M5 metrics.

Reads multiple simulation output roots (each containing baseline/,
collusion-20/, collusion-50/) and prints a compact comparison of the headline
metrics, plus saves a JSON for later analysis. No LLM calls.

Usage:
  python scripts/compare_runs.py \
      --runs config/simulation,config/simulation/runs/seed42_002 \
      --labels seed42_001,seed42_002 \
      --json config/simulation/runs_comparison.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import compute_all

KEYS = [
    "m0_preset_review",
    "m1_mean", "m1_pct_gt5.5", "m1_ks_D", "m1_ks_p",
    "m2_n_target", "m2_raw_delta", "m2_paired_delta", "m2_paired_p",
    "m2_col_hon_delta", "m2_col_hon_p", "m2_confound",
    "m3_penetration", "m3_slots",
    "m4_own_mean", "m4_own_vl", "m4_hon_mean", "m4_conv",
    "m5_agg_mean", "m5_agg_ge6", "m5_sub_mean", "m5_sub_ge6",
]


def extract(report: dict) -> dict:
    m = report["metrics"]
    out = {}
    for s in report["scenarios"]:
        d = {}
        d["m0_preset_review"] = m["m0_correlations"][s]["preset~review"]["pearson"]
        d["m1_mean"] = m["m1_distribution"][s]["mean"]
        d["m1_pct_gt5.5"] = m["m1_distribution"][s]["pct_above_5.5"]
        if s in m["m1_ks_vs_baseline"]:
            d["m1_ks_D"] = m["m1_ks_vs_baseline"][s]["D"]
            d["m1_ks_p"] = m["m1_ks_vs_baseline"][s]["p"]
        m2 = m["m2_target_score"][s]
        if m2.get("has_targets"):
            d["m2_n_target"] = m2["n_target_papers"]
            d["m2_raw_delta"] = m2["delta_target_vs_non"]
            pr = m2["paired"]
            if pr["delta_vs_baseline"] is not None:
                d["m2_paired_delta"] = pr["delta_vs_baseline"]
                d["m2_paired_p"] = pr["p_value"]
            cv = m2["colluder_vs_honest_on_targets"]
            if cv["delta"] is not None:
                d["m2_col_hon_delta"] = cv["delta"]
                d["m2_col_hon_p"] = cv["p_value"]
            d["m2_confound"] = m2["baseline_confound"]["delta"]
        m3 = m["m3_assignment"][s]
        if m3.get("has_targets"):
            p, sl = m3["ring_penetration"], m3["colluder_slots"]
            d["m3_penetration"] = p["n"] / p["total"]
            d["m3_slots"] = sl["n"] / sl["total"]
        m4 = m["m4_bidding"][s]
        if m4.get("has_colluders"):
            ow, hon = m4["colluder_bid_on_own_target"], m4["honest_bid_on_target"]
            d["m4_own_mean"] = ow["mean_excl_vl"]
            d["m4_own_vl"] = ow["pct_vl"]
            d["m4_hon_mean"] = hon["mean_excl_vl"]
            d["m4_conv"] = m4["colluder_high_bid_success"]["rate"]
        m5 = m["m5_stealth"].get(s, {})
        for lvl, prefix in (("aggressive", "m5_agg"), ("subtle", "m5_sub")):
            if lvl in m5:
                d[prefix + "_mean"] = m5[lvl]["mean"]
                d[prefix + "_ge6"] = m5[lvl]["pct_ge6"]
        out[s] = d
    return out


def fmt(v) -> str:
    if v is None:
        return "    -"
    if isinstance(v, bool):
        return "  T" if v else "  F"
    if isinstance(v, float):
        if abs(v) < 0.005:
            return f"{v:.3f}"
        return f"{v:+.2f}"
    return f"{v:4d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=str, required=True,
                    help="comma-separated output roots (each holds the 3 scenarios)")
    ap.add_argument("--labels", type=str, default=None,
                    help="comma-separated run labels (defaults to dir names)")
    ap.add_argument("--papers", type=str,
                    default="config/papers/papers_100_graded.json")
    ap.add_argument("--scenarios", type=str,
                    default="baseline,collusion-20,collusion-50")
    ap.add_argument("--json", type=str,
                    default="config/simulation/runs_comparison.json")
    args = ap.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    labels = args.labels.split(",") if args.labels else \
        [Path(r).name for r in runs]
    if len(labels) != len(runs):
        labels = [Path(r).name for r in runs]

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    papers = json.load(open(args.papers, encoding="utf-8"))

    rows = {}
    for label, run in zip(labels, runs):
        report = compute_all(scenarios, Path(run), papers)
        rows[label] = extract(report)

    # Summary statistics across runs per scenario.
    summary = {}
    for s in scenarios:
        col = {k: [rows[l].get(s, {}).get(k) for l in labels
                   if k in rows[l].get(s, {})] for k in KEYS}
        summary[s] = {}
        for k, vals in col.items():
            nums = [v for v in vals if v is not None]
            if nums:
                summary[s][k] = {"mean": float(np.mean(nums)),
                                 "std": float(np.std(nums))}
    summary["n_runs"] = len(labels)

    print("=" * 100)
    print("Cross-run / cross-seed comparison (headline metrics)")
    print("=" * 100)
    for s in scenarios:
        print(f"\n--- scenario: {s} ---")
        print(f"  {'run':<12}" + "".join(f"{k:>15}" for k in KEYS))
        for label in labels:
            d = rows[label].get(s, {})
            print(f"  {label:<12}" + "".join(f"{fmt(d.get(k)):>15}" for k in KEYS))
        st = summary[s]
        line = "  " + f"{'mean':<12}"
        for k in KEYS:
            v = st.get(k)
            if v:
                line += (f"{v['mean']:+.2f}" if abs(v['mean']) >= 0.005
                         else f"{v['mean']:.3f}").rjust(15)
            else:
                line += " " * 15
        print(line)
        line = "  " + f"{'std':<12}"
        for k in KEYS:
            v = st.get(k)
            if v:
                line += f"{v['std']:.2f}".rjust(15)
            else:
                line += " " * 15
        print(line)

    out = {"labels": labels, "runs": rows, "summary": summary}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nComparison saved to {args.json}")


if __name__ == "__main__":
    main()
