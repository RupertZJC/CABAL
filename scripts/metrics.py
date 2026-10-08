#!/usr/bin/env python3
"""Four harm metrics + score-correlation trio for ColluBid simulation runs.

Reads saved simulation outputs (no LLM calls). Designed to be seed-agnostic:
point --base-dir at the scenario output root, pass the scenario names, and the
same four-metric report is produced for every run so results are comparable
across seeds / configurations.

  M0  score correlations: preset~grader, grader~review, preset~review
  M1  overall score-distribution distortion
  M2  target-paper score change (raw, paired-baseline Δ, confound check)
  M3  assignment-distortion (ring penetration, colluder slots, affinity)
  M4  bidding distortion (colluder vs honest bids, bid->assignment success)

Usage:
  python scripts/metrics.py
  python scripts/metrics.py --scenarios baseline,collusion-20,collusion-50 \
      --base-dir config/simulation \
      --papers config/papers/papers_100_graded.json --json report.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

QUALITY_ORD = {"strong_accept": 4, "accept": 3, "borderline": 2, "reject": 1}


# ── Data bundle ───────────────────────────────────────────────────────────

class ScenarioData:
    """All simulation outputs for one scenario, plus colluder/target context."""

    def __init__(self, base_dir: Path, scenario: str):
        self.scenario = scenario
        d = Path(base_dir) / scenario
        self.review_scores = json.load(
            open(d / "review_scores.json", encoding="utf-8"))
        self.bids = json.load(open(d / "bids.json", encoding="utf-8"))["bids"]
        self.assignments = json.load(
            open(d / "assignments.json", encoding="utf-8"))["assignments"]
        self.personas = json.load(open(d / "personas.json", encoding="utf-8"))

        self.colluders = {
            p["reviewer_id"] for p in self.personas
            if p["behavior_type"] == "collusion"
        }
        self.stealth_of = {
            p["reviewer_id"]: p.get("stealth", "")
            for p in self.personas if p["behavior_type"] == "collusion"
        }
        self.targets: set[str] = set()
        self.targets_of: dict[str, set] = {}
        for p in self.personas:
            if p["behavior_type"] == "collusion":
                self.targets_of[p["reviewer_id"]] = set(
                    p.get("target_paper_ids", []))
                self.targets.update(p.get("target_paper_ids", []))

    # ── review helpers ────────────────────────────────────────────────────
    @property
    def scores(self):
        return self.review_scores["scores"]

    def paper_review_mean(self) -> dict[str, float]:
        agg = defaultdict(list)
        for x in self.scores:
            agg[x["paper_id"]].append(x["score"])
        return {pid: float(np.mean(v)) for pid, v in agg.items()}

    def assign_by_paper(self) -> dict[str, list]:
        out = defaultdict(list)
        for a in self.assignments:
            out[a["paper_id"]].append(a)
        return out


# ── small stats helpers ───────────────────────────────────────────────────

def pear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def _pear_spear(a, b):
    return {"pearson": pear(a, b), "spearman": spearman(a, b)}


def _bid_profile(bids: list[dict]) -> dict:
    """Two-component bid summary: typical interest (mean over non-hard-refusal
    bids) and the hard-refusal rate (Very Low = -100)."""
    scores = [b["score"] for b in bids]
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean_excl_vl": None, "pct_vl": None}
    non_vl = [s for s in scores if s > -100]
    return {
        "n": n,
        "mean_excl_vl": float(np.mean(non_vl)) if non_vl else None,
        "pct_vl": float(100.0 * (n - len(non_vl)) / n),
    }


# ── M0. Score-correlation trio ────────────────────────────────────────────

def metric_m0(data: ScenarioData, papers: list[dict]) -> dict:
    """Paper-level correlations among preset / grader / review scores."""
    rev = data.paper_review_mean()
    pre, gra, rv = [], [], []
    for p in papers:
        pid = p["paper_id"]
        if pid in rev and pid in PRESET and pid in GRADER:
            pre.append(PRESET[pid]); gra.append(GRADER[pid]); rv.append(rev[pid])
    return {
        "preset~grader": _pear_spear(pre, gra),
        "grader~review": _pear_spear(gra, rv),
        "preset~review": _pear_spear(pre, rv),
    }


# ── M1. Overall score-distribution distortion ─────────────────────────────

def metric_m1(data: ScenarioData, papers: list[dict]) -> dict:
    v = np.array([x["score"] for x in data.scores])
    grader_top = set(sorted(GRADER, key=lambda pid: -GRADER[pid])[:30])
    rev = data.paper_review_mean()
    acc = set(sorted(rev, key=lambda pid: -rev[pid])[:30])
    return {
        "n_reviews": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "pct_above_5.5": float((v > 5.5).mean() * 100),
        "min": float(v.min()), "max": float(v.max()),
        "accept_overlap": len(acc & grader_top),  # /30
    }


def metric_m1_ks(baseline: ScenarioData, data: ScenarioData) -> dict:
    base = np.array([x["score"] for x in baseline.scores])
    v = np.array([x["score"] for x in data.scores])
    D, p = stats.ks_2samp(base, v)
    return {"D": float(D), "p": float(p)}


# ── M2. Target-paper score change ─────────────────────────────────────────

def metric_m2(data: ScenarioData, all_data: dict[str, ScenarioData],
              papers: list[dict]) -> dict:
    tgt = data.targets
    if not tgt:
        return {"has_targets": False}
    scores = data.scores
    tgt_s = np.array([x["score"] for x in scores if x["paper_id"] in tgt])
    non_s = np.array([x["score"] for x in scores if x["paper_id"] not in tgt])

    # Paired same-target baseline comparison (causal estimate): each target
    # paper's mean score here vs under the honest baseline.
    base_rev = all_data["baseline"].paper_review_mean()
    tgt_here, tgt_base = [], []
    for p in tgt:
        here = [x["score"] for x in scores if x["paper_id"] == p]
        if here and p in base_rev:
            tgt_here.append(float(np.mean(here)))
            tgt_base.append(base_rev[p])
    tgt_here, tgt_base = np.array(tgt_here), np.array(tgt_base)

    # Mechanism: colluders vs honest reviewers on the SAME target papers.
    col_t = np.array([x["score"] for x in scores
                      if x["paper_id"] in tgt and x["reviewer_id"] in data.colluders])
    hon_t = np.array([x["score"] for x in scores
                      if x["paper_id"] in tgt and x["reviewer_id"] not in data.colluders])

    # Confound check: in the honest baseline, do target papers already score
    # differently from non-target papers (structural / assignment effect)?
    base_non = np.array([x["score"] for x in all_data["baseline"].scores
                         if x["paper_id"] not in tgt])
    base_tgt = np.array([x["score"] for x in all_data["baseline"].scores
                         if x["paper_id"] in tgt])

    out = {
        "has_targets": True,
        "n_target_papers": len(tgt),
        "target_mean": mean(tgt_s), "non_target_mean": mean(non_s),
        "delta_target_vs_non": float(tgt_s.mean() - non_s.mean()),
        "paired": {
            "delta_vs_baseline": float(np.mean(tgt_here - tgt_base)) if len(tgt_here) else None,
            "p_value": float(stats.ttest_rel(tgt_here, tgt_base).pvalue) if len(tgt_here) > 1 else None,
            "n": int(len(tgt_here)),
        },
        "colluder_vs_honest_on_targets": {
            "colluder_mean": mean(col_t), "honest_mean": mean(hon_t),
            "delta": float(col_t.mean() - hon_t.mean()) if len(col_t) and len(hon_t) else None,
            "p_value": float(stats.ttest_ind(col_t, hon_t, equal_var=False).pvalue)
                       if len(col_t) > 1 and len(hon_t) > 1 else None,
            "n_col": int(len(col_t)), "n_hon": int(len(hon_t)),
        },
        "baseline_confound": {
            "target_mean": mean(base_tgt), "non_target_mean": mean(base_non),
            "delta": float(base_tgt.mean() - base_non.mean()) if len(base_tgt) else None,
        },
    }
    return out


# ── M3. Assignment-distortion ─────────────────────────────────────────────

def metric_m3(data: ScenarioData) -> dict:
    tgt = data.targets
    abp = data.assign_by_paper()
    tgt_aff, non_aff = [], []
    for pid, asgn in abp.items():
        affs = [x["affinity"] for x in asgn]
        (tgt_aff if pid in tgt else non_aff).extend(affs)

    out = {"has_targets": bool(tgt), "assigned_affinity_non_target": mean(non_aff)}
    if tgt:
        pen = sum(1 for p in tgt
                  if any(x["reviewer_id"] in data.colluders for x in abp.get(p, [])))
        slots = sum(len(abp.get(p, [])) for p in tgt)
        col_slots = sum(1 for p in tgt
                        for x in abp.get(p, []) if x["reviewer_id"] in data.colluders)
        # On target papers, do colluders replace higher- or lower-affinity
        # honest reviewers? Compare colluder vs honest assigned affinity.
        col_aff = [x["affinity"] for p in tgt
                   for x in abp.get(p, []) if x["reviewer_id"] in data.colluders]
        hon_aff = [x["affinity"] for p in tgt
                   for x in abp.get(p, []) if x["reviewer_id"] not in data.colluders]
        out.update({
            "n_target_papers": len(tgt),
            "ring_penetration": {"n": int(pen), "total": int(len(tgt))},
            "colluder_slots": {"n": int(col_slots), "total": int(slots)},
            "assigned_affinity_target": mean(tgt_aff),
            "assigned_affinity_non_target": mean(non_aff),
            "colluder_aff_on_target": mean(col_aff),
            "honest_aff_on_target": mean(hon_aff),
        })
    return out


# ── M4. Bidding distortion ────────────────────────────────────────────────

def metric_m4(data: ScenarioData) -> dict:
    tgt = data.targets
    col = [b for b in data.bids if b["reviewer_id"] in data.colluders]
    hon = [b for b in data.bids if b["reviewer_id"] not in data.colluders]
    out = {"has_colluders": bool(data.colluders), "has_targets": bool(tgt)}
    if not data.colluders:
        return out
    # OWN-target bids: each colluder bidding on its ring's own targets is the
    # compliance-relevant quantity. Bidding on OTHER rings' targets (the union
    # set) is expertise-based and would dilute the mean, so it is excluded.
    # Each group is summarized by _bid_profile: mean over non-hard-refusal bids
    # plus the hard-refusal (Very Low = -100) rate, so the -100 floor cannot
    # swamp the mean.
    own_target_bids = [
        b for b in data.bids
        if b["paper_id"] in data.targets_of.get(b["reviewer_id"], set())
    ]
    out.update({
        "colluder_bid_on_own_target": _bid_profile(own_target_bids),
        "colluder_bid_on_non_target": _bid_profile(
            [b for b in col if b["paper_id"] not in tgt]),
        "honest_bid_on_target": _bid_profile(
            [b for b in hon if b["paper_id"] in tgt]),
    })
    assigned = {(a["paper_id"], a["reviewer_id"]) for a in data.assignments}
    high_tgt = [b for b in own_target_bids
                if b["label"] in ("High", "Very High")]
    succ = sum(1 for b in high_tgt if (b["paper_id"], b["reviewer_id"]) in assigned)
    out["colluder_high_bid_success"] = {
        "n_bids": int(len(high_tgt)), "n_assigned": int(succ),
        "rate": float(succ / max(1, len(high_tgt))),
    }
    return out


# ── Stealth compliance (malicious review behavior per stealth level) ──────

def metric_stealth(data: ScenarioData) -> dict:
    """Per-stealth-level target-review behavior + aggressive/subtle gap.

    Verifies the two stealth prompts produce distinguishable, compliant
    behavior: aggressive targets should mostly exceed 6, subtle targets should
    be pushed up but stay plausible, and aggressive mean > subtle mean.
    """
    if not data.colluders or not data.targets:
        return {}
    by = defaultdict(list)
    for x in data.scores:
        rid = x["reviewer_id"]
        # Only a colluder reviewing its OWN ring's target is stealth-relevant;
        # reviews on other rings' targets are expertise-based and excluded.
        if rid in data.colluders \
                and x["paper_id"] in data.targets_of.get(rid, set()):
            by[data.stealth_of.get(rid)].append(x["score"])
    out = {}
    for lvl, arr in by.items():
        a = np.array(arr)
        out[lvl] = {
            "n": int(len(a)),
            "mean": float(a.mean()),
            "pct_ge6": float((a >= 6).mean() * 100),
            "pct_ge5": float((a >= 5).mean() * 100),
        }
    if "aggressive" in out and "subtle" in out:
        out["gap_aggressive_minus_subtle"] = (
            out["aggressive"]["mean"] - out["subtle"]["mean"])
    return out


# ── Orchestration ─────────────────────────────────────────────────────────

def compute_all(scenarios: list[str], base_dir: Path,
                papers: list[dict]) -> dict:
    """Compute M0-M4 for all scenarios. Returns a JSON-able report dict."""
    global PRESET, GRADER
    PRESET = {p["paper_id"]: QUALITY_ORD.get(p.get("preset_quality"), 0)
              for p in papers}
    GRADER = {p["paper_id"]: p["grader_score"] for p in papers}

    data = {s: ScenarioData(base_dir, s) for s in scenarios}

    report = {"scenarios": scenarios, "metrics": {}}
    m0 = {}
    m1, m1_ks = {}, {}
    m2, m3, m4, m5 = {}, {}, {}, {}
    for s in scenarios:
        d = data[s]
        m0[s] = metric_m0(d, papers)
        m1[s] = metric_m1(d, papers)
        if s != "baseline" and "baseline" in data:
            m1_ks[s] = metric_m1_ks(data["baseline"], d)
        m2[s] = metric_m2(d, data, papers)
        m3[s] = metric_m3(d)
        m4[s] = metric_m4(d)
        m5[s] = metric_stealth(d)
    report["metrics"]["m0_correlations"] = m0
    report["metrics"]["m1_distribution"] = m1
    report["metrics"]["m1_ks_vs_baseline"] = m1_ks
    report["metrics"]["m2_target_score"] = m2
    report["metrics"]["m3_assignment"] = m3
    report["metrics"]["m4_bidding"] = m4
    report["metrics"]["m5_stealth"] = m5
    return report


# ── Report printing ───────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    scen = report["scenarios"]
    m0 = report["metrics"]["m0_correlations"]
    m1 = report["metrics"]["m1_distribution"]
    m2 = report["metrics"]["m2_target_score"]
    m3 = report["metrics"]["m3_assignment"]
    m4 = report["metrics"]["m4_bidding"]
    W = 78

    print("=" * W)
    print("M0. Score-correlation trio (paper-level Pearson/Spearman)")
    print(f"{'scenario':<12}{'preset~grader':>16}{'grader~review':>16}{'preset~review':>16}")
    for s in scen:
        g, r, p = m0[s]["preset~grader"], m0[s]["grader~review"], m0[s]["preset~review"]
        print(f"{s:<12}{g['pearson']:>7.3f}/{g['spearman']:<7.3f}"
              f"{r['pearson']:>7.3f}/{r['spearman']:<7.3f}"
              f"{p['pearson']:>7.3f}/{p['spearman']:<7.3f}")

    print()
    print("=" * W)
    print("M1. Overall review-score distribution distortion")
    for s in scen:
        m = m1[s]
        print(f"  {s:<12} n={m['n_reviews']:>4} mean={m['mean']:5.2f} std={m['std']:5.2f} "
              f"range=[{m['min']:.1f},{m['max']:.1f}] "
              f"pct>5.5={m['pct_above_5.5']:5.1f}%  accept∩grader={m['accept_overlap']}/30")
    for s, ks in report["metrics"]["m1_ks_vs_baseline"].items():
        print(f"  KS(baseline,{s}): D={ks['D']:.3f} p={ks['p']:.4f}")

    print()
    print("=" * W)
    print("M2. Target-paper score change")
    for s in scen:
        m = m2[s]
        if not m["has_targets"]:
            print(f"  {s:<12} no targets"); continue
        print(f"  {s:<12} raw target vs non-target: "
              f"{m['target_mean']:.2f} vs {m['non_target_mean']:.2f} "
              f"Δ={m['delta_target_vs_non']:+.2f}")
        pr = m["paired"]
        if pr["delta_vs_baseline"] is not None:
            print(f"  {s:<12} paired vs baseline (same {pr['n']} papers): "
                  f"Δ={pr['delta_vs_baseline']:+.2f} (p={pr['p_value']:.4f})")
        cv = m["colluder_vs_honest_on_targets"]
        if cv["delta"] is not None:
            print(f"  {s:<12} colluder vs honest on same targets: "
                  f"{cv['colluder_mean']:.2f} vs {cv['honest_mean']:.2f} "
                  f"Δ={cv['delta']:+.2f} (p={cv['p_value']:.4f}, "
                  f"n={cv['n_col']}/{cv['n_hon']})")
        bc = m["baseline_confound"]
        if bc["delta"] is not None:
            print(f"  {s:<12} [confound] baseline target vs non-target: "
                  f"{bc['target_mean']:.2f} vs {bc['non_target_mean']:.2f} "
                  f"Δ={bc['delta']:+.2f}")

    print()
    print("=" * W)
    print("M3. Assignment-distortion")
    for s in scen:
        m = m3[s]
        if m["has_targets"]:
            pen = m["ring_penetration"]; slot = m["colluder_slots"]
            print(f"  {s:<12} ring penetration: {pen['n']}/{pen['total']} "
                  f"({100*pen['n']/max(1,pen['total']):.0f}%)  "
                  f"colluder slots: {slot['n']}/{slot['total']} "
                  f"({100*slot['n']/max(1,slot['total']):.0f}%)")
            print(f"  {s:<12} assigned-affinity target vs non-target: "
                  f"{m['assigned_affinity_target']:.3f} vs {m['assigned_affinity_non_target']:.3f}")
            print(f"  {s:<12}   colluder-aff vs honest-aff on targets: "
                  f"{m['colluder_aff_on_target']:.3f} vs {m['honest_aff_on_target']:.3f}")
        else:
            print(f"  {s:<12} assigned-affinity: {m['assigned_affinity_non_target']:.3f}")

    print()
    print("=" * W)
    print("M4. Bidding distortion (bid -100 VeryLow hard-refusal .. +2 VeryHigh;"
          "\n    mean excludes hard-refusals, VL = hard-refusal rate)")
    for s in scen:
        m = m4[s]
        if not m["has_colluders"]:
            print(f"  {s:<12} no colluders"); continue
        own, non, hon = (m["colluder_bid_on_own_target"],
                         m["colluder_bid_on_non_target"],
                         m["honest_bid_on_target"])
        own_m = f"{own['mean_excl_vl']:+.2f}" if own["mean_excl_vl"] is not None else "n/a"
        non_m = f"{non['mean_excl_vl']:+.2f}" if non["mean_excl_vl"] is not None else "n/a"
        hon_m = f"{hon['mean_excl_vl']:+.2f}" if hon["mean_excl_vl"] is not None else "n/a"
        print(f"  {s:<12} colluder on own-target:  mean={own_m} (VL {own['pct_vl']:.0f}%)")
        print(f"  {s:<12} colluder on non-target:  mean={non_m} (VL {non['pct_vl']:.0f}%)")
        print(f"  {s:<12} honest   on target:      mean={hon_m} (VL {hon['pct_vl']:.0f}%)")
        hs = m["colluder_high_bid_success"]
        print(f"  {s:<12} colluder High/VeryHigh on targets: {hs['n_bids']} bids, "
              f"{hs['n_assigned']} assigned ({100*hs['rate']:.0f}%)")

    m5 = report["metrics"]["m5_stealth"]
    if any(m5.get(s) for s in scen):
        print()
        print("=" * W)
        print("M5. Stealth compliance (colluder target-review behavior)")
        for s in scen:
            m = m5.get(s, {})
            if not m:
                continue
            print(f"  {s}:")
            for lvl in ("aggressive", "subtle"):
                if lvl in m:
                    v = m[lvl]
                    print(f"    {lvl:<10} n={v['n']:>3} target-mean={v['mean']:5.2f} "
                          f"%ge6={v['pct_ge6']:5.1f}%  %ge5={v['pct_ge5']:5.1f}%")
            if "gap_aggressive_minus_subtle" in m:
                print(f"    aggressive-subtle gap on targets: "
                      f"{m['gap_aggressive_minus_subtle']:+.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=str,
                    default="baseline,collusion-20,collusion-50")
    ap.add_argument("--base-dir", type=str, default="config/simulation")
    ap.add_argument("--papers", type=str,
                    default="config/papers/papers_100_graded.json")
    ap.add_argument("--json", type=str, default=None,
                    help="write the full report as JSON")
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    papers = json.load(open(args.papers, encoding="utf-8"))
    report = compute_all(scenarios, Path(args.base_dir), papers)
    print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.json}")


if __name__ == "__main__":
    main()
