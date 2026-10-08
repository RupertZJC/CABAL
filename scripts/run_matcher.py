#!/usr/bin/env python3
"""
Run paper-reviewer matching and compare baselines.

Two configurations:
  Baseline A: Affinity-Only (w_bid=0)    — pure expertise match
  Baseline B: Bid-Influenced (w_bid=2.0) — honest bidding included

Algorithm: greedy assignment respecting COI + quotas.
Each paper gets top-K reviewers by aggregate score.

Usage:
    python scripts/run_matcher.py
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def build_coi_pairs(papers: list[dict]) -> set:
    """Build COI pairs: (paper_id, reviewer_id) that cannot match."""
    coi = set()
    for paper in papers:
        for seed_pid in paper.get("author_profile_ids", []):
            coi.add((paper["paper_id"], seed_pid))
    return coi


def greedy_match(
    papers: list[dict],
    reviewers: list[str],
    affinity_scores: dict,   # (paper_id, reviewer_id) → float
    bid_scores: dict,        # (paper_id, reviewer_id) → float
    coi_pairs: set,
    w_affinity: float = 1.0,
    w_bid: float = 2.0,
    n_reviewers_per_paper: int = 3,
    min_per_reviewer: dict | None = None,
    max_per_reviewer: int = 5,
) -> list[dict]:
    """
    Greedy paper-reviewer matching with per-reviewer min/max load constraints.

    For each paper, picks the top-K reviewers by aggregate score, respecting
    COI and load limits. Reviewers who have not yet met their minimum load
    get a large priority boost, so author-reviewers (min=3) are guaranteed
    to be assigned first (their reviewing obligation).

    This mirrors the MinMax solver's min/max quota constraints, but with a
    simpler greedy algorithm.
    """
    reviewer_loads = {r: 0 for r in reviewers}
    min_loads = min_per_reviewer or {r: 0 for r in reviewers}
    assignments = []

    # Sort papers by quality (strong_accept first — they benefit from best reviewers)
    quality_order = {
        "strong_accept": 0, "accept": 1, "borderline": 2, "reject": 3,
    }
    papers_sorted = sorted(
        papers,
        key=lambda p: quality_order.get(p.get("quality", "borderline"), 2),
    )

    for paper in papers_sorted:
        pid = paper["paper_id"]

        # Score all eligible reviewers
        candidates = []
        for rid in reviewers:
            if (pid, rid) in coi_pairs:
                continue
            if reviewer_loads[rid] >= max_per_reviewer:
                continue

            aff = affinity_scores.get((pid, rid), 0.0)
            bid = bid_scores.get((pid, rid), 0.0)
            aggregate = w_affinity * aff + w_bid * bid

            # Reviewers below their minimum load get a large boost so the
            # "author must review ≥3" obligation is satisfied first.
            still_needs_min = reviewer_loads[rid] < min_loads.get(rid, 0)
            boost = 1000.0 if still_needs_min else 0.0
            candidates.append((aggregate + boost, aggregate, aff, bid, rid))

        candidates.sort(key=lambda x: x[0], reverse=True)

        # Assign top-K
        assigned = 0
        for _, agg, aff, bid, rid in candidates:
            if assigned >= n_reviewers_per_paper:
                break
            if reviewer_loads[rid] >= max_per_reviewer:
                continue

            assignments.append({
                "paper_id": pid,
                "reviewer_id": rid,
                "affinity": round(aff, 4),
                "bid_score": round(bid, 2),
                "aggregate_score": round(agg, 4),
                "paper_quality": paper.get("quality", ""),
            })
            reviewer_loads[rid] += 1
            assigned += 1

    return assignments


def compute_metrics(assignments: list[dict]) -> dict:
    """Compute per-paper and global matching quality metrics."""
    by_paper = defaultdict(list)
    for a in assignments:
        by_paper[a["paper_id"]].append(a)

    per_paper = {}
    for pid, assignees in by_paper.items():
        affinities = [a["affinity"] for a in assignees]
        per_paper[pid] = {
            "n_assigned": len(assignees),
            "avg_affinity": round(float(np.mean(affinities)), 4),
            "min_affinity": round(float(np.min(affinities)), 4),
            "max_affinity": round(float(np.max(affinities)), 4),
            "quality": assignees[0]["paper_quality"],
        }

    all_aff = [a["affinity"] for a in assignments]
    avg_by_quality = defaultdict(list)
    for a in assignments:
        avg_by_quality[a["paper_quality"]].append(a["affinity"])

    return {
        "total_assignments": len(assignments),
        "papers_covered": len(per_paper),
        "global_avg_affinity": round(float(np.mean(all_aff)), 4),
        "global_min_affinity": round(float(np.min(all_aff)), 4),
        "avg_affinity_by_quality": {
            q: round(float(np.mean(scores)), 4)
            for q, scores in avg_by_quality.items()
        },
        "per_paper": per_paper,
    }


def compute_paper_level_metrics(
    review_scores: list[dict],
    target_paper_ids: set,
    paper_quality_map: dict[str, str],
) -> dict:
    """
    Compute paper-level detection metrics.

    Measures score deviation of TARGET papers relative to same-quality
    NON-target papers. This is the key signal that global correlation ρ
    misses: collusion inflates target papers uniformly regardless of quality,
    so the paper-level deviation is the sharper detection metric.

    Args:
        review_scores: list of {paper_id, score, quality, behavior_type, ...}
        target_paper_ids: set of paper IDs that are malicious targets.
        paper_quality_map: {paper_id: quality}.

    Returns:
        dict with:
          - target_score_deviation: mean(target score) - mean(non-target
            score), broken down by quality level.
          - per_paper: {paper_id: {avg_score, is_target, quality}}
    """
    # Group average scores by paper
    score_by_paper = defaultdict(list)
    for s in review_scores:
        score_by_paper[s["paper_id"]].append(s["score"])

    per_paper_avg = {
        pid: float(np.mean(scores))
        for pid, scores in score_by_paper.items()
    }

    # Split into target vs non-target, grouped by quality
    target_scores_by_quality = defaultdict(list)
    nontarget_scores_by_quality = defaultdict(list)

    for pid, avg_score in per_paper_avg.items():
        quality = paper_quality_map.get(pid, "borderline")
        if pid in target_paper_ids:
            target_scores_by_quality[quality].append(avg_score)
        else:
            nontarget_scores_by_quality[quality].append(avg_score)

    # Per-quality deviation: Δ = mean(target) - mean(non-target)
    deviation_by_quality = {}
    for quality in ["strong_accept", "accept", "borderline", "reject"]:
        t = target_scores_by_quality.get(quality, [])
        nt = nontarget_scores_by_quality.get(quality, [])
        if t and nt:
            deviation_by_quality[quality] = round(
                float(np.mean(t) - np.mean(nt)), 4
            )
        elif t:
            deviation_by_quality[quality] = None  # no non-target baseline

    # Global deviation
    all_target = [v for vals in target_scores_by_quality.values() for v in vals]
    all_nontarget = [v for vals in nontarget_scores_by_quality.values() for v in vals]
    global_deviation = (
        round(float(np.mean(all_target) - np.mean(all_nontarget)), 4)
        if all_target and all_nontarget else None
    )

    return {
        "target_score_deviation": global_deviation,
        "deviation_by_quality": deviation_by_quality,
        "n_target_papers": len(all_target),
        "n_nontarget_papers": len(all_nontarget),
        "per_paper": {
            pid: {
                "avg_score": round(avg, 2),
                "is_target": pid in target_paper_ids,
                "quality": paper_quality_map.get(pid, "borderline"),
            }
            for pid, avg in per_paper_avg.items()
        },
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run paper-reviewer matching")
    parser.add_argument("--affinity", type=str,
                       default="config/affinity/affinity_100.json")
    parser.add_argument("--bids", type=str,
                       default="config/simulation/collusion-20/bids.json")
    parser.add_argument("--papers", type=str,
                       default="config/papers/papers_100_graded.json")
    parser.add_argument("--output", type=str,
                       default="config/match")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load data
    with open(args.affinity, "r", encoding="utf-8") as f:
        affinity_data = json.load(f)
    with open(args.bids, "r", encoding="utf-8") as f:
        bid_data = json.load(f)
    with open(args.papers, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Build lookup dicts
    affinity_scores = {
        (s["paper_id"], s["reviewer_id"]): s["score"]
        for s in affinity_data["scores"]
    }
    bid_scores = {
        (b["reviewer_id"], b["paper_id"]): b["score"]
        for b in bid_data["bids"]
    }
    # Normalize bid keys: in bids it's (reviewer_id, paper_id)
    bid_scores_norm = {}
    for b in bid_data["bids"]:
        bid_scores_norm[(b["paper_id"], b["reviewer_id"])] = b["score"]

    reviewer_ids = sorted(set(
        s["reviewer_id"] for s in affinity_data["scores"]
    ))
    coi_pairs = build_coi_pairs(papers)

    logger.info("Data: %d papers, %d reviewers, %d COI pairs",
               len(papers), len(reviewer_ids), len(coi_pairs))

    # ── Baseline A: Affinity-Only ────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Baseline A: Affinity-Only (w_affinity=1.0, w_bid=0)")
    logger.info("=" * 60)

    assign_a = greedy_match(
        papers, reviewer_ids, affinity_scores, bid_scores_norm,
        coi_pairs, w_affinity=1.0, w_bid=0,
    )
    metrics_a = compute_metrics(assign_a)

    logger.info("Assignments: %d | Papers covered: %d/%d",
               metrics_a["total_assignments"],
               metrics_a["papers_covered"], len(papers))
    logger.info("Avg affinity: %.4f (min: %.4f)",
               metrics_a["global_avg_affinity"],
               metrics_a["global_min_affinity"])
    logger.info("By quality:")
    for q, avg in metrics_a["avg_affinity_by_quality"].items():
        logger.info("  %s: %.4f", q, avg)

    # ── Baseline B: Bid-Influenced ───────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Baseline B: Bid-Influenced (w_affinity=1.0, w_bid=2.0)")
    logger.info("=" * 60)

    assign_b = greedy_match(
        papers, reviewer_ids, affinity_scores, bid_scores_norm,
        coi_pairs, w_affinity=1.0, w_bid=2.0,
    )
    metrics_b = compute_metrics(assign_b)

    logger.info("Assignments: %d | Papers covered: %d/%d",
               metrics_b["total_assignments"],
               metrics_b["papers_covered"], len(papers))
    logger.info("Avg affinity: %.4f (min: %.4f)",
               metrics_b["global_avg_affinity"],
               metrics_b["global_min_affinity"])
    logger.info("By quality:")
    for q, avg in metrics_b["avg_affinity_by_quality"].items():
        logger.info("  %s: %.4f", q, avg)

    # ── Comparison ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Comparison: Affinity-Only vs Bid-Influenced")
    logger.info("=" * 60)

    # Per-paper affinity change
    logger.info("\nPer-paper avg affinity change (B - A):")
    changes = []
    for pid in sorted(metrics_a["per_paper"].keys()):
        avg_a = metrics_a["per_paper"][pid]["avg_affinity"]
        avg_b = metrics_b["per_paper"].get(pid, {}).get("avg_affinity", 0)
        diff = avg_b - avg_a
        changes.append(diff)
        quality = metrics_a["per_paper"][pid]["quality"]
        direction = "↑" if diff > 0.005 else ("↓" if diff < -0.005 else "=")
        logger.info("  %s (%s): %.4f → %.4f  %+.4f %s",
                   pid, quality, avg_a, avg_b, diff, direction)

    logger.info("\nSummary:")
    logger.info("  Avg change: %+.4f", float(np.mean(changes)))
    logger.info("  Papers improved: %d/%d",
               sum(1 for c in changes if c > 0.005), len(changes))
    logger.info("  Papers degraded:  %d/%d",
               sum(1 for c in changes if c < -0.005), len(changes))

    # ── Save ─────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "assignments_affinity_only.json", "w", encoding="utf-8") as f:
        json.dump({"config": "affinity-only", "metrics": metrics_a,
                   "assignments": assign_a}, f, indent=2, ensure_ascii=False)

    with open(output_dir / "assignments_bid_influenced.json", "w", encoding="utf-8") as f:
        json.dump({"config": "bid-influenced", "metrics": metrics_b,
                   "assignments": assign_b}, f, indent=2, ensure_ascii=False)

    # Comparison summary
    comparison = {
        "affinity_only": {
            "avg_affinity": metrics_a["global_avg_affinity"],
            "min_affinity": metrics_a["global_min_affinity"],
            "by_quality": metrics_a["avg_affinity_by_quality"],
        },
        "bid_influenced": {
            "avg_affinity": metrics_b["global_avg_affinity"],
            "min_affinity": metrics_b["global_min_affinity"],
            "by_quality": metrics_b["avg_affinity_by_quality"],
        },
        "per_paper_changes": {
            pid: {
                "quality": metrics_a["per_paper"][pid]["quality"],
                "affinity_only": metrics_a["per_paper"][pid]["avg_affinity"],
                "bid_influenced": metrics_b["per_paper"].get(pid, {}).get("avg_affinity", 0),
                "change": round(
                    metrics_b["per_paper"].get(pid, {}).get("avg_affinity", 0)
                    - metrics_a["per_paper"][pid]["avg_affinity"], 4,
                ),
            }
            for pid in sorted(metrics_a["per_paper"].keys())
        },
    }
    with open(output_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info("\nResults saved → %s", output_dir)


if __name__ == "__main__":
    main()
