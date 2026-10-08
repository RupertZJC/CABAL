#!/usr/bin/env python3
"""Port the malicious-bidding detectors from Jecmen et al. (WWW 2023)
"A Dataset on Malicious Paper Bidding in Peer Review" onto ColluBid runs.

The reference repo (github.com/sjecmen/malicious_bidding_dataset, analysis/
analysis.py) ranks reviewers by suspiciousness using only the bid matrix B,
the author-conflict matrix M, and -- for cluster detection -- the paper each
reviewer authored. Detection is unsupervised: ground truth (which reviewers are
colluders, read from personas.json) is used only to score the detectors.

Detectors
  simple_detect    #refusals - #eager bids per reviewer (conflicts zeroed)
  cluster_detect   reciprocal positive bids on each other's authored papers
  low_rank_detect  deviation from a rank-r SVD approximation of the bid matrix

ColluBid adaptations
  * Bids are sparse (top-20 non-COI pool) and 5-level {2,1,0,-1,-100}; they are
    mapped onto the paper's {-1,0,1} scale: VeryHigh/High -> +1, Low/VeryLow
    -> -1, Neutral and never-bid -> 0. Pass --bid-scale raw to feed the
    detectors the original 5-level scores instead.
  * cluster_detect is generalized to reviewers who co-authored >1 paper by
    taking the max reciprocal score over all authored-paper pairs.

Metric (same as the paper): Average Detection Rank -- the position of each
colluder in the most-suspicious-first ordering (0 = most suspicious),
normalized by the number of ranked reviewers so that 0.5 = chance. A detector
that captures the bid-phase signal ranks colluders well below 0.5; the honest
reviewers' mean rank is reported alongside for comparison.

--protocol full       rank all reviewers once on the full reviewer x paper
                      matrix (deterministic)
--protocol subsample  replicate the paper's protocol: subsample honest
                      reviewers so the matrix is square (#rev = #pap) and
                      average per-colluder ranks over --num-trials trials

Usage (from the ColluBid/ root, so the default config/ paths resolve):
  python baseline_detection/detect_baselines.py
  python baseline_detection/detect_baselines.py --scenarios collusion-50 --protocol subsample
  python baseline_detection/detect_baselines.py --base-dir config/simulation/runs/seed0921 \
      --json baseline_detection/results/detect_baseline_report.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.linalg

LABEL_BIN = {"Very High": 1, "High": 1, "Neutral": 0, "Low": -1, "Very Low": -1}


# ── Data loading ──────────────────────────────────────────────────────────

def load_matrices(base_dir: Path, scenario: str, papers_path: str,
                  unbid: float, bid_scale: str) -> dict:
    """Build dense bid/conflict matrices, reviewer/paper order, and ground truth."""
    bids = json.load(open(base_dir / scenario / "bids.json", encoding="utf-8"))["bids"]
    personas = json.load(open(base_dir / scenario / "personas.json", encoding="utf-8"))
    papers = json.load(open(papers_path, encoding="utf-8"))

    reviewer_ids = [p["reviewer_id"] for p in personas]
    paper_ids = [p["paper_id"] for p in papers]
    ridx = {r: i for i, r in enumerate(reviewer_ids)}
    pidx = {p: i for i, p in enumerate(paper_ids)}

    nrev, npap = len(reviewer_ids), len(paper_ids)
    B = np.full((nrev, npap), unbid, dtype=float)
    for b in bids:
        ri, pi = ridx.get(b["reviewer_id"]), pidx.get(b["paper_id"])
        if ri is None or pi is None:
            continue
        B[ri, pi] = b["score"] if bid_scale == "raw" else LABEL_BIN.get(b["label"], 0.0)

    # conflict matrix: reviewer authored the paper (a reviewer can co-author
    # up to 2 papers in this dataset)
    M = np.zeros((nrev, npap))
    authored_by = defaultdict(list)
    for p in papers:
        for a in p.get("author_profile_ids", []):
            ri, pi = ridx.get(a), pidx.get(p["paper_id"])
            if ri is None or pi is None:
                continue
            M[ri, pi] = 1
            authored_by[ri].append(pi)

    colluder_p = [p for p in personas if p["behavior_type"] != "honest"]
    return {
        "B": B, "M": M,
        "authored_by": {k: sorted(v) for k, v in authored_by.items()},
        "colluders": [ridx[p["reviewer_id"]] for p in colluder_p],
        "colluder_info": [{k: p[k] for k in ("reviewer_id", "stealth", "ring_id")}
                          for p in colluder_p],
        "reviewer_ids": reviewer_ids,
        "nrev": nrev, "npap": npap,
    }


# ── Detectors (faithful ports, return rank 0 = most suspicious) ────────────

def _rank_descending(scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(len(scores))
    return ranks


def simple_detect(B, M):
    Bz = np.where(M == 1, 0.0, B)
    num_pos = (Bz > 0).sum(axis=1)
    num_neg = (Bz < 0).sum(axis=1)
    return _rank_descending(num_neg - num_pos)


def cluster_detect(B, M, authored_by):
    Bz = np.where(M == 1, 0.0, B)
    nrev, npap = Bz.shape
    bidsums = Bz.sum(axis=1)
    scores = np.zeros(nrev)
    for i in range(nrev):
        auth_i = authored_by.get(i, [])
        if not auth_i:
            continue
        for j in range(i):
            auth_j = authored_by.get(j, [])
            if not auth_j:
                continue
            best = -np.inf
            for pi in auth_i:
                for pj in auth_j:
                    # original formula, applied per authored-paper pair:
                    # score = recip(positive mutual bids) - normalized bids elsewhere
                    recip = max(Bz[j, pi], 0.0) + max(Bz[i, pj], 0.0)
                    i_out = bidsums[i] - Bz[i, pj]
                    j_out = bidsums[j] - Bz[j, pi]
                    best = max(best, recip - (i_out + j_out) / (2.0 * npap))
            scores[i] = max(scores[i], best)
            scores[j] = max(scores[j], best)
    return _rank_descending(scores)


def low_rank_detect(B, M, rank=3, method="sum"):
    Bz = np.where(M == 1, 0.0, B)
    U, s, Vh = scipy.linalg.svd(Bz, full_matrices=False)
    s = s.copy()
    if rank is not None and len(s) > rank:
        s[rank:] = 0.0
    L = (U * s) @ Vh
    delta = np.abs(L - Bz)
    delta = np.where(M == 1, 0.0, delta)
    if method == "max":
        diffs = delta.max(axis=1)
    else:
        diffs = delta.sum(axis=1)
    return _rank_descending(diffs)


# Uniform (B, M, authored_by, rank, method) -> ranks signature for evaluate().
def _simple_wrap(B, M, authored, rank, method):
    return simple_detect(B, M)


def _cluster_wrap(B, M, authored, rank, method):
    return cluster_detect(B, M, authored)


def _low_rank_wrap(B, M, authored, rank, method):
    return low_rank_detect(B, M, rank, method)


# ── Evaluation ─────────────────────────────────────────────────────────────

def _run_trials(data: dict, target_ids: list, detector, protocol: str,
                num_trials: int, seed: int, rank: int, low_method: str):
    """Rank reviewers per trial and accumulate normalized ranks for the target
    reviewers (target_ids, a list of reviewer_id strings) and for everyone else
    in the same matrix. Returns
    (per_target_avg_rank, honest_mean_rank, topK_hit_rate, n_total)."""
    rng = np.random.default_rng(seed)
    B, M, authored = data["B"], data["M"], data["authored_by"]
    ridx = {r: i for i, r in enumerate(data["reviewer_ids"])}
    tgt = [ridx[r] for r in target_ids if r in ridx]
    nrev, npap = data["nrev"], data["npap"]
    n_total = nrev if protocol == "full" else npap

    def run(Bm, Mm, am):
        return detector(Bm, Mm, am, rank, low_method)

    acc, hon_acc, hits = [[] for _ in tgt], [], []
    if protocol == "full":
        ranks = run(B, M, authored)
        nrm = ranks / n_total
        for k, i in enumerate(tgt):
            acc[k].append(nrm[i])
        tset = set(tgt)
        hon_acc = nrm[[r for r in range(nrev) if r not in tset]].tolist()
        K = len(tgt)
        hits = [len(set(np.argsort(ranks)[:K].tolist()) & tset) / K]
    else:  # subsample: all targets + random others to make the matrix square
        tset = set(tgt)
        for _ in range(num_trials):
            honest = [r for r in range(nrev) if r not in tset]
            need = npap - len(tgt)
            if need < 0 or need > len(honest):
                raise ValueError(
                    f"cannot subsample: #targets {len(tgt)} vs #papers {npap}")
            chosen = tgt + list(rng.choice(honest, size=need, replace=False))
            rng.shuffle(chosen)
            new_idx = {old: new for new, old in enumerate(chosen)}
            ranks = run(B[chosen, :], M[chosen, :],
                        {new_idx[o]: v for o, v in authored.items() if o in new_idx})
            nrm = ranks / n_total
            ntgt = [new_idx[i] for i in tgt]
            for k, ni in enumerate(ntgt):
                acc[k].append(nrm[ni])
            nset = set(ntgt)
            hon_acc.extend(nrm[[r for r in range(npap) if r not in nset]].tolist())
            K = len(tgt)
            hits.append(len(set(np.argsort(ranks)[:K].tolist()) & nset) / K)

    per = np.array([np.mean(a) for a in acc]) if acc else np.array([])
    return per, (float(np.mean(hon_acc)) if hon_acc else None), \
        (float(np.mean(hits)) if hits else None), n_total


def evaluate(data: dict, base_data: dict | None, colluder_ids: list,
             detector, protocol: str, num_trials: int, seed: int,
             rank: int, low_method: str) -> dict:
    """Detection stats for a collusion scenario, plus a paired comparison
    against the all-honest baseline scenario: the SAME colluders' suspiciousness
    rank when they bid honestly (baseline run) vs when they collude (collusion
    run). The delta isolates the collusion-driven signal from stable traits
    (e.g. benign co-authorship reciprocity), mirroring the M2 paired-baseline
    confound control."""
    per, hon, hit, n_total = _run_trials(
        data, colluder_ids, detector, protocol, num_trials, seed, rank, low_method)
    out = {
        "n_total": n_total,
        "n_colluders": int(len(per)),
        "colluder_avg_norm_rank": float(per.mean()) if len(per) else None,
        "colluder_std_norm_rank": float(per.std()) if len(per) else None,
        "honest_avg_norm_rank": hon,
        "topK_hit_rate": hit,
        "colluder_ranks": sorted(per.tolist()),
    }
    if base_data is not None:
        base_per, _, _, _ = _run_trials(
            base_data, colluder_ids, detector, protocol, num_trials,
            seed + 1, rank, low_method)
        if len(base_per) and len(per):
            out.update({
                "colluder_avg_norm_rank_baseline": float(base_per.mean()),
                "colluder_std_norm_rank_baseline": float(base_per.std()),
                # negative = colluders rank more suspicious when colluding than
                # when honest → detector picks up the collusion-specific signal
                "colluder_delta_vs_own_baseline": float(per.mean() - base_per.mean()),
            })
    return out


# ── CLI / orchestration ───────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=str, default="collusion-20,collusion-50")
    ap.add_argument("--base-dir", type=str, default="config/simulation")
    ap.add_argument("--papers", type=str, default="config/papers/papers_100.json")
    ap.add_argument("--protocol", type=str, choices=["full", "subsample"],
                    default="full")
    ap.add_argument("--num-trials", type=int, default=100,
                    help="trials for --protocol subsample (paper uses 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=3,
                    help="rank threshold for low_rank_detect (paper uses 3)")
    ap.add_argument("--low-rank-method", type=str, choices=["sum", "max"],
                    default="sum")
    ap.add_argument("--bid-scale", type=str, choices=["disc", "raw"],
                    default="disc",
                    help="disc = map bids to {-1,0,1} (paper's native scale); "
                         "raw = feed detectors the original 5-level scores")
    ap.add_argument("--unbid", type=float, default=0.0,
                    help="value for papers a reviewer never bid on (paper's "
                         "reviewers bid on every paper, so this is a choice)")
    ap.add_argument("--baseline-scenario", type=str, default="baseline",
                    help="all-honest scenario used for the paired comparison; "
                         "pass 'none' to disable it")
    ap.add_argument("--json", type=str, default=None,
                    help="write the full report as JSON")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    report = {"protocol": args.protocol, "bid_scale": args.bid_scale,
              "num_trials": args.num_trials, "rank": args.rank,
              "results": {}}
    for s in [x.strip() for x in args.scenarios.split(",") if x.strip()]:
        data = load_matrices(base_dir, s, args.papers,
                             unbid=args.unbid, bid_scale=args.bid_scale)
        if not data["colluders"]:
            print(f"[skip] {s}: no colluders (baseline scenario)")
            continue
        base_data = None
        if args.baseline_scenario.lower() != "none":
            base_path = base_dir / args.baseline_scenario
            if (base_path / "bids.json").exists():
                base_data = load_matrices(
                    base_dir, args.baseline_scenario, args.papers,
                    unbid=args.unbid, bid_scale=args.bid_scale)
            else:
                print(f"[warn] no baseline scenario at {base_path}; "
                      f"skipping paired comparison")
        colluder_ids = [c["reviewer_id"] for c in data["colluder_info"]]
        report["results"][s] = {}
        for name, fn in [("simple_detect", _simple_wrap),
                         ("cluster_detect", _cluster_wrap),
                         ("low_rank_detect", _low_rank_wrap)]:
            res = evaluate(data, base_data, colluder_ids, fn, args.protocol,
                           args.num_trials, args.seed, args.rank,
                           args.low_rank_method)
            report["results"][s][name] = res

    print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.json}")


def print_report(report: dict) -> None:
    W = 100
    print("=" * W)
    print("Malicious-bidding detection baselines (Jecmen et al. WWW 2023) on ColluBid")
    print(f"  protocol={report['protocol']}  bid_scale={report['bid_scale']}  "
          f"low_rank rank={report['rank']}")
    print("  normalized detection rank: 0 = most suspicious, 0.5 = chance; "
          "lower colluder rank = better detection")
    print("  'col baseline' = the same colluders' rank in the all-honest baseline run;")
    print("  delta = collusion rank - own baseline rank (negative = collusion-specific signal)")
    print("=" * W)
    hdr = (f"{'scenario':<12}{'detector':<16}{'col avg':>9}{'col base':>9}"
           f"{'delta':>9}{'honest avg':>12}{'top-K hit':>10}")
    print(hdr)
    print("-" * W)
    for s, dets in report["results"].items():
        for name, r in dets.items():
            def fmt(k, fspec=".3f"):
                v = r.get(k)
                return format(v, fspec) if v is not None else "n/a"
            col = fmt("colluder_avg_norm_rank")
            colb = fmt("colluder_avg_norm_rank_baseline")
            delta = fmt("colluder_delta_vs_own_baseline", "+.3f")
            hon = fmt("honest_avg_norm_rank")
            hit = (f"{100*r['topK_hit_rate']:.1f}%" if r["topK_hit_rate"] is not None
                   else "n/a")
            print(f"{s:<12}{name:<16}{col:>9}{colb:>9}{delta:>9}{hon:>12}{hit:>10}")
    print("=" * W)


if __name__ == "__main__":
    main()
