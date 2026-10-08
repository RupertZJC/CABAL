#!/usr/bin/env python3
"""Port the reviewer-author collusion-ring detection baselines from
Jecmen et al. (2024) "On the Detection of Reviewer-Author Collusion Rings From
Paper Bidding" (arXiv:2402.07860, repo sjecmen/peer-review-collusion-detection)
onto ColluBid runs.

Core idea of the reference: model the collusion ring as a DENSE SUBGRAPH in the
reviewer->reviewer "bid-author" graph BA[i,j] = 1 iff reviewer i submitted a
positive bid on some paper authored by reviewer j. Detection = find the densest
quasi-clique / block in BA (unipartite) or in the reviewer x paper bid matrix
(bipartite, fraudar). Detection is unsupervised; ground truth (colluders and
rings from personas.json) is used only to score the detectors.

Detectors (faithful ports of the reference repo; densest_subgraph uses the
Charikar greedy 1/2-approximation because the reference's exact version needs
gurobipy, which is not installed):
  oqc_greedy        greedy peeling, maximize edges - alpha*k(k-1), alpha=1/3
  oqc_local         local search from a triangle heuristic start + random starts
  telltail          TellTail (Hooi et al. 2020) tail-3 objective, local search
  densest_subgraph  Charikar greedy max-density subgraph
  fraudar           Fraudar log-weighted average degree on the bid bipartite
                    graph (reviewer x paper)

ColluBid adaptations / design choices
  * Reference bids are binary yes/maybe. ColluBid bids are 5-level
    {2,1,0,-1,-100} (VH/H/N/L/VL) and sparse (top-20 non-COI pool). The BA edge
    threshold is a parameter --bid-thresh: thresh=1 counts any positive bid
    (H/VH, the reference semantics), thresh=2 counts only Very High bids (the
    collusion-specific inflation signal).
  * Detectors return a SET of reviewers (the one densest block), not a ranking.
    Evaluation therefore uses the reference's native metrics: precision /
    recall / Jaccard of the detected set vs the true colluder set, plus recall
    of the best-single-ring.
  * Paired honest-baseline: the same detector is run on the all-honest
    "baseline" scenario graph. Its detected set there is a pure false-positive
    probe -- a "ring" flagged when nobody colludes. If its overlap with the
    (eventual) colluders is high, the detector is picking up stable structural
    traits (co-authorship reciprocity), not collusion itself.

Usage (from the ColluBid/ root, so the default config/ paths resolve):
  python baseline_detection/detect_ring_baselines.py
  python baseline_detection/detect_ring_baselines.py --bid-thresh 2 --methods oqc_local,telltail
  python baseline_detection/detect_ring_baselines.py --base-dir config/simulation/runs/seed0921 \
      --json baseline_detection/results/detect_ring_report.json
"""

import argparse
import hashlib
import heapq
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

import scipy as _scipy  # for __version__ in provenance

ALPHA = 1 / 3


def _repo_head(path: Path):
    """ColluBid git HEAD if inside a repo, else None (never raises)."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip() or None
    except Exception:
        return None


def _provenance(base_dir: Path, papers_path: str) -> dict:
    """Reproducibility metadata: runtime versions + sha256 of the input data.
    Written into the report JSON so results are self-describing."""
    files = []
    if base_dir.is_dir():
        for sc in sorted(p.name for p in base_dir.iterdir()
                         if (p / "bids.json").exists()):
            files += [base_dir / sc / "bids.json", base_dir / sc / "personas.json"]
    files.append(Path(papers_path))
    h = hashlib.sha256()
    for f in sorted(set(files), key=str):
        h.update(f.read_bytes())
    return {
        "python": platform.python_version(),
        "numpy": getattr(np, "__version__", None),
        "scipy": getattr(_scipy, "__version__", None),
        "collubid_git_head": _repo_head(Path(__file__).resolve().parents[1]),
        "data_sha256": h.hexdigest(),
        "data_files": [str(f) for f in sorted(set(files), key=str)],
    }

# ── Data loading ──────────────────────────────────────────────────────────

def load_matrices(base_dir: Path, scenario: str, papers_path: str) -> dict:
    """Reviewer x paper score matrix S, paper x reviewer authorship A, and
    ground truth (colluder indices, ring membership)."""
    bids = json.load(open(base_dir / scenario / "bids.json", encoding="utf-8"))["bids"]
    personas = json.load(open(base_dir / scenario / "personas.json", encoding="utf-8"))
    papers = json.load(open(papers_path, encoding="utf-8"))

    reviewer_ids = [p["reviewer_id"] for p in personas]
    paper_ids = [p["paper_id"] for p in papers]
    ridx = {r: i for i, r in enumerate(reviewer_ids)}
    pidx = {p: i for i, p in enumerate(paper_ids)}
    nrev, npap = len(reviewer_ids), len(paper_ids)

    S = np.zeros((nrev, npap))
    for b in bids:
        ri, pi = ridx.get(b["reviewer_id"]), pidx.get(b["paper_id"])
        if ri is None or pi is None:
            continue
        S[ri, pi] = b["score"]

    A = np.zeros((npap, nrev))
    authored_by = defaultdict(list)
    for p in papers:
        for a in p.get("author_profile_ids", []):
            ri, pi = ridx.get(a), pidx.get(p["paper_id"])
            if ri is None or pi is None:
                continue
            A[pi, ri] = 1
            authored_by[ri].append(pi)

    colluder_p = [p for p in personas if p["behavior_type"] != "honest"]
    rings = defaultdict(list)
    for p in colluder_p:
        rings[p.get("ring_id")].append(ridx[p["reviewer_id"]])

    return {
        "S": S, "A": A,
        "colluders": sorted(ridx[p["reviewer_id"]] for p in colluder_p),
        "rings": {k: sorted(v) for k, v in rings.items()},
        "authored_by": {k: sorted(v) for k, v in authored_by.items()},
        "nrev": nrev, "npap": npap,
        "reviewer_ids": reviewer_ids,
    }


# ── Graph construction ────────────────────────────────────────────────────

def build_BA(data: dict, thresh: float) -> np.ndarray:
    """Directed reviewer graph: BA[i,j] = 1 iff i bid >= thresh on a paper
    authored by j (the reference's make_bid_auth, thresholded on score)."""
    mask = (data["S"] >= thresh).astype(int)
    BA = ((mask @ data["A"]) > 0).astype(int)
    np.fill_diagonal(BA, 0)
    return BA


# ── Detectors (faithful ports; each returns a list of reviewer indices) ────

def _directed_edges(G, idx):
    if len(idx) < 2:
        return 0
    return int(G[np.ix_(idx, idx)].sum())


def greedy_peeling(G):
    """Port of oqc.greedy_peeling: remove min-(in+out)-degree vertex in order,
    return the nested subsets and their internal directed edge counts."""
    n = G.shape[0]
    degrees = (G.sum(axis=0) + G.sum(axis=1)).astype(int)
    deg = degrees.copy()
    buckets = defaultdict(list)
    for i in range(n):
        buckets[int(deg[i])].append(i)
    # Multiset adjacency as in the reference oqc.greedy_peeling: each DIRECTED
    # edge i->j adds j to adj[i] AND i to adj[j], so a mutual edge appears twice
    # and removing v decrements a mutual neighbor twice (one per directed edge).
    adj = [list(np.flatnonzero(G[i])) + list(np.flatnonzero(G[:, i]))
           for i in range(n)]

    order = []
    removed = [False] * n
    min_degree = 0
    while len(order) < n:
        while not buckets[min_degree]:
            min_degree += 1
        v = buckets[min_degree].pop()
        deg[v] = -1
        removed[v] = True
        order.append(v)
        for u in adj[v]:
            if deg[u] == -1:
                continue
            buckets[deg[u]].remove(u)
            deg[u] -= 1
            buckets[deg[u]].append(u)
            if deg[u] < min_degree:
                min_degree = deg[u]

    subsets, edge_counts = [], []
    cur, ec = [], 0
    for v in reversed(order):
        ec += int(G[v, cur].sum()) + int(G[cur, v].sum())
        cur.append(v)
        subsets.append(cur.copy())
        edge_counts.append(ec)
    return subsets, edge_counts


def oqc_greedy(G, rng=None, alpha=ALPHA):
    """Port of oqc.oqc_greedy: best subset under f(S)=edges(S)-alpha*k*(k-1).
    Tie-break matches the reference `max(zip(fs, subsets))`, which on an f-tie
    prefers the lexicographically LARGER subset (nested subsets -> the longer
    one), not the first argmax."""
    subsets, edge_counts = greedy_peeling(G)
    fs = [ec - alpha * len(S) * (len(S) - 1) for S, ec in zip(subsets, edge_counts)]
    k_star = max(range(len(subsets)), key=lambda k: (fs[k], subsets[k]))
    return subsets[k_star], "alpha_%.3f" % alpha


def get_local_search_init(G):
    """Port of detection_eval.get_local_search_init: start at the vertex with
    max triangles / degree."""
    degs = G.sum(axis=0) + G.sum(axis=1)
    triangles = np.diag(G @ G @ G)
    with np.errstate(divide="ignore", invalid="ignore"):
        heuristic = np.divide(triangles, degs, out=np.zeros(degs.shape),
                              where=(degs > 0)).argmax()
    return int(heuristic)


def _oqc_f(G, S_bool, alpha):
    k = int(S_bool.sum())
    if k < 2:
        return 0.0
    idx = np.flatnonzero(S_bool)
    return float(G[np.ix_(idx, idx)].sum()) - alpha * k * (k - 1)


def oqc_local_v(G, S, alpha, max_iter):
    """Port of oqc.oqc_local_v: alternating add/remove local search on f."""
    n = G.shape[0]
    t = 0
    S_prev = None
    while True:
        while True:
            change = False
            for u in range(n):
                if not S[u]:
                    f_cur = _oqc_f(G, S, alpha)
                    S[u] = True
                    if _oqc_f(G, S, alpha) >= f_cur:
                        change = True
                    else:
                        S[u] = False
            if not change:
                break
        change = False
        for u in range(n):
            if S[u]:
                f_cur = _oqc_f(G, S, alpha)
                S[u] = False
                if _oqc_f(G, S, alpha) >= f_cur:
                    change = True
                    break
                else:
                    S[u] = True
        t += 1
        if (not change) or (max_iter is not None and t >= max_iter) or \
                (S_prev is not None and np.array_equal(S, S_prev)):
            break
        S_prev = S.copy()
    if _oqc_f(G, ~S, alpha) > _oqc_f(G, S, alpha):
        S = ~S
    return np.flatnonzero(S).tolist(), _oqc_f(G, S, alpha)


def oqc_local(G, rng=None, alpha=ALPHA, num_rand_start=10, max_iter=50):
    """Port of oqc.oqc_local: best local-search result over heuristic + random
    starts."""
    n = G.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    start = get_local_search_init(G)
    neighbors = (G[:, start] + G[start, :]) > 0
    neighbors[start] = True
    starts = [neighbors.copy()]
    for _ in range(num_rand_start):
        v = int(rng.integers(0, n))
        s = np.zeros(n, dtype=bool)
        s[v] = True
        starts.append(s)
    best = None
    for S0 in starts:
        S, f = oqc_local_v(G, S0.copy(), alpha, max_iter)
        if best is None or f > best[1]:
            best = (S, f)
    return best[0], "alpha_%.3f" % alpha


def modularity_matrix(A):
    d = np.sum(A, axis=0)
    m = np.sum(d) / 2
    return A - np.outer(d, d) / (2 * m)


def metric_tail3(A, x):
    n = A.shape[0]
    k = int(np.sum(x))
    if k == 0 or k == n:
        return 0.0
    s = n // 2
    deg = np.sum(A, axis=0)
    m = np.sum(deg) / 2
    sumB = np.sum(deg ** 2) / (4 * m)
    sumB2 = m + (np.sum(deg ** 2) ** 2 - np.sum(deg ** 4)) / (8 * m ** 2) \
        - np.dot(deg, np.dot(A, deg)) / (2 * m)
    sumBrow2 = np.sum((deg ** 2 / (2 * m)) ** 2)
    p2 = s * (s - 1) / (n * (n - 1))
    p3 = s * (s - 1) * (s - 2) / (n * (n - 1) * (n - 2))
    p4 = s * (s - 1) * (s - 2) * (s - 3) / (n * (n - 1) * (n - 2) * (n - 3))
    Ymean = p2 * sumB
    wedgesum = sumBrow2 - 2 * sumB2
    Ymeansq = p2 * sumB2 + p3 * wedgesum + p4 * (sumB ** 2 - sumB2 - wedgesum)
    Ystd = np.sqrt(max(Ymeansq - Ymean ** 2, 0.0))
    adjsum = np.dot(x, np.dot(A, x)) / 2 - np.dot(x, deg) ** 2 / (4 * m) \
        + np.dot(x, deg ** 2) / (4 * m)
    beta, delta = 0.9, 0.8
    return k ** (-delta) * (adjsum - (Ymean + 1.28 * Ystd) * (k / s) ** beta)


def detect_telltail(A, S_start):
    n = A.shape[0]
    M = modularity_matrix(A)
    x = S_start.copy()
    deg = np.sum(M[x, :], axis=0).reshape(-1, 1)
    score = metric_tail3(A, x)
    while True:
        deg_add = deg.copy()
        deg_add[x] = np.nan
        deg_del = deg.copy()
        deg_del[~x] = np.nan
        try:
            idx_add = int(np.nanargmax(deg_add))
        except ValueError:
            idx_add = None
        try:
            idx_del = int(np.nanargmin(deg_del))
        except ValueError:
            idx_del = None
        x_add = x.copy()
        if idx_add is not None:
            x_add[idx_add] = 1
        x_del = x.copy()
        if idx_del is not None:
            x_del[idx_del] = 0
        score_add = metric_tail3(A, x_add)
        score_del = metric_tail3(A, x_del)
        if np.sum(x) == 0:
            score_del = -np.inf
        if np.sum(x) == n:
            score_add = -np.inf
        if score >= score_add and score >= score_del:
            break
        elif score_add >= score_del:
            deg = deg + M[:, idx_add].reshape(-1, 1)
            x = x_add
            score = score_add
        else:
            deg = deg - M[:, idx_del].reshape(-1, 1)
            x = x_del
            score = score_del
    return x, score


def run_telltail(G, rng=None, num_rand_start=10):
    """Port of telltail.run_telltail; on a directed BA graph the intersection
    G*G^T (mutual bids) is used, as in the reference."""
    if np.any(G != G.T):
        G_undir = np.clip(G * G.T, 0, 1)
    else:
        G_undir = G
    n = G_undir.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    start = get_local_search_init(G_undir)
    neighbors = (G_undir[:, start] + G_undir[start, :]) > 0
    neighbors[start] = True
    starts = [neighbors.copy()]
    for _ in range(num_rand_start):
        starts.append(rng.random(n) > rng.random(1))
    best = None
    for S0 in starts:
        x, score = detect_telltail(G_undir, S0)
        if best is None or score > best[1]:
            best = (x, score)
    return np.flatnonzero(best[0]).tolist(), "telltail"


def densest_subgraph(G, rng=None):
    """Charikar greedy densest subgraph (1/2-approximation). The reference uses
    an exact LP requiring gurobipy (not installed). Maximizes edges/|S|."""
    n = G.shape[0]
    alive = set(range(n))
    deg = {i: int(G[i].sum() + G[:, i].sum()) for i in range(n)}
    best = (-1.0, None)
    while alive:
        # density = INTERNAL directed edges / |S|, the same objective as the
        # reference gurobi LP. sum(deg)//2 would count boundary edges; the
        # independent Charikar in verify_independent.py recomputes G[ix,ix].sum()
        # directly, so use the identical quantity here for set-level agreement.
        ai = list(alive)
        edges = int(G[np.ix_(ai, ai)].sum())
        density = edges / len(ai)
        if density > best[0]:
            best = (density, set(alive))
        v = min(alive, key=lambda i: deg[i])
        alive.remove(v)
        for u in list(alive):
            if G[v, u] or G[u, v]:
                # directed edge accounting: one decrement per directed edge
                # (v->u and/or u->v), so a mutual pair decrements by 2
                deg[u] -= int(G[v, u]) + int(G[u, v])
    return sorted(best[1]), "greedy_charikar"


class _MinHeap:
    """Lazy-deletion min-heap over explicit values (replaces MinTree)."""

    def __init__(self, values):
        self.vals = np.array(values, dtype=float).copy()
        self.heap = [(float(self.vals[i]), i) for i in range(len(self.vals))]
        heapq.heapify(self.heap)

    def get_min(self):
        while True:
            v, i = self.heap[0]
            if self.vals[i] == v:
                return i, v
            heapq.heappop(self.heap)

    def change_val(self, i, delta):
        self.vals[i] += delta
        heapq.heappush(self.heap, (float(self.vals[i]), i))


def fast_greedy_decreasing(W, col_weights):
    """Port of fraudar.greedy.fastGreedyDecreasing: peel the min-weight node
    (row or col), track average score = total_weight / (|rows|+|cols|)."""
    m, n = W.shape
    Ml = W.tolil()
    Mlt = W.transpose().tolil()
    row_set = set(range(m))
    col_set = set(range(n))
    cur_score = float(W.sum())
    best_score = cur_score / (m + n)
    best_num_deleted = 0
    row_deltas = np.asarray(W.sum(axis=1)).ravel().astype(float)
    col_deltas = np.asarray(W.sum(axis=0)).ravel().astype(float)
    row_tree = _MinHeap(row_deltas)
    col_tree = _MinHeap(col_deltas)
    deleted = []
    num = 0
    while row_set and col_set:
        nr, rd = row_tree.get_min()
        nc, cd = col_tree.get_min()
        if rd <= cd:
            cur_score -= rd
            for j in Ml.rows[nr]:
                col_tree.change_val(j, -col_weights[j])
            row_set.discard(nr)
            row_tree.change_val(nr, float("inf"))
            deleted.append((True, nr))
        else:
            cur_score -= cd
            for i in Mlt.rows[nc]:
                row_tree.change_val(i, -col_weights[nc])
            col_set.discard(nc)
            col_tree.change_val(nc, float("inf"))
            deleted.append((False, nc))
        num += 1
        cur_ave = cur_score / (len(row_set) + len(col_set))
        if cur_ave > best_score:
            best_score = cur_ave
            best_num_deleted = num
    final_rows = set(range(m))
    final_cols = set(range(n))
    for is_row, idx in deleted[:best_num_deleted]:
        if is_row:
            final_rows.discard(idx)
        else:
            final_cols.discard(idx)
    return final_rows, final_cols


def run_fraudar(G, rng=None):
    """Port of fraudar.main.run_fraudar: log-weighted average degree on the
    reviewer x paper bid matrix; returns the flagged reviewer rows."""
    M = sparse.coo_matrix(G) > 0
    M = M.astype("int")
    col_sums = np.asarray(M.sum(axis=0)).ravel()
    col_weights = 1.0 / np.log(col_sums + 5)
    col_diag = sparse.lil_matrix((M.shape[1], M.shape[1]))
    col_diag.setdiag(col_weights)
    W = M @ col_diag
    rows, _ = fast_greedy_decreasing(W, col_weights)
    return list(rows), "logWeightedAveDegree"


# ── Detector dispatch ─────────────────────────────────────────────────────

def _mk_oqc_greedy():
    def fn(data, thresh, rng):
        S, _ = oqc_greedy(build_BA(data, thresh), rng)
        return S
    return fn


def _mk_oqc_local():
    def fn(data, thresh, rng):
        S, _ = oqc_local(build_BA(data, thresh), rng)
        return S
    return fn


def _mk_telltail():
    def fn(data, thresh, rng):
        S, _ = run_telltail(build_BA(data, thresh), rng)
        return S
    return fn


def _mk_densest():
    def fn(data, thresh, rng):
        S, _ = densest_subgraph(build_BA(data, thresh), rng)
        return S
    return fn


def _mk_fraudar():
    def fn(data, thresh, rng):
        S, _ = run_fraudar((data["S"] >= thresh).astype(int), rng)
        return S
    return fn


METHODS = {
    "oqc_greedy": _mk_oqc_greedy,
    "oqc_local": _mk_oqc_local,
    "telltail": _mk_telltail,
    "densest_subgraph": _mk_densest,
    "fraudar": _mk_fraudar,
}


# ── Evaluation ────────────────────────────────────────────────────────────

def _ba_density(data, thresh, S):
    if len(S) < 2:
        return 0.0
    G = build_BA(data, thresh)
    return _directed_edges(G, S) / (len(S) * (len(S) - 1))


def _set_metrics(S, data):
    C = set(data["colluders"])
    s = set(S)
    n = data["nrev"]
    n_c = len(C)
    overlap = len(s & C)
    recall = overlap / n_c if n_c else 0.0
    prec = overlap / len(s) if s else None
    jac = overlap / len(s | C) if (s or C) else 0.0
    best_ring = max((len(s & set(r)) / len(r) for r in data["rings"].values()),
                    default=0.0)
    # size-matched random baseline: a random set of size |s| covers |s|/n of
    # the colluders in expectation. >1 = enrichment over chance.
    chance = len(s) / n if n else 0.0
    return {
        "size": len(s),
        "overlap": overlap,
        "recall": recall,
        "precision": prec,
        "jaccard": jac,
        "best_ring_recall": best_ring,
        "recall_vs_chance": recall / chance if chance > 0 else None,
    }


def _avg_metrics(sets, data):
    mets = [_set_metrics(S, data) for S in sets]
    out = {}
    for k in mets[0]:
        vals = [m[k] for m in mets if m[k] is not None]
        if k == "size":
            out[k] = float(np.mean([m[k] for m in mets]))
        else:
            out[k] = float(np.mean(vals)) if vals else None
    return out


def run_detector(data, thresh, fn, num_trials, seed):
    rng = np.random.default_rng(seed)
    return [fn(data, thresh, rng) for _ in range(num_trials)]


def evaluate(data, base_data, thresh, fn, num_trials, seed):
    sets = run_detector(data, thresh, fn, num_trials, seed)
    out = _avg_metrics(sets, data)
    out["density"] = float(np.mean([_ba_density(data, thresh, S) for S in sets]))
    # per-trial sets (as sorted reviewer_id strings) and metrics, so an
    # independent reproduction can verify at SET level, not just aggregates.
    C = set(data["colluders"])
    out["sets"] = [sorted(data["reviewer_ids"][i] for i in S) for S in sets]
    out["size_trials"] = [float(len(set(S))) for S in sets]
    out["overlap_trials"] = [float(len(set(S) & C)) for S in sets]
    if base_data is not None:
        base_sets = run_detector(base_data, thresh, fn, num_trials, seed + 1)
        # Scenario files share the same 140 reviewers but in DIFFERENT row
        # orders, so data["colluders"] (collusion-scenario indices) is invalid
        # against base_sets (baseline-scenario indices). Map by reviewer_id.
        colluder_ids = [data["reviewer_ids"][i] for i in data["colluders"]]
        base_ridx = {r: i for i, r in enumerate(base_data["reviewer_ids"])}
        C_base = {base_ridx[r] for r in colluder_ids}
        bsizes = [len(set(S)) for S in base_sets]
        boverlaps = [len(set(S) & C_base) for S in base_sets]
        out["base_size"] = float(np.mean(bsizes))
        out["base_overlap_count"] = float(np.mean(boverlaps))
        out["base_overlap_frac"] = \
            float(np.mean([len(set(S) & C_base) / len(set(S)) if len(set(S)) else 0.0
                           for S in base_sets]))
        out["base_density"] = float(
            np.mean([_ba_density(base_data, thresh, S) for S in base_sets]))
        # negative = collusion flags MORE colluders than the all-honest run
        out["overlap_delta_vs_baseline"] = out["overlap"] - out["base_overlap_count"]
        out["base_sets"] = [sorted(base_data["reviewer_ids"][i] for i in S)
                            for S in base_sets]
        out["base_size_trials"] = [float(x) for x in bsizes]
        out["base_overlap_trials"] = [float(x) for x in boverlaps]
    return out


# ── CLI / orchestration ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=str, default="collusion-20,collusion-50")
    ap.add_argument("--base-dir", type=str, default="config/simulation")
    ap.add_argument("--papers", type=str, default="config/papers/papers_100.json")
    ap.add_argument("--bid-thresh", type=str, default="1,2",
                    help="comma list of BA edge thresholds on the 5-level bid "
                         "score (1 = H/VH, 2 = VH only)")
    ap.add_argument("--methods", type=str,
                    default="oqc_greedy,oqc_local,telltail,densest_subgraph,fraudar")
    ap.add_argument("--num-trials", type=int, default=10,
                    help="trials for stochastic detectors (oqc_local, telltail)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-scenario", type=str, default="baseline",
                    help="all-honest scenario for the paired comparison; "
                         "'none' disables it")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    methods = [m for m in (x.strip() for x in args.methods.split(",")) if m]
    threshs = [int(t) for t in args.bid_thresh.split(",") if t.strip()]
    report = {"num_trials": args.num_trials, "seed": args.seed,
              "bid_thresholds": threshs, "methods": methods,
              "provenance": _provenance(base_dir, args.papers),
              "results": {}}

    for s in [x.strip() for x in args.scenarios.split(",") if x.strip()]:
        data = load_matrices(base_dir, s, args.papers)
        if not data["colluders"]:
            print(f"[skip] {s}: no colluders (baseline scenario)")
            continue
        base_data = None
        if args.baseline_scenario.lower() != "none":
            bp = base_dir / args.baseline_scenario
            if (bp / "bids.json").exists():
                base_data = load_matrices(base_dir, args.baseline_scenario,
                                          args.papers)
            else:
                print(f"[warn] no baseline scenario at {bp}; "
                      f"skipping paired comparison")
        report["results"][s] = {}
        for thr in threshs:
            report["results"][s][f"thresh_{thr}"] = {}
            for name in methods:
                res = evaluate(data, base_data, thr, METHODS[name](),
                               args.num_trials, args.seed)
                report["results"][s][f"thresh_{thr}"][name] = res

    print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.json}")


def _fmt(v, fspec=".3f"):
    return format(v, fspec) if v is not None else "n/a"


def print_report(report: dict) -> None:
    W = 126
    print("=" * W)
    print("Collusion-ring detection baselines (Jecmen et al. 2024) on ColluBid")
    print("  detected SET vs true colluders C (|C| = n_colluders).")
    print("  |S^C| = #colluders in detected set; recall = |S^C|/|C|;")
    print("  prec = |S^C|/|S|; bestRing = max_r |S^ring_r|/|ring_r|;")
    print("  vChance = recall / (|S|/#rev)  (size-matched random coverage).")
    print("  Paired baseline: SAME detector on the all-honest scenario.")
    print("  |S|b, |S^C|b = size and #eventual-colluders of the honest-scenario")
    print("  detected set;  delta = |S^C| - |S^C|b  (negative = collusion adds")
    print("  nothing beyond what honest co-authorship already flags).")
    print("=" * W)
    for s, thr_dict in report["results"].items():
        for thr_key, dets in thr_dict.items():
            thr = thr_key.replace("thresh_", "")
            print(f"\n--- {s}  (BA edge threshold: bid score >= {thr}) ---")
            hdr = (f"{'method':<16}{'|S|':>7}{'|S^C|':>7}{'recall':>8}"
                   f"{'prec':>8}{'bestRing':>9}{'vChance':>8}{'density':>8}"
                   f"{'|S|b':>7}{'|S^C|b':>8}{'delta':>8}")
            print(hdr)
            print("-" * W)
            for name, r in dets.items():
                recall = _fmt(r.get("recall"))
                prec = _fmt(r.get("precision"))
                best = _fmt(r.get("best_ring_recall"))
                vsc = _fmt(r.get("recall_vs_chance"))
                dens = _fmt(r.get("density"))
                bsz = _fmt(r.get("base_size"))
                bo = _fmt(r.get("base_overlap_count"))
                dl = _fmt(r.get("overlap_delta_vs_baseline"), "+.2f")
                print(f"{name:<16}{r['size']:>7.1f}{r['overlap']:>7.1f}"
                      f"{recall:>8}{prec:>8}{best:>9}{vsc:>8}{dens:>8}"
                      f"{bsz:>7}{bo:>8}{dl:>8}")
    print("=" * W)


if __name__ == "__main__":
    main()
