"""Regression checks against the released fixed-triplet detector reports."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]   # ColluBid/
BASE_DIR = ROOT / "config" / "simulation"
PAPERS = ROOT / "config" / "papers" / "papers_100.json"
CANONICAL = ROOT / "baseline_detection" / "results" / "detect_ring_report.json"
PORT_PY = ROOT / "baseline_detection" / "detect_ring_baselines.py"


# every metric evaluate() reports (excluding per-trial lists / sets)
_METRIC_FIELDS = ["size", "overlap", "recall", "precision", "jaccard",
                  "best_ring_recall", "recall_vs_chance", "density",
                  "base_size", "base_overlap_count", "base_overlap_frac",
                  "base_density", "overlap_delta_vs_baseline"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_port(canonical):
    """Re-run the port exactly as its CLI would (same code path as main())."""
    port = _load("detect_ring_baselines", PORT_PY)
    num_trials, seed = canonical["num_trials"], canonical["seed"]
    threshs, methods = canonical["bid_thresholds"], canonical["methods"]
    scenarios = list(canonical["results"].keys())
    report = {"num_trials": num_trials, "seed": seed,
              "bid_thresholds": threshs, "methods": methods,
              "provenance": port._provenance(BASE_DIR, str(PAPERS)),
              "results": {}}
    for s in scenarios:
        data = port.load_matrices(BASE_DIR, s, str(PAPERS))
        base = port.load_matrices(BASE_DIR, "baseline", str(PAPERS))
        report["results"][s] = {}
        for thr in threshs:
            report["results"][s][f"thresh_{thr}"] = {}
            for name in methods:
                report["results"][s][f"thresh_{thr}"][name] = port.evaluate(
                    data, base, thr, port.METHODS[name](), num_trials, seed)
    return report


def test_golden_json_regression():
    canonical = json.load(open(CANONICAL, encoding="utf-8"))
    fresh = _run_port(canonical)
    for sc, thr_dict in canonical["results"].items():
        assert sc in fresh["results"], f"scenario {sc} missing"
        for thr_key, dets in thr_dict.items():
            for name, ref in dets.items():
                got = fresh["results"][sc][thr_key][name]
                for f in _METRIC_FIELDS:
                    if ref[f] is None:
                        assert got[f] is None, f"{sc} {thr_key} {name}.{f}: expected None"
                        continue
                    assert abs(got[f] - ref[f]) <= 1e-9, \
                        f"{sc} {thr_key} {name}.{f}: port {got[f]} != golden {ref[f]}"
                assert got["sets"] == ref["sets"], \
                    f"{sc} {thr_key} {name}: detected sets differ from golden"
                assert got["base_sets"] == ref["base_sets"], \
                    f"{sc} {thr_key} {name}: baseline sets differ from golden"


def test_stochastic_determinism():
    """Same-seed re-runs of stochastic detectors must be bit-identical."""
    port = _load("detect_ring_baselines", PORT_PY)
    data = port.load_matrices(BASE_DIR, "collusion-20", str(PAPERS))
    base = port.load_matrices(BASE_DIR, "baseline", str(PAPERS))
    for name in ("oqc_local", "telltail"):
        a = port.evaluate(data, base, 1, port.METHODS[name](), 30, 0)
        b = port.evaluate(data, base, 1, port.METHODS[name](), 30, 0)
        assert a["sets"] == b["sets"], f"{name}: sets not reproducible at seed 0"
        assert a["size"] == b["size"], f"{name}: size not reproducible at seed 0"
