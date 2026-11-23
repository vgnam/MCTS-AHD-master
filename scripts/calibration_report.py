"""Summarize GENESIS uncertainty calibration and eval-noise robustness runs.

Usage:
    python scripts/calibration_report.py RUN_DIR [RUN_DIR ...] [--min-obs 3] [--json out.json]

Each RUN_DIR is a hydra output directory of an ``algorithm=ab-mcts-ahd`` run. It must
contain ``calibration.json`` (written by GENESIS); ``best_code_overall_val_stdout.txt``
is used, when present, to report validation averages per run.

For every posterior that an evaluation updates (``node``: the expanded heuristic's
subtree, ``gen``: the operator that generated it, ``global``: the shared hyper-prior)
the one-step-ahead NIG predictive is scored on the observed objective:
    coverage@q  fraction of observations inside the central q predictive interval
    PIT         predictive CDF at the observation (uniform when calibrated)
    KS          Kolmogorov-Smirnov distance of the PITs from Uniform(0, 1)
    NLL         mean negative log predictive density
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

LEVELS = (0.5, 0.8, 0.95)
SOURCES = ("node", "gen", "global")


def load_records(run_dir):
    path = Path(run_dir) / "calibration.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(records, source, min_obs=0):
    count_key = "gen_observations" if source == "gen" else "parent_observations"
    rows = [r for r in records if source in r and (source == "global" or r.get(count_key, 0) >= min_obs)]
    if not rows:
        return None
    pit = np.array([r[source]["pit"] for r in rows])
    nll = np.array([r[source]["nll"] for r in rows])
    out = {"n": len(rows)}
    for q in LEVELS:
        out[f"coverage@{q:g}"] = float(np.mean(np.abs(pit - 0.5) <= q / 2))
    out["ks"] = float(stats.kstest(pit, "uniform").statistic)
    out["nll_mean"] = float(np.mean(nll))
    out["nll_median"] = float(np.median(nll))
    out["pit_hist"] = np.histogram(pit, bins=10, range=(0, 1))[0].tolist()
    return out


def validation(run_dir):
    path = Path(run_dir) / "best_code_overall_val_stdout.txt"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"Average for (\d+): ([0-9.eE+-]+)", text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--min-obs", type=int, default=3,
                        help="also report records whose posterior had at least this many observations")
    parser.add_argument("--json", help="write the full summary to this file")
    args = parser.parse_args()

    records = []
    per_run = {}
    for run in args.runs:
        recs = load_records(run)
        records += recs
        per_run[run] = {"records": len(recs), "validation": validation(run)}

    summary = {"runs": per_run, "all": {}, f"min_obs_{args.min_obs}": {}}
    for source in SOURCES:
        summary["all"][source] = summarize(records, source)
        summary[f"min_obs_{args.min_obs}"][source] = summarize(records, source, args.min_obs)

    header = f"{'subset':<10} {'source':<7} {'n':>5} " + " ".join(f"cov@{q:g}".rjust(8) for q in LEVELS) + f" {'KS':>6} {'NLL':>8}"
    print(header)
    for subset in ("all", f"min_obs_{args.min_obs}"):
        for source in SOURCES:
            s = summary[subset][source]
            if s is None:
                continue
            cov = " ".join(f"{s[f'coverage@{q:g}']:8.3f}" for q in LEVELS)
            print(f"{subset:<10} {source:<7} {s['n']:>5} {cov} {s['ks']:6.3f} {s['nll_median']:8.3f}")
    print("nominal              " + " ".join(f"{q:8.3f}" for q in LEVELS))
    for run, info in per_run.items():
        if info["validation"]:
            print(run, info["validation"])
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
