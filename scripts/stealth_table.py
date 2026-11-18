"""Read stealth_summary.csv (and round_evaluations.csv) into a tidy table.

stealth_rounds.txt is the human view; this is the machine view of the same numbers.
Nothing is computed here that the harness did not already record -- the point is to
stop every analysis re-deriving the same aggregation.

    # one row per run, aggregated over exploit rounds
    python scripts/stealth_table.py outputs/noniid_a05_cut11_long/*/

    # one row per round for a single run
    python scripts/stealth_table.py --per-round outputs/iid_a05_cut11_long/*/

    # tidy CSV to stdout, pipe it anywhere
    python scripts/stealth_table.py --csv outputs/*/*/ > stealth.csv

Exploit rounds default to "attacker-present rounds after the first --plastic (6)",
matching how the partial protocol splits its phases. --plastic 0 keeps every
attacker round. Rounds with no attacker are always excluded from the aggregate but
are kept by --per-round, so a benign-only run still prints its band.
"""
import argparse
import csv
import glob
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUMMARY = "stealth_summary.csv"
EVALS = "round_evaluations.csv"

# The headline fields, in reading order: magnitude, then direction, then who fired.
# Every one of these is a column the harness writes; none is recomputed here.
HEADLINE = [
    ("benign_l2_mean", "ben_l2"),
    ("benign_l2_sd", "ben_sd"),
    ("attacker_l2_mean", "att_l2"),
    ("l2_ratio", "ratio"),
    ("l2_z", "z_l2"),
    ("benign_cos_to_median_mean", "ben_cosM"),
    ("attacker_cos_to_median_mean", "att_cosM"),
    ("cos_to_median_z", "z_cosM"),
    ("benign_cos_to_benign_mean", "ben_cosB"),
    ("attacker_cos_to_benign_mean", "att_cosB"),
    ("cos_to_benign_z", "z_cosB"),
    ("dist_to_benign_z", "z_dstB"),
    ("multikrum_tpr", "mk_tpr"),
    ("normclip_tpr", "nc_tpr"),
    ("cluster_tpr", "cl_tpr"),
]


def _f(v):
    """CSV cell -> float. Empty and 'nan' both mean not measured."""
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return st.mean(vals) if vals else float("nan")


LEGACY_WARNED = set()


def load_summary(run_dir):
    """All stealth_summary.csv rows for a run, values coerced to float where possible.

    Runs recorded before the benign-band columns were added have only AUC and
    Cohen's d, so every band and z field reads NaN. That is a property of the run,
    not a parse failure, so it is announced once per run on stderr rather than
    silently producing a table of NaN.
    """
    path = os.path.join(run_dir, SUMMARY)
    if not os.path.isfile(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        row = {k: _f(v) for k, v in r.items()}
        row["round"] = int(row["round"])
        out.append(row)
    if out and "benign_l2_mean" not in out[0] and run_dir not in LEGACY_WARNED:
        LEGACY_WARNED.add(run_dir)
        print(f"note: {run_dir} predates the benign-band columns; band and z fields "
              f"will be NaN (ASR and round counts are still valid)", file=sys.stderr)
    return out


def load_asr(run_dir):
    """round -> ASR, skipping rounds the run did not evaluate (--eval-every)."""
    path = os.path.join(run_dir, EVALS)
    if not os.path.isfile(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        a = _f(r.get("ASR"))
        if not math.isnan(a):
            out[int(r["round"])] = a
    return out


def exploit_rows(rows, plastic=6):
    """Attacker-present rounds after the plastic phase. Empty for a benign-only run."""
    att = [r for r in rows if r.get("n_attackers", 0) >= 1]
    return att[plastic:] if plastic else att


def aggregate(run_dir, plastic=6):
    """One record per run: identity, ASR headline, and the mean of each stealth field."""
    rows = load_summary(run_dir)
    if not rows:
        return None
    asr = load_asr(run_dir)
    ex = exploit_rows(rows, plastic)
    scored = ex if ex else rows          # benign-only run: describe the band itself
    rec = {"run": run_dir.rstrip("/"),
           "rounds": len(rows),
           "attacker_rounds": sum(1 for r in rows if r.get("n_attackers", 0) >= 1),
           "scored_rounds": len(scored),
           "peak_asr": max(asr.values()) if asr else float("nan"),
           "final_asr": asr[max(asr)] if asr else float("nan")}
    for key, short in HEADLINE:
        rec[short] = _mean([r.get(key, float("nan")) for r in scored])
    return rec


def per_round(run_dir):
    """One record per round, ASR joined on where the round was evaluated."""
    asr = load_asr(run_dir)
    out = []
    for r in load_summary(run_dir):
        rec = {"run": os.path.basename(run_dir.rstrip("/")), "round": r["round"],
               "n_clients": int(r.get("n_clients", 0)),
               "n_attackers": int(r.get("n_attackers", 0)),
               "asr": asr.get(r["round"], float("nan"))}
        for key, short in HEADLINE:
            rec[short] = r.get(key, float("nan"))
        out.append(rec)
    return out


def _print_table(records, cols):
    if not records:
        print("no stealth_summary.csv found in the given directories")
        return
    widths = {c: max(len(c), *(len(_cell(r.get(c))) for r in records)) for c in cols}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    for r in records:
        print("  ".join(_cell(r.get(c)).rjust(widths[c]) for c in cols))


def _cell(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if math.isnan(v) else (f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run directories (globs are fine)")
    ap.add_argument("--per-round", action="store_true",
                    help="one row per round instead of one per run")
    ap.add_argument("--csv", action="store_true", help="emit CSV on stdout")
    ap.add_argument("--plastic", type=int, default=6,
                    help="attacker rounds to drop as the plastic phase (default 6; 0 keeps all)")
    args = ap.parse_args()

    dirs = []
    for pattern in args.run_dirs:
        dirs.extend(sorted(glob.glob(pattern)) or [pattern])
    dirs = [d for d in dirs if os.path.isdir(d)]

    if args.per_round:
        records = [rec for d in dirs for rec in per_round(d)]
        cols = ["run", "round", "n_clients", "n_attackers", "asr"] + [s for _, s in HEADLINE]
    else:
        records = [r for r in (aggregate(d, args.plastic) for d in dirs) if r]
        cols = (["run", "rounds", "attacker_rounds", "scored_rounds", "peak_asr", "final_asr"]
                + [s for _, s in HEADLINE])

    if args.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    else:
        _print_table(records, cols)


if __name__ == "__main__":
    main()
