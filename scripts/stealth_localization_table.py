###########################
# Per-parameter-group stealth decomposition for the detectability pilots.
#
# Reads a run's stealth_per_client.csv (the passive harness output) plus a
# checkpoint (for per-group parameter counts) and, per parameter group, reports:
#   * energy share for benign clients      (% of ||Delta||^2 in this group)
#   * energy share for malicious clients
#   * rms/param per cohort                  (||Delta_g|| / sqrt(n_params_g))
#   * benign-to-malicious ratio of rms/param
#   * whole-update distance-to-median per cohort (per-group not derivable from CSV)
#
# Two claims this is built to support:
#   1. regression-head attacks are NOT more subtle  -> compare the per-group
#      profiles of a regression run vs a classification run.
#   2. the backdoor is highly localized             -> the ben/mal ratio (and the
#      malicious energy share) spikes in one head group while the param-heavy
#      backbone barely moves per parameter.
###########################

import argparse
import os

import numpy as np
import pandas as pd
import torch

from FL.groups import GROUP_NAMES, name_to_group

# CSV column -> friendly group name. Mirrors the harness's 13-group + "other"
# decomposition (see FL.stealth_defenses._STEALTH_GROUPS). Order = display
# order, which matches FL.groups.GROUP_NAMES (backbone front to ROI back).
GROUP_COLS = {f"dl2_{g}": g for g in list(GROUP_NAMES) + ["other"]}

CHECKPOINT_NAMES = ("latest_fl_checkpoint.pth", "best_model.pth")
PER_CLIENT_CSV = "stealth_per_client.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-parameter-group stealth decomposition table for detectability pilots."
    )
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL:DIR",
        help="A run as 'label:path/to/run_dir' (repeatable). The dir must contain "
             f"{PER_CLIENT_CSV}; a checkpoint is read once for parameter counts.",
    )
    parser.add_argument("--out", default=None, help="Optional path to write the Markdown tables.")
    parser.add_argument("--csv-out-dir", default=None, help="Optional dir to also drop a per-run CSV table.")
    return parser.parse_args()


def find_checkpoint(run_dir):
    for name in CHECKPOINT_NAMES:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    return None


def param_counts_per_group(checkpoint_path):
    """Float-parameter count per group, keyed by friendly group name. Matches the
    harness's compute_delta (integer buffers excluded). Names that aren't in
    any parameter group (BN running stats etc.) fall into 'other'."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    counts = {g: 0 for g in list(GROUP_NAMES) + ["other"]}
    for k, v in sd.items():
        if torch.is_floating_point(v):
            counts[name_to_group(k) or "other"] += v.numel()
    return counts


def _cohort_energy_share(sub, cols):
    """Mean over clients of per-client energy share dl2_g^2 / sum_g dl2_g^2 (percent)."""
    total = (sub[cols] ** 2).sum(axis=1)
    return {c: ((sub[c] ** 2) / total).mean() * 100.0 for c in cols}


def per_group_table(csv_path, param_counts):
    df = pd.read_csv(csv_path)
    cols = [c for c in GROUP_COLS if c in df.columns]
    ben, mal = df[df.is_attacker == 0], df[df.is_attacker == 1]

    es_ben, es_mal = _cohort_energy_share(ben, cols), _cohort_energy_share(mal, cols)

    rows = []
    for c in cols:
        g = GROUP_COLS[c]
        n = param_counts.get(g, 0)
        rms_ben = (ben[c] / np.sqrt(n)).mean() if n else float("nan")
        rms_mal = (mal[c] / np.sqrt(n)).mean() if n else float("nan")
        # n_g cancels, so this is also the cohort magnitude ratio of this group.
        ratio = rms_ben / rms_mal if rms_mal else float("nan")
        rows.append({
            "group": g, "params": n,
            "energy%_ben": es_ben[c], "energy%_mal": es_mal[c],
            "rms/param_ben": rms_ben, "rms/param_mal": rms_mal,
            "ben/mal": ratio,
        })
    return pd.DataFrame(rows)


def cohort_dist_to_median(csv_path):
    """Whole-update distance-to-median per cohort. Per-group is NOT derivable from
    the CSV (it needs the raw coordinate-wise median over client deltas), so this is
    a single global figure per cohort."""
    df = pd.read_csv(csv_path)
    if "dist_to_median" not in df.columns:
        return None
    return (df[df.is_attacker == 0]["dist_to_median"].mean(),
            df[df.is_attacker == 1]["dist_to_median"].mean())


def render_markdown(table):
    cols = ["group", "params", "energy%_ben", "energy%_mal", "rms/param_ben", "rms/param_mal", "ben/mal"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['params']):,} | {r['energy%_ben']:.1f} | {r['energy%_mal']:.1f} "
            f"| {r['rms/param_ben']:.2e} | {r['rms/param_mal']:.2e} | {r['ben/mal']:.2f} |"
        )
    return "\n".join(lines)


def parse_run_spec(spec):
    label, sep, path = spec.partition(":")
    if not sep:
        raise ValueError(f"--run must be 'label:path', got: {spec}")
    return label.strip(), path.strip()


def main():
    args = parse_args()
    runs = [parse_run_spec(s) for s in args.run]

    # Architecture is identical across runs, so resolve param counts once.
    param_counts = None
    for _, path in runs:
        ck = find_checkpoint(path)
        if ck:
            param_counts = param_counts_per_group(ck)
            break
    if param_counts is None:
        raise SystemExit("[ERROR] No checkpoint found in any run dir; cannot compute rms/param.")

    blocks = []
    for label, path in runs:
        csv_path = os.path.join(path, PER_CLIENT_CSV)
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        table = per_group_table(csv_path, param_counts)
        block = [f"### {label}", "", render_markdown(table)]
        dm = cohort_dist_to_median(csv_path)
        if dm:
            ratio = dm[1] / dm[0] if dm[0] else float("nan")
            block.append(
                f"\n_dist-to-median (whole update): benign {dm[0]:.4f}, malicious {dm[1]:.4f} "
                f"(mal/ben {ratio:.2f}). Per-group not available from the CSV._"
            )
        blocks.append("\n".join(block))
        if args.csv_out_dir:
            os.makedirs(args.csv_out_dir, exist_ok=True)
            out_csv = os.path.join(args.csv_out_dir, f"localization_{label.replace(' ', '_')}.csv")
            table.to_csv(out_csv, index=False)
            print(f"[INFO] wrote {out_csv}")

    text = "\n\n".join(blocks)
    print("\n" + text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[INFO] Markdown written to {args.out}")


if __name__ == "__main__":
    main()
