###########################
# Standalone script to plot backdoor attacks.
# Plots ASR, mAP50 and number of malicious clients
###########################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
import os

from utils.global_settings import ProjectPaths, ANSIColor

# Filenames the script reads/writes. Single source of truth so the single-run and
# batch (--plot-missing) paths agree on what "already has a plot" means.
EVAL_CSV_NAME = "round_evaluations.csv"
PLOT_BASENAME = "convergence_plot"
PLOT_EXT = ".png"

DEFAULT_PLOT_TITLE = "Standard Plot"

def parse_args():
    parser = argparse.ArgumentParser(description="Parser used for plotting FOD backdoor experiments")

    parser.add_argument("--sub-output-dir", type=str, default="", help="Denotes subdirectory in the 'output' directory where the training directory of the given model is found. If not given, the default ProjectPaths string is used. Ignored when --plot-missing is set.")
    parser.add_argument("--model-date", type=str, default=None, help="Datestamp used to identify the model whose training process is to be plotted. Required unless --plot-missing is set.")
    parser.add_argument("--plot-title", type=str, default=DEFAULT_PLOT_TITLE, help="Title used for the plot. In --plot-missing mode this is only used if explicitly overridden; otherwise each run is titled with its own directory name.")
    parser.add_argument("--plot-missing", action="store_true", help=f"Scan the whole outputs tree and generate a plot for every run directory that has a {EVAL_CSV_NAME} but no {PLOT_BASENAME}{PLOT_EXT} yet. Ignores --model-date / --sub-output-dir.")

    return parser.parse_args()

def plot_backdoor(df, ax, title, start_round, end_round):
    # 1. Bar chart for the number of attackers (in the background)
    ax_bar = ax.twinx()
    
    if 'num_attackers' in df.columns:
        # Make the bars semi-transparent gray
        ax_bar.bar(df['round'], df['num_attackers'], color='gray', alpha=0.2, label='# Attackers', width=0.6)
    
    # Configure the axis for attackers (max 5 in this setup)
    ax_bar.set_ylim(0, 5) 
    ax_bar.set_yticks(range(6))
    ax_bar.set_ylabel('Number of Malicious Clients', color='gray', fontsize=10)
    ax_bar.tick_params(axis='y', labelcolor='gray')
    ax_bar.grid(False) # Turn off gridlines for the bar chart to prevent visual clutter
    
    # Lock the X-axis bounds to the start and end rounds
    ax.set_xlim(start_round - 0.5, end_round + 0.5) 
    
    # Force the X-axis to only show integer rounds (prevents 17.5, 18.5 etc.)
    step_size = max(1, (int(end_round) - int(start_round)) // 15)
    ax.set_xticks(range(int(start_round), int(end_round) + 1, step_size))
    
    # 2. Line for mAP (Left Axis)
    line1, = ax.plot(df['round'], df['mAP_50'], color='tab:blue', marker='o', markersize=4, linewidth=2, label='mAP@50 (Clean)')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('mAP@50', color='tab:blue', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xlabel('Communication Round', fontweight='bold')

    # 3. Line for ASR (Right Axis)
    ax_asr = ax.twinx()
    line2, = ax_asr.plot(df['round'], df['ASR'], color='tab:red', marker='x', markersize=4, linewidth=2, linestyle='--', label='ASR (Poison)')
    ax_asr.set_ylim(0, 1.0)
    ax_asr.set_ylabel('Attack Success Rate', color='tab:red', fontweight='bold')
    ax_asr.tick_params(axis='y', labelcolor='tab:red')
    ax_asr.grid(False)

    # 4. Combine legends
    # Extract handles from the bar chart only if it was drawn
    bars, labels_bars = ax_bar.get_legend_handles_labels() if 'num_attackers' in df.columns else ([], [])
    lines = [line1, line2]
    labels_lines = [line.get_label() for line in lines]
    
    ax.legend(bars + lines, labels_bars + labels_lines, loc='lower right', framealpha=0.9)
    ax.set_title(title, fontweight='bold', pad=15)

    # Prevent the two right Y-axes from overlapping by shifting the bar axis
    ax_bar.spines["right"].set_position(("axes", 1.15))

def get_unique_save_path(dir, base_name="convergence_plot", ext=".png"):
    candidate = os.path.join(dir, f"{base_name}{ext}")
    if not os.path.exists(candidate):
        return candidate

    i = 1
    while True:
        candidate = os.path.join(dir, f"{base_name}_{i:02d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1

def set_plot_theme():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def resolve_single_run_dir(sub_output_dir, model_date):
    """Locate the run directory for a single --model-date plot. Preserves the
    original ".." + sub_output_dir join semantics."""
    model_subdir = "" if not sub_output_dir else ".." + sub_output_dir
    return os.path.join(ProjectPaths.OUTPUT_DIR, model_subdir, model_date)

def run_dir_has_plot(run_dir):
    """True if run_dir already contains a convergence plot (any suffix variant)."""
    return bool(glob.glob(os.path.join(run_dir, f"{PLOT_BASENAME}*{PLOT_EXT}")))

def find_runs_missing_plots(search_root):
    """Walk search_root and return run directories that hold an evaluation CSV but
    no convergence plot yet. Sorted for stable, predictable output."""
    missing = []
    for current_dir, _subdirs, files in os.walk(search_root):
        if EVAL_CSV_NAME in files and not run_dir_has_plot(current_dir):
            missing.append(current_dir)
    return sorted(missing)

def render_plot_for_run(run_dir, title):
    """Read the evaluation CSV in run_dir, build the convergence figure and save it.
    Returns the save path, or None when the run can't be plotted. Closes the figure
    so batch runs don't accumulate open figures."""
    csv_file_path = os.path.join(run_dir, EVAL_CSV_NAME)
    if not os.path.exists(csv_file_path):
        print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Path: {csv_file_path}, does not exist! Skipping.")
        return None

    df = pd.read_csv(csv_file_path)
    if df.empty:
        print(f"{ANSIColor.YELLOW}[WARNING]{ANSIColor.RESET} {csv_file_path} has no rows. Skipping.")
        return None

    start_round, end_round = df["round"].min(), df["round"].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_backdoor(df, ax, title, start_round, end_round)
    fig.tight_layout()

    save_path = get_unique_save_path(dir=run_dir)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"{ANSIColor.GREEN}[INFO]{ANSIColor.RESET} Plot successfully saved to: {save_path}")
    return save_path

def plot_single(args):
    """Plot one run identified by --model-date (+ optional --sub-output-dir)."""
    set_plot_theme()
    run_dir = resolve_single_run_dir(args.sub_output_dir, args.model_date)
    render_plot_for_run(run_dir, args.plot_title)

def plot_all_missing(args):
    """Plot every run under the outputs tree that is still missing a plot."""
    set_plot_theme()
    search_root = ProjectPaths.OUTPUT_DIR.parent  # outputs/ root: covers every sub-output dir
    runs = find_runs_missing_plots(search_root)

    if not runs:
        print(f"{ANSIColor.GREEN}[INFO]{ANSIColor.RESET} No run directories missing a plot under {search_root}.")
        return

    print(f"{ANSIColor.GREEN}[INFO]{ANSIColor.RESET} Found {len(runs)} run(s) missing a plot.")
    for run_dir in runs:
        # Use the explicit title only if the user overrode the default; otherwise
        # title each run with its own folder name so batch plots stay distinguishable.
        title = args.plot_title if args.plot_title != DEFAULT_PLOT_TITLE else os.path.basename(os.path.normpath(run_dir))
        render_plot_for_run(run_dir, title)

if __name__ == "__main__":
    args = parse_args()
    if args.plot_missing:
        plot_all_missing(args)
    elif args.model_date:
        plot_single(args)
    else:
        print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Provide --model-date for a single plot, or --plot-missing to plot every run lacking one.")
