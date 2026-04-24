###########################
# Standalone script to plot backdoor attacks.
# Plots ASR, mAP50 and number of malicious clients
###########################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

from utils.global_settings import ProjectPaths, ANSIColor

def parse_args():
    parser = argparse.ArgumentParser(description="Parser used for plotting FOD backdoor experiments")

    parser.add_argument("--sub-output-dir", type=str, default="", help="Denotes subdirectory in the 'output' directory where the training directory of the given model is found. If not given, the default ProjectPaths string is used.")
    parser.add_argument("--model-date", type=str, required=True, help="Datestamp used to identify the model whose training process is to be plotted.")
    parser.add_argument("--plot-title", type=str, default="Standard Plot", help="Title used for the plot.")

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

def plot(args):
    # Set theme for plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


    model_subdir = "" if not args.sub_output_dir else ".." + args.sub_output_dir
    csv_file_path = os.path.join(ProjectPaths.OUTPUT_DIR, model_subdir, args.model_date, "round_evaluations.csv")

    # Validate
    if not os.path.exists(csv_file_path):
        print(f"{ANSIColor.RED}[ERROR]{ANSIColor.RESET} Path: {csv_file_path}, does not exist! Terminating!")
        return

    df = pd.read_csv(csv_file_path)
    start_round, end_round = df["round"].min(), df["round"].max()

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_backdoor(df, ax, args.plot_title, start_round, end_round)

    fig.tight_layout()

    save_path = os.path.join(ProjectPaths.OUTPUT_DIR, model_subdir, args.model_date, "convergence_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"{ANSIColor.GREEN}[INFO]{ANSIColor.RESET} Plot successfully saved to: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    plot(args)
