# scripts/

Standalone analysis tools. None are imported by the training code — they read a finished
run's output files and produce tables, plots or dry-run reports. Run all from the project
root.

| File | Purpose |
|---|---|
| `partition_report.py` | Dry-runs the partitioner from annotations alone (no training): prints per-client class compositions and skew diagnostics for a range of `--dirichlet-alpha`. Use to pick α before running. |
| `stealth_table.py` | Reads `stealth_summary.csv` (+ `round_evaluations.csv`) across runs into one tidy table, aggregated over exploit rounds. |
| `stealth_localization_table.py` | Per-parameter-group stealth decomposition: reads a run's `stealth_per_client.csv` and a checkpoint and reports per-group energy shares for benign vs malicious clients. |
| `plot_backdoor.py` | Reads a run's `round_evaluations.csv` and writes a convergence plot (clean mAP@50, ASR, #malicious clients per round). |
| `server.py` | A small web server that serves generated plot PNGs as a browsable gallery. |
