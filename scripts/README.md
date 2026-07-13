# scripts/

Loose command-line scripts. The two below form the plotting workflow:
**`plot_backdoor.py`** generates the plot images, **`server.py`** serves them as a
browsable web gallery.

Run both from the **project root** (not from inside `scripts/`).

---

## 1. `plot_backdoor.py` — generate the plots

Reads a run's `round_evaluations.csv` and writes a `convergence_plot.png`
(clean mAP@50, ASR, and #malicious-clients per round) into that run's directory.

```bash
# Plot every run under outputs/ that has a CSV but no plot yet (most common)
python scripts/plot_backdoor.py --plot-missing

# Plot one specific run in the default output dir (outputs/prelim_exp/<date>)
python scripts/plot_backdoor.py --model-date 2026-06-02-20-42-16

# Plot one run in a different tree, with a custom title
python scripts/plot_backdoor.py --model-date 2026-05-18-23-52-53 \
    --sub-output-dir /fl_retention_exp/cluster --plot-title "Retention run"
```

| Flag | Meaning |
|------|---------|
| `--plot-missing` | Scan the whole `outputs/` tree and plot every run missing one. Ignores the flags below. |
| `--model-date`   | Folder name of a single run to plot. Resolved under the default output dir (`outputs/prelim_exp`) unless `--sub-output-dir` redirects it. |
| `--sub-output-dir` | Output subdirectory the run lives in, relative to `outputs/` (e.g. `/fl_retention_exp/cluster`). Single-run mode only. |
| `--plot-title`   | Plot title (batch mode titles each run by its folder name). |

A run is "already plotted" if it contains a `convergence_plot*.png`, so
`--plot-missing` is safe to re-run — it only fills in the gaps.

---

## 2. `server.py` — view the plots in a browser

Serves all `*.png` under `outputs/` as an HTML gallery. The direct
subdirectories of `outputs/` (e.g. `fl_retention_exp`, `other_exp`) are
**collections** you switch between in the nav bar; inside a collection, plots are
grouped by run into collapsible sections (newest first, collapsed by default).

```bash
python scripts/server.py                       # lands on outputs/fl_retention_exp
python scripts/server.py --subdir other_exp    # land on a different collection
python scripts/server.py --port 8080           # custom port
```

Then open `http://localhost:8000` (or `http://<tailscale-hostname>:8000` from
another machine — it binds `0.0.0.0` by default).

| Flag | Default | Meaning |
|------|---------|---------|
| `--root`   | `outputs` | Base directory scanned; its subdirectories become collections. |
| `--subdir` | `fl_retention_exp` | Collection to show first (switch live via the nav bar). |
| `--port`   | `8000` | Port to listen on. |
| `--host`   | `0.0.0.0` | Interface to bind. |

The page reads from disk on every load, so after generating new plots with
`plot_backdoor.py` just refresh the browser (or tick **Auto-refresh**).

---

**Typical flow:** run experiments → `python scripts/plot_backdoor.py --plot-missing`
→ `python scripts/server.py` → browse.
