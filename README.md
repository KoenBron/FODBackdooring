# Federated Object Detection Backdooring

A simulation framework for studying **targeted backdoor attacks on CNN-based object
detectors (Faster R-CNN) trained under Federated Learning**, together with a set of robust
aggregation defences and a passive detectability harness that measures how visible each
attack is to those defences.

It accompanies the MSc thesis *Partial Amplification: Selective parameter scaling for
federated object detection backdoor attacks under intermittent adversarial participation*,
available from the TU Delft repository: <http://repository.tudelft.nl/>.

The framework lets you:

- run a federated training session with any mix of benign and malicious clients;
- inject disappearance, mislabelling or box-transform backdoors behind several trigger types;
- aggregate with FedAvg, FedOpt, Multi-Krum, FoolsGold or norm clipping;
- partition client data IID or non-IID (Dirichlet label skew at fixed partition size);
- log, each round, both utility/attack metrics and a passive read of what each defence
  *would* have flagged.

---

## Setup

Developed on **Python 3.12** (3.10+ should work).

1. Install PyTorch (we used torch 2.10.0 / CUDA 13.0 on Linux):
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```
2. Install the remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place a COCO-annotated dataset in a `dataset/` directory at the project root. The layout
   below is expected (configurable in `utils/dataset.py`; images must be `.jpg`):
   ```
   dataset/training_annotations.json
   dataset/validation_annotations.json
   dataset/training_images/
   dataset/validation_images/
   ```

---

## Quick start

`custom_fl_train.py` is the single entry point. The simplest way to drive it is the
provided launcher template, which groups the arguments into readable arrays:

```bash
cp run_custom_fl_train.sh.example run_fl_train.sh
chmod +x run_fl_train.sh
./run_fl_train.sh
```

Or call it directly for a minimal benign run:

```bash
python custom_fl_train.py \
  --num-rounds 50 \
  --total-clients 10 \
  --clients-per-round 5 \
  --attacker-fraction 0.0
```

Each run creates a timestamped directory `outputs/<output-subdir>/<date>/` containing the
checkpoints, metrics CSVs and configuration dump described under
[Output files](#output-files).

---

## Configuration reference

Arguments are grouped by concern. Pass any of them to `custom_fl_train.py`; run
`python custom_fl_train.py --help` for the full list with defaults. The groups:

### Federated learning
Controls the FL loop and aggregation.

| Argument | Meaning |
|---|---|
| `--num-rounds` | number of training rounds |
| `--total-clients` | size of the client pool |
| `--clients-per-round` | clients sampled each round |
| `--attacker-fraction` | fraction of the pool that is malicious (lowest client ids) |
| `--aggregation-strategy` | `fedavg` \| `fedopt` \| `multikrum` \| `foolsgold` \| `normclip` |
| `--server-lr` | server learning rate η (FedOpt only) |
| `--gamma-cap` | upper bound on the model-replacement scale γ |
| `--seed` / `--partitioning-seed` | RNG seeds for client selection / data partitioning |
| `--resume-from` | resume a run from its output directory name (see [Resuming](#resuming-and-determinism)) |
| `--use-client-schedule` / `--client-schedule` | force specific clients into specific rounds, `"round:client,..."` |
| `--eval-every` / `--eval-start-round` / `--eval-on-injection-rounds` | when per-round evaluation runs |
| `--skip-round-eval` / `--skip-map-eval` / `--skip-asr-eval` | skip evaluation (whole / clean mAP / poison ASR) to cut compute |
| `--output-subdir` | subdirectory under `outputs/` for this run |

### Training hyperparameters
`--training-batch-size`, `--training-num-workers`, and separate epoch/learning-rate settings
for benign and malicious clients (`--benign-client-epochs`, `--malicious-client-epochs`,
`--benign-client-lr`, `--malicious-client-lr`).

### Attacker configuration
Selects the malicious-client behaviour. `--attack-protocol` picks the routing:

| Protocol | Behaviour |
|---|---|
| `composable` | knob-driven path: model replacement / Neurotoxin / the `--attack-type` backdoor composed via flags |
| `partial` | partial-amplification: per-tensor carrying-layer selection with selective scaling |
| `pilot_a` / `pilot_b` | observational localization pilots (per-group / per-tensor) |

Model-replacement and Neurotoxin toggles (`--enable-model-replacement`,
`--enable-neurotoxin`, `--neurotoxin-mask-ratio`) apply to the `composable` path. The
`--partial-*` family (select mode, cutoff/topk, gamma mode, norm budget) configures the
`partial` protocol. A target-class quota for attacker partitions is available via
`--guarantee-target-objects` and the `--attacker-target-*` floors.

### Partial-protocol phases
`--partial-plastic-*` governs the observation ("plastic") phase that freezes the carrying
set; `--partial-constrain`, `--partial-freeze-noncarrying` and `--partial-mask-noncarrying`
govern how non-carrying parameters are handled during the exploit phase.

### Data partitioning
| Argument | Meaning |
|---|---|
| `--partition-mode` | `iid` (seeded shuffle, equal chunks) or `dirichlet` (label skew at fixed size) |
| `--dirichlet-alpha` | Dirichlet concentration; small = strongly skewed, large → IID |
| `--dirichlet-label-rule` | image→class proxy: `max_area` \| `count_majority` \| `first` \| `stochastic_majority` |
| `--dirichlet-attacker-mode` | `independent` or `matched` (attackers share a label composition) |

### Backdoor properties
`--attack-type` (`GLOBAL_DISAPPEARANCE`, `TARGETED_MISLABELING`, `TARGETED_DISAPPEARANCE`,
`TARGETED_BOX_TRANSFORM`), `--target-class`, `--mislabel-class`, `--poison-rate`, and the
trigger geometry: `--trigger-pattern` (`UNIFORM`, `FLOATING_UNIFORM`, `CHECKERBOARD`,
`FULL_CHECKERBOARD`, `FULL_NOISE`, `FLOATING_CHECKERBOARD`), `--trigger-color`,
`--floating-pos` (`CENTER`, `TOP_LEFT`), `--w-ratio`, `--h-ratio`, `--trigger-alpha`.

The box-transform (regression-head) backdoor adds `--box-transform-mode`
(`shift` \| `shrink` \| `shift_shrink`) with the matching `--box-shift-*` / `--box-shrink-factor`.

### Detectability harness (passive)
`--do-stealth-eval` enables per-client stealth logging without altering aggregation.
Per-defence parameters: `--krum-f`, `--multikrum-m` (Multi-Krum), `--foolsgold-kappa`,
`--foolsgold-feature-group` (FoolsGold), `--normclip-c`, `--normclip-agg-c` (norm clipping).
`--stealth-disable` turns off individual metric blocks; `--do-normclip-report` writes a
separate exceedance report (on by default).

---

## Output files

A run writes everything under `outputs/<output-subdir>/<date>/`:

| File / dir | Contents |
|---|---|
| `config.txt` | the full resolved configuration for the run (plus the backdoor banner and raw arguments) |
| `round_evaluations.csv` | one row per round: `mAP_50`, `mAP_50_95`, `ASR`, `fail_rate`, `avg_conf_fail` |
| `partition_composition.csv` | one row per client: partition size, per-class image counts, and composition drift |
| `stealth_per_client.csv` | one row per (round, client): the passive defence measurements (`update_l2`, `cos_to_median`, `dist_to_median`, Krum score, …) |
| `stealth_summary.csv` | per-round aggregates of the stealth metrics |
| `stealth_rounds.txt` | the same, human-readable |
| `normclip_report.csv` | observational norm-clip exceedance sweep per malicious update |
| `foolsgold_weights.csv` | per-client FoolsGold aggregation weights (FoolsGold runs only) |
| `normclip_aggregation.csv` | per-client norm-clip rescaling factors (norm-clip aggregation runs only) |
| `latest_fl_checkpoint.pth` | the round's aggregated model + round number (used by `--resume-from`) |
| `best_model.pth` | the highest-mAP checkpoint so far |
| `attack_logs/` | `attack_state.json` (resumable attacker state) and per-protocol logs (`partial/`, `pilot_a/`, `pilot_b/`) |
| tensorboard events | scalar logs written alongside the CSVs |

The CSVs are the primary artifacts; the analysis scripts under `scripts/` read them into
tables and plots (see [`scripts/README.md`](scripts/README.md)).

---

## Running the program

All runs go through `custom_fl_train.py`. Below are common configurations; combine the
argument groups above as needed.

### Benign federated training
```bash
python custom_fl_train.py \
  --num-rounds 50 --total-clients 10 --clients-per-round 5 \
  --attacker-fraction 0.0
```

### Model-replacement disappearance attack
```bash
python custom_fl_train.py \
  --num-rounds 100 --total-clients 20 --clients-per-round 4 \
  --attacker-fraction 0.1 \
  --benign-client-epochs 2 --malicious-client-epochs 5 \
  --enable-model-replacement \
  --attack-type TARGETED_DISAPPEARANCE --target-class "person" \
  --poison-rate 0.15 \
  --trigger-pattern FLOATING_UNIFORM --floating-pos CENTER --trigger-color "255,0,0"
```

### Partial-amplification mislabelling attack
```bash
python custom_fl_train.py \
  --num-rounds 60 --total-clients 20 --clients-per-round 9 \
  --attacker-fraction 0.2 \
  --attack-protocol partial \
  --partial-select-mode cutoff --partial-cutoff 1.5 \
  --partial-gamma-mode replacement \
  --partial-plastic-rounds 4 \
  --attack-type TARGETED_MISLABELING --target-class "person" --mislabel-class "boat" \
  --poison-rate 0.5 --trigger-pattern FLOATING_UNIFORM --trigger-color "255,0,255"
```

### Non-IID data against a robust aggregator
```bash
python custom_fl_train.py \
  --num-rounds 60 --total-clients 20 --clients-per-round 9 \
  --attacker-fraction 0.2 \
  --partition-mode dirichlet --dirichlet-alpha 0.5 \
  --aggregation-strategy multikrum --krum-f 1 \
  --do-stealth-eval
```

### Fixed client schedule (intermittent participation)
```bash
python custom_fl_train.py \
  --num-rounds 20 --total-clients 10 --clients-per-round 2 \
  --attacker-fraction 0.2 \
  --use-client-schedule --client-schedule "10:0,10:1,15:0"
```
Forces clients 0 and 1 into round 10 and client 0 into round 15; remaining slots are filled
by randomly sampled benign clients.

---

## Resuming and determinism

A run is fully derived from `--seed` and `--partitioning-seed`, so the data partition and
client selection reproduce exactly. Pass `--resume-from <date-directory-name>` to continue
an interrupted run: the server reloads `latest_fl_checkpoint.pth` and the attacker state
from `attack_logs/attack_state.json`, and training continues from the next round. The
checkpoint is written last each round, so an interrupted round simply re-runs on resume.

---

## Repository structure

| Path | Contains |
|---|---|
| `custom_fl_train.py` | the entry point: builds the server and client manager and runs the FL loop |
| `run_custom_fl_train.sh.example` | launcher template grouping the arguments |
| [`FL/`](FL/README.md) | the federated simulation — server/client orchestration, aggregation strategies, attack protocols, and the passive detectability harness |
| [`utils/`](utils/README.md) | dataset loading and poisoning, data partitioning, argument parsing, logging, and per-round evaluation |
| [`models/`](models/README.md) | the Faster R-CNN model behind a common wrapper interface, including the box-transform backdoor surgery |
| [`scripts/`](scripts/README.md) | standalone analysis tools that read a run's CSVs into tables and plots (not part of the training loop) |
| [`notebooks/`](notebooks/README.md) | exploratory analysis notebooks |
| `dataset/`, `outputs/` | input data and run results (not tracked in git) |
