# Federated Object Detection Backdooring

This repository implements targeted **disappearance**, **mislabelling** and **box-transform**
backdoor attacks against CNN-based object detection models (Faster R-CNN) trained in a
Federated Learning session, together with a set of robust-aggregation defences and a
passive detectability harness used to measure how visible each attack is to them.

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
3. Place a COCO-annotated dataset in a `dataset/` directory at the project root. The
   layout below is expected (configurable in `utils/dataset.py`; images must be `.jpg`):
    ```
    dataset/training_annotations.json
    dataset/validation_annotations.json
    dataset/training_images/
    dataset/validation_images/
    ```
4. Run a federated training session. `custom_fl_train.py` is the single entry point;
   copy the example launcher and edit its argument arrays:
    ```bash
    cp run_custom_fl_train.sh.example run_fl_train.sh
    chmod +x run_fl_train.sh
    ./run_fl_train.sh
    ```

## Repository Structure

### Entry points
* `custom_fl_train.py` — the federated learning simulation. All experiments run through this.
* `run_custom_fl_train.sh.example` — template launcher showing how the arguments are grouped.
* `evaluation.py` — clean-utility (mAP) and backdoor-efficacy (ASR) evaluation. Also imported
  by `utils/model_log.py`, so it is part of the training path, not just a standalone tool.

### Modules
* `FL/` — the federated simulation: server/client orchestration, attack protocols, aggregation
  strategies and the passive detectability harness.
* `utils/` — dataset loading and poison injection, data partitioning (IID and non-IID),
  argument parsing, logging and per-round evaluation.
* `models/` — Faster R-CNN implementation behind a common wrapper interface.
* `scripts/` — standalone analysis tools: result plotting, stealth summary tables and a
  dry-run of the partitioner. Not imported by the training code.
* `notebooks/` — exploratory analysis notebooks (outputs stripped to keep the clone small).

### Aggregation strategies
Selected with `--aggregation-strategy`:

| value | description |
|---|---|
| `fedavg` | sample-weighted mean (default) |
| `fedopt` | server-side learning-rate blend, `(1-η)·w_global + η·avg(w_client)` |
| `multikrum` | Multi-Krum (Blanchard et al. 2017) — keep the `m` most central updates, then FedAvg |
| `foolsgold` | FoolsGold (Fung et al. 2020) — down-weight clients whose historical update directions agree |
| `normclip` | norm clipping — rescale any update whose L2 norm exceeds `c ×` the round median |

Defences only ever see submitted updates, never the attacker/benign labels.

### Attack protocols
Selected with `--attack-protocol`: `composable` (the knob-driven Neurotoxin / model-replacement
path), `partial` (partial-amplification with per-tensor carrying-layer selection),
`pilot_a` / `pilot_b` (observational localization pilots).

### Data partitioning
`--partition-mode iid` (default) splits a seeded shuffle into equal chunks.
`--partition-mode dirichlet` gives every client the same *number* of images but skews each
client's *class composition* via `--dirichlet-alpha`, so label heterogeneity varies while
partition size is held constant. `scripts/partition_report.py` dry-runs the partitioner and
prints the resulting per-client compositions without training.

### Output directories (not tracked)
* `dataset/` — input data.
* `outputs/` — checkpoints (`.pth`), per-round metrics CSVs and run configuration.

### Environment
* `requirements.txt` — Python dependencies.

## Usage Examples

### 1. Basic Federated Learning (benign)
```bash
python custom_fl_train.py \
  --num-rounds 50 \
  --total-clients 10 \
  --clients-per-round 5 \
  --attacker-fraction 0.0
```
50 rounds, 5 of 10 clients sampled per round, no malicious clients.

### 2. Model-replacement backdoor
A targeted disappearance attack on `person` objects, amplified by model replacement.
```bash
python custom_fl_train.py \
  --num-rounds 100 \
  --total-clients 20 \
  --clients-per-round 4 \
  --attacker-fraction 0.1 \
  --benign-client-epochs 2 \
  --malicious-client-epochs 5 \
  --enable-model-replacement \
  --attack-type TARGETED_DISAPPEARANCE \
  --target-class "person" \
  --poison-rate 0.15 \
  --trigger-pattern FLOATING_UNIFORM \
  --floating-pos CENTER \
  --trigger-color "255,0,0"
```
* **Attackers**: 10% of clients, training 5 local epochs against the benign 2, poisoning 15%
  of the `person` objects in their partition.
* **Trigger**: a red square scaled to and floating inside each target bounding box.

### 3. Fixed client schedule
```bash
python custom_fl_train.py \
  --num-rounds 20 \
  --total-clients 10 \
  --clients-per-round 2 \
  --attacker-fraction 0.2 \
  --use-client-schedule \
  --client-schedule "10:0,10:1,15:0"
```
Forces clients 0 and 1 into round 10 and client 0 into round 15; remaining slots are filled
by randomly sampled benign clients.

### 4. Non-IID partitioning against a robust aggregator
```bash
python custom_fl_train.py \
  --num-rounds 60 \
  --total-clients 20 \
  --clients-per-round 9 \
  --attacker-fraction 0.2 \
  --partition-mode dirichlet \
  --dirichlet-alpha 0.5 \
  --aggregation-strategy multikrum \
  --krum-f 1 \
  --do-stealth-eval
```
Label-skewed partitions, Multi-Krum aggregation, and the passive detectability harness
enabled so the per-client stealth metrics are written alongside the usual round metrics.
