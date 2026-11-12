# Folder Purpose
Holds the Federated Learning specific code: server/client orchestration, the malicious-client
attack protocols, the aggregation strategies, and the passive detectability harness.

## Folder Structure
- `custom_server.py`: Server implementation. Broadcasts the global model, hands client updates
  to the selected aggregation strategy, runs per-round evaluation and checkpointing, and drives
  the observational stealth harness. Strategies are given updates with the attacker/benign
  label stripped, so no defence can read ground truth.
- `custom_client.py`: Client manager. Owns the dataset partitions and the model wrapper, and
  dispatches each selected client to its attack protocol (or plain benign training).
- `attack_state.py`: Cross-round malicious-client state (carrying sets, plastic-phase
  accumulators, per-client benign references), persisted so a run can resume mid-attack.
- `stealth_defenses.py`: Passive detectability harness. Computes what each defence *would*
  have flagged, without altering aggregation.
- `normclip_report.py`: Observational norm-clip exceedance report. Split so the threshold
  logic stays torch-free.
- `groups.py`: Single source of truth for the per-tensor / per-group parameter map.
- `strategies/`:
    - `fedavg.py`: Federated averaging (sample-weighted mean).
    - `fedopt.py`: Server-side learning-rate blend, `w_new = (1-eta)*w_global + eta*avg(w_client)`; `eta=1` reduces to FedAvg.
    - `multikrum.py`: Multi-Krum robust aggregation (Blanchard et al. 2017) — keeps the `m` most-central updates by Krum score, then FedAvgs them.
    - `foolsgold.py`: FoolsGold (Fung et al. 2020) — maintains per-client historical update directions and down-weights clients whose histories agree.
    - `normclip.py`: Norm clipping — rescales any update whose L2 norm exceeds a multiple of the round median. Magnitude only; direction is preserved.
    - `_robust.py`: Torch-free shared core for the robust aggregators (Krum scoring/selection, norm-clip thresholds), also used by the observational stealth harness so passive and active defences cannot drift.
- `attack_protocols/`: The malicious-client strategies, including `partial.py`
  (partial-amplification with per-tensor carrying-layer selection) and the `pilot_*`
  localization pilots.
- `scoring/`: Per-tensor divergence scoring used to select carrying layers.
