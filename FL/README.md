# FL/

The federated learning simulation: server/client orchestration, the aggregation
strategies, the malicious-client attack protocols, and the passive detectability harness.

## Top level

| File | Purpose |
|---|---|
| `custom_server.py` | The server. Broadcasts the global model, hands client updates to the aggregation strategy (with attacker/benign labels stripped), runs per-round evaluation and checkpointing, and drives the detectability harness. |
| `custom_client.py` | The client manager. Owns the per-client dataset partitions and the model wrapper, builds the data partition (IID or Dirichlet), and dispatches each selected client to its attack protocol or to benign training. |
| `attack_state.py` | Cross-round malicious-client state (carrying sets, plastic-phase accumulators, per-client benign references), persisted to `attack_logs/attack_state.json` so a run resumes mid-attack. |
| `stealth_defenses.py` | The passive detectability harness. Computes, per round, what each defence *would* have flagged (Krum scores, cosine-to-median, norm ratios, clustering) without altering aggregation. |
| `normclip_report.py` | Observational norm-clip exceedance report: per malicious update, whether its L2 norm exceeds a sweep of thresholds. Does not alter aggregation. |
| `groups.py` | Single source of truth for the per-tensor / per-group parameter map of the Faster R-CNN model, used by the localizers, penalties and the harness. |
| `strategy.py` | `BaseStrategy` — the abstract aggregation-strategy interface (`reset_state`, `aggregate`). |

## strategies/

Server-side aggregation. Each implements `BaseStrategy`; the server selects one via
`--aggregation-strategy`.

| File | Strategy |
|---|---|
| `fedavg.py` | Federated averaging — sample-weighted mean of the client updates. |
| `fedopt.py` | Server-side learning-rate blend, `w_new = (1-η)·w_global + η·avg(w_client)`; `η=1` reduces to FedAvg. |
| `multikrum.py` | Multi-Krum (Blanchard et al. 2017) — keep the `m` most-central updates by Krum score, then FedAvg them. |
| `foolsgold.py` | FoolsGold (Fung et al. 2020) — maintain per-client historical update directions and down-weight clients whose histories agree. |
| `normclip.py` | Norm clipping — rescale any update whose L2 norm exceeds a multiple of the round median; direction preserved. |
| `_robust.py` | Torch-free shared core (Krum scoring/selection, norm-clip thresholds) used by both the active strategies above and the passive harness, so the two cannot drift. |

## attack_protocols/

Per-round malicious-client procedures. The client manager selects one via
`--attack-protocol`.

| File | Purpose |
|---|---|
| `base.py` | `AttackProtocol` — the abstract contract every protocol implements. |
| `partial.py` | The partial-amplification protocol: per-tensor carrying-layer selection, the plastic/exploit phase machinery, and selective scaling of the carrying set. |
| `amplify.py` | Torch-free numeric core for partial amplification (carrying-set scaling, norm-budget allocation, non-carrying masking). |
| `pilot_a.py` | Observational per-group localization pilot. |
| `pilot_b.py` | Observational per-tensor localization pilot. |

## scoring/

Divergence scoring used to rank and select carrying layers.

| File | Purpose |
|---|---|
| `l2_based.py` | Per-tensor / per-group L2-divergence scoring and the cutoff / top-k selectors. |
| `layselfl.py` | Per-tensor L1 averaging helpers. |
