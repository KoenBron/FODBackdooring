# Phase 0a — Foundation Cleanup

This batch is the first step toward the four-module orchestrator pipeline
(LaySelFL baseline + our per-group constrain-and-scale method). It only lays
foundation: no orchestrator is wired yet, no `--orchestrator` flag exists, no
behavior of today's runs changes.

## Goals of this batch

1. Remove the template constrain-and-scale (C&S) scaffolding that was added
   earlier but never reached real functionality. It would conflict with the
   per-group design coming next.
2. Add three additions to `wrapper.train_step` that future orchestrators will
   need: a freeze-groups stub, a per-epoch snapshot hook, and the fp32/autocast
   boundary fix for `loss_augment_fn`.
3. Introduce a per-tensor / per-group map for `fasterrcnn_resnet50_fpn` that
   downstream selectors, penalties, and amplifiers will read from.
4. Ship a smoke test that asserts coverage of the group map against the actual
   model.

Nothing else in this batch — no orchestrator, no localization, no penalty
strategy, no CLI flag. Those land in subsequent phases.

## Files deleted

| Path                       | Reason                                                                 |
|----------------------------|------------------------------------------------------------------------|
| `FL/cs_loss.py`            | Stub `cs_global_l2`. Never used outside an `--enable-constrain-and-scale` run that nothing reads anymore. |
| `FL/cs_localization.py`    | Stub `compute_bc_mask` (returned `None`). Same.                        |

## Files edited

### `utils/global_settings.py`
Removed `FLSettings.CS_ENABLED`, `CS_COMPUTE_BENIGN_ANCHOR`, `CS_PENALTY_LAMBDA`,
`CS_SCALE` and the surrounding doc comment.

### `utils/parse_args.py`
Removed four argparse calls (`--enable-constrain-and-scale`, `--cs-scale`,
`--cs-compute-benign-anchor`, `--cs-penalty-lambda`) and the four matching
`apply_settings` assignments. Passing any of those flags now fails argparse with
"unrecognized argument," which is the intended visible failure mode — old runs
that toggled these flags weren't doing anything useful anyway.

### `FL/custom_client.py`
- Removed the two CS imports.
- Removed the two CS-specific methods entirely (`_run_benign_anchor_pass`,
  `_do_constrain_and_scale_submission`).
- Removed the CS anchor pass branch in `client_side_train`.
- Removed the CS `loss_augment_fn` construction block.
- Removed the commented-out CS submission block.
- Removed `loss_augment_fn=loss_augment_fn` from the `train_step` call — there's
  no augmenter source left in the legacy path. Future orchestrators will pass
  their own augmenter directly.

Net effect on today's malicious-client runs (Neurotoxin and/or model
replacement): byte-identical. Same `train_step` call shape, same model
replacement, same gradient mask path.

### `models/fasterrcnn/wrapper.py`
Three additions to `train_step`:

1. **`freeze_groups=None` parameter.** Reserved for LaySelFL's CT (alternating
   sensitive-subset training where non-subset params are frozen with
   `requires_grad=False` so no optimizer state is allocated). Raises
   `NotImplementedError` if passed; lands in Phase 2 alongside the LaySelFL
   orchestrator.

2. **`epoch_callback=None` parameter.** Called as
   `epoch_callback(epoch_idx, state_dict_cpu)` at the end of every epoch with
   a CPU-side clone of `model.state_dict()`. Used by:
     - LaySelFL SLI's dynamic score (per-epoch L1 deltas)
     - Pilot A's retroactive selection at multiple (σ, k)

3. **Penalty-term fp32 boundary fix.** `loss_augment_fn(self.model)` was being
   evaluated inside `torch.amp.autocast("cuda")`. Small anchor distances
   (`||theta - anchor||^2` over a few SGD steps) round to zero in fp16 and the
   penalty silently disappears. The call is now outside the autocast block, and
   `losses` is cast to fp32 first so the sum stays in full precision. Model
   forward stays in autocast unchanged. With `loss_augment_fn=None` (today's
   default) the numeric path is identical to before — this is a fix for a
   latent bug that would surface the moment a real penalty is wired in.

### Reformatting / control-flow effects

`client_side_train` is materially shorter (~40 fewer lines) but its public
contract is unchanged: same arguments, same return shape, same behavior for
both benign and `--orchestrator none` paths (Neurotoxin + model replacement
still compose).

## Files added

### `FL/groups.py`
Per-tensor / per-group map for torchvision `fasterrcnn_resnet50_fpn`.

- **`GROUPS_REGEX`** — 13 groups, each defined by one or more regex patterns
  against named-parameter names:

  | Group           | Patterns                                              |
  |-----------------|-------------------------------------------------------|
  | `backbone.stem` | `backbone.body.conv1.*`, `backbone.body.bn1.*`        |
  | `backbone.layerN` (N=1..4) | `backbone.body.layerN.*`                  |
  | `fpn.PN` (N=2..5) | `backbone.fpn.{inner,layer}_blocks.i.*`, i=N-2     |
  | `rpn_head`      | `rpn.head.*`                                          |
  | `roi.box_head`  | `roi_heads.box_head.*`                                |
  | `roi.cls_score` | `roi_heads.box_predictor.cls_score.*`                 |
  | `roi.bbox_pred` | `roi_heads.box_predictor.bbox_pred.*`                 |

  FPN's `inner_blocks[i]` / `layer_blocks[i]` correspond to FPN levels P2..P5
  (largest to smallest stride). The P6 max-pool extension is parameter-free
  so not a group.

- **`build_group_map(model)`** — returns `(tensor_list, tensor_to_group,
  group_to_tensors)`. Raises `ValueError` listing any unmatched names.
  Orchestrators do per-tensor selection on `tensor_list`; group-level
  selection consumes `tensor_to_group`.

- **Distance utilities (additive forms):**
    - `per_tensor_l1(state_a, state_b, tensor_list)` → `{name: ||a-b||₁}`
    - `per_tensor_sq_l2(state_a, state_b, tensor_list)` → `{name: ||a-b||₂²}`
      (squared, so it sums cleanly across tensors and groups)
    - `group_sum(per_tensor, tensor_to_group)` → `{group: sum}` — works for
      both L1 and squared-L2 since both are additive.
    - `group_l2(per_tensor_sq, tensor_to_group)` → `{group: sqrt(group_sum)}`,
      the canonical per-group L2 wrapper.

  L1 is what LaySelFL's sensitivity score uses; squared-L2 is what our
  localization signal uses. Logging both per tensor and aggregating to groups
  on demand is the canonical path — there is no information benefit to logging
  only per-group sums.

- **state_dict caveat.** `named_parameters()` excludes BN running stats
  (`running_mean`, `running_var`, `num_batches_tracked`). Distance utilities
  here restrict to `tensor_list` and therefore implicitly drop those buffers.
  Computing distance over the raw state_dict directly would inflate the
  signal as BN buffers drift independently of gradient updates.

### `smoketest_groups.py` (project root)
Run with `python smoketest_groups.py`. Asserts:
1. every `named_parameter` lands in exactly one group;
2. no group is empty (catches dead regex patterns);
3. no name is multi-matched (regex order is for stability, not disambiguation);
4. per-tensor L1 and sq-L2 are zero for identical state dicts;
5. perturbing one tensor isolates the nonzero to exactly one group.

Prints a per-group summary (tensor count and parameter count) on pass.
**Run this on the GPU desktop** — the local laptop env has no torch.
No GPU is required; the model is built CPU-side with random weights.

## What still works (acceptance criteria)

- `python smoketest_regression_head.py` — unchanged (orthogonal Path-B attack).
- `python smoketest_box_transform_asr.py` — unchanged.
- A benign FL run with `--orchestrator none`-equivalent settings (no `--enable-*`
  attacker flags) is byte-identical to before.
- `--enable-model-replacement` + `--enable-neurotoxin` + `--attack-type
  TARGETED_*` runs follow the same code path as before, minus the CS hooks.

## What now fails (intended)

- Any `run_*.sh` script that passes `--enable-constrain-and-scale`, `--cs-scale`,
  `--cs-compute-benign-anchor`, or `--cs-penalty-lambda` will exit with an
  argparse "unrecognized argument" error. These flags toggled a template that
  never reached real functionality, so failure is the intended visible signal.
  `run_custom_fl_train.sh.example` doesn't reference them.

## Next phase (Phase 0b — not in this batch)

- Delete `client_side_train`'s legacy attacker branch into a private
  `_legacy_attacker_path` helper.
- Introduce `--orchestrator {none, pilot_a, laysel_fl, ours}` (defaulting to
  `none`); only `none` is valid until the orchestrator modules land.
- Stand up `FL/attack_state.py` for cross-attack-round state (carrying set,
  L̂, attack-round index).
- Add the `RoundOrchestrator` ABC and a dispatcher in `client_side_train`.

After Phase 0b, the codebase will have the *seams* for orchestrators but no
orchestrator yet. Phase 1 lands `PilotAOrchestrator` first; Phase 2 lands
`LaySelFLOrchestrator` alongside; Phase 3 lands `OursOrchestrator`.
