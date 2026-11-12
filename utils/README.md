# Folder Purpose
Shared configuration and utility code used across the project.

## Structure
- `dataset.py`: Custom COCO dataset that injects poison at load time, supporting the
  disappearance, mislabelling and box-transform backdoors.
- `global_settings.py`: Every tunable constant in one place — the single source of truth that
  `parse_args.py` writes into.
- `parse_args.py`: All command-line arguments accepted by `custom_fl_train.py`, plus the
  validation and the structured run-configuration dump.
- `partitioning.py`: Splits the dataset across clients. Holds the IID path (seeded shuffle into
  equal chunks) and the attacker target-class quota rebalance.
- `noniid.py`: Non-IID partitioning by label skew at equal partition size — per-client Dirichlet
  class compositions, the image-to-class proxy, and the composition report.
- `select_clients.py`: Per-round client selection, including fixed schedules and fast-forwarding
  the selection RNG when a run resumes.
- `round_evaluator.py`: Per-round clean mAP and Attack Success Rate evaluation.
- `model_log.py`: Logger owning the run's output paths, metrics CSVs and checkpoint layout.
- `bbox_transform.py`: Bounding-box transforms shared by the regression-head backdoor and its
  evaluation, so training and scoring cannot disagree.
