# utils/

Shared configuration and utility code used across the project.

| File | Purpose |
|---|---|
| `dataset.py` | Custom COCO dataset that injects poison at load time — the disappearance, mislabelling and box-transform backdoors, and the trigger patterns. |
| `global_settings.py` | Every tunable constant in one place: the single source of truth that `parse_args.py` writes into, plus the attack-type / trigger-type enums. |
| `parse_args.py` | All command-line arguments for `custom_fl_train.py`, their validation, and the config-to-text rendering written into `config.txt`. |
| `partitioning.py` | The IID partition: a seeded shuffle split into equal chunks, with the optional attacker target-class quota rebalance. |
| `noniid.py` | The non-IID partition: per-client Dirichlet class compositions at fixed partition size, the image→class proxy, and the composition report. |
| `select_clients.py` | Per-round client selection — random sampling, fixed schedules, and fast-forwarding the selection RNG when a run resumes. |
| `round_evaluator.py` | Per-round clean mAP and Attack Success Rate evaluation on the validation set (used by the server each round). |
| `model_log.py` | The logger: owns the run's output directory, writes the metrics CSVs and `config.txt`, and manages checkpoints. |
| `bbox_transform.py` | Bounding-box transforms shared by the box-transform backdoor and its evaluation, so training and scoring cannot disagree. |
