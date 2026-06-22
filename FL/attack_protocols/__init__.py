"""Per-round malicious-client procedures.

Each AttackProtocol is a preset recipe for one malicious training round —
unlike `_composable_attacker_path` on ClientManager, which is built by composing
independent knobs (--enable-neurotoxin, --enable-model-replacement, --attack-type
TARGETED_*). Protocols here run multiple trajectories, compose the four
strategies (Localization, Penalty, Amplification, Clipping), and read/write
cross-round state via FL.attack_state.AttackState.

Concrete protocols land in later phases:
  - Phase 1: PilotAProtocol (observational, no penalty, no submission deformation)
  - Phase 2: LaySelFLProtocol (SLI + CT + TC, faithful baseline of Li et al. 2026)
  - Phase 3: OursProtocol (per-group benign-anchored constrain-and-scale)
"""
