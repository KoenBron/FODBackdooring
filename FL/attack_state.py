"""Cross-round malicious-client state.

Lives as a module-level singleton (FL.attack_state.AttackState) so any
AttackProtocol can read and update it without explicit plumbing through
ClientManager. Keyed by client_id so distinct attackers maintain independent
carrying sets, attack-round counters, etc.

Persistence boundary (Phase 0b)
-------------------------------
Only `carrying_set` is written to disk. The other slots (attack_round_idx,
lhat, benign_anchor, beta_max) are ephemeral and reset on each
`custom_fl_train.py` invocation. The 4-hour cluster shift writes the carrying
set after each round and the next shift reads it back on resume, so malicious
clients pick up their previous round's localization without recomputing.

When LaySelFL lands (Phase 2), `attack_round_idx` joins the persisted set
because L_p = L_{r mod P} subset selection requires the counter to survive
across shifts. The persistence widens via to_dict() / from_dict() — adding a
field is one line in each.

Why a singleton, not state-on-ClientManager
-------------------------------------------
ClientManager already owns per-client model state (subsets, wrapper, Neurotoxin
round caches). AttackState describes the *attack protocol's* notion of what
each attacker has done across rounds — a separate concern. Keeping it in its
own module keeps the protocol implementations from coupling to ClientManager's
internals (and vice-versa), and matches the "stored in a global class/variable"
direction.
"""
from pathlib import Path
from typing import Any
import json
import statistics


class _AttackState:
    """Singleton-by-convention — do not instantiate. Import the module-level
    AttackState binding below; use reset() to wipe between in-process tests.
    """

    def __init__(self):
        # ---- Persisted across resumes ----

        # client_id -> set of param-name or group-name strings (protocol-defined).
        # LaySelFL stores per-tensor names; ours stores per-group names.
        self.carrying_set: dict[int, set[str]] = {}

        # ---- Plastic phase (Partial protocol) — POOLED across attackers ----
        # The plastic phase observes the per-round localizer for the first N pooled
        # injection rounds, then FREEZES one shared carrying set. Pooling is fair
        # because the client partitions are IID (exchangeable). All fields are
        # persisted so a 4h cluster shift landing mid-plastic resumes correctly.
        self.plastic_div_sum: dict[str, float] = {}  # name -> Σ div_t over plastic rounds
        self.plastic_b_sum: dict[str, float] = {}    # name -> Σ b_t   over plastic rounds
        self.plastic_count: int = 0                  # pooled injection rounds accumulated
        self.plastic_frozen: bool = False            # has the freeze happened?
        self.plastic_frozen_set: set[str] = set()    # the single shared post-freeze carrying set

        # ---- Benign-norm baseline (norm-clip report) — POOLED across attackers ----
        # ||theta_ben1 - theta_t|| collected over the plastic injection rounds (when the
        # benign anchor is still trained); at the plastic freeze their MEDIAN becomes the
        # single reference the exploit-phase norm-clip report uses (the anchors are skipped
        # there). Both persist so the frozen reference survives a resume.
        self.plastic_benign_norms: list = []               # per-round ||theta_ben1 - theta_t||
        self.plastic_benign_norm_ref: float | None = None  # frozen median (None until freeze)

        # ---- Norm-clip report context (ephemeral, NOT persisted) ----
        # Per-cid {benign_ref, phase} the protocol stashes each round for the server-side
        # reporter to read (the server can't see the per-round ben1 norm otherwise).
        self.normclip_ctx: dict[int, dict] = {}

        # ---- In-memory only (reset on every fresh process) ----

        # Counter that increments each time THIS attacker actually attacks.
        # NOT a calendar-round counter — non-attacking rounds don't advance it.
        # Drives LaySelFL's L_p = L_{r mod P} subset selection.
        self.attack_round_idx: dict[int, int] = {}

        # Graded sensitivity scores from the last attack round (for continuous
        # penalty/amplification variants in our method).
        self.lhat: dict[int, dict[str, float]] = {}

        # Cached benign-anchor tensors (CPU dict, name -> Tensor) for EMA-style
        # anchors that survive within a process. Excluded from persistence
        # because tensors are bulky and cheap to recompute on demand.
        self.benign_anchor: dict[int, dict[str, Any]] = {}

        # Last calibrated penalty coefficients (value-ratio and gradient-ratio
        # variants are kept side-by-side; downstream picks one).
        self.beta_max: dict[int, dict[str, float]] = {}

    # ---- Accessors (preferred over direct dict access) ----

    def get_carrying_set(self, client_id: int) -> set[str]:
        """Most recent carrying set for this attacker, or empty set if none.
        Returns a fresh set so callers can't accidentally mutate the stored one.
        Does NOT auto-create an entry — unlike a defaultdict — so iterating
        carrying_set.items() afterwards stays clean.
        """
        return set(self.carrying_set.get(client_id, ()))

    def set_carrying_set(self, client_id: int, names) -> None:
        """Record this attacker's new carrying set. Accepts any iterable of
        strings; stored as a set so Jaccard-style set algebra is direct."""
        self.carrying_set[client_id] = set(names)

    def bump_attack_round(self, client_id: int) -> int:
        """Increment this attacker's attack-round counter; return the new value.
        Call ONCE per round in which this attacker actually attacks — i.e. at
        the top of the protocol's run() method."""
        new = self.attack_round_idx.get(client_id, 0) + 1
        self.attack_round_idx[client_id] = new
        return new

    def get_attack_round(self, client_id: int) -> int:
        """How many times has this attacker attacked so far. 0 if never."""
        return self.attack_round_idx.get(client_id, 0)

    # ---- Plastic-phase accumulator (pooled across attackers) ----

    def accumulate_plastic(self, div, b) -> None:
        """Add one round's per-bucket div_t / b_t into the pooled running sums and
        bump the pooled injection count. div and b are name->float dicts over the
        same (eligible) key set, so the sums stay aligned across rounds."""
        for n, v in div.items():
            self.plastic_div_sum[n] = self.plastic_div_sum.get(n, 0.0) + v
        for n, v in b.items():
            self.plastic_b_sum[n] = self.plastic_b_sum.get(n, 0.0) + v
        self.plastic_count += 1

    def plastic_means(self):
        """Pooled means (mean_div, mean_b) = sum / count. Empty dicts if no samples."""
        c = self.plastic_count
        if c == 0:
            return {}, {}
        return (
            {n: s / c for n, s in self.plastic_div_sum.items()},
            {n: s / c for n, s in self.plastic_b_sum.items()},
        )

    def is_plastic_frozen(self) -> bool:
        return self.plastic_frozen

    def set_plastic_frozen(self, frozen_set) -> None:
        """Commit the single shared frozen carrying set and close the plastic phase."""
        self.plastic_frozen_set = set(frozen_set)
        self.plastic_frozen = True

    # ---- Benign-norm baseline (norm-clip report) ----

    def accumulate_benign_norm(self, norm) -> None:
        """Collect one plastic round's benign-anchor update norm ||theta_ben1 - theta_t||."""
        self.plastic_benign_norms.append(float(norm))

    def freeze_benign_norm(self) -> None:
        """Freeze the MEDIAN of the collected benign norms as the exploit-phase reference
        (None if none were collected — e.g. the report was off during the plastic phase)."""
        self.plastic_benign_norm_ref = (
            statistics.median(self.plastic_benign_norms) if self.plastic_benign_norms else None
        )

    def set_normclip_ctx(self, cid: int, benign_ref, phase: str) -> None:
        """Stash this round's reporter context for an attacker (read by the server)."""
        self.normclip_ctx[cid] = {"benign_ref": benign_ref, "phase": phase}

    def get_normclip_ctx(self, cid: int):
        """The stashed reporter context for an attacker this round, or None."""
        return self.normclip_ctx.get(cid)

    def reset(self) -> None:
        """Wipe every field. Used between in-process simulations and by tests."""
        self.carrying_set.clear()
        self.attack_round_idx.clear()
        self.lhat.clear()
        self.benign_anchor.clear()
        self.beta_max.clear()
        self.plastic_div_sum.clear()
        self.plastic_b_sum.clear()
        self.plastic_count = 0
        self.plastic_frozen = False
        self.plastic_frozen_set.clear()
        self.plastic_benign_norms.clear()
        self.plastic_benign_norm_ref = None
        self.normclip_ctx.clear()

    # ---- Persistence ----

    def to_dict(self) -> dict:
        """JSON-serializable snapshot of persisted fields.

        Only fields that need to survive a process restart are included.
        Tensors (benign_anchor) and ephemeral counters (attack_round_idx for
        Phase 0b) are intentionally excluded — they're cheap to rebuild and
        including them couples the persistence format to in-flight protocol
        state. As more protocols come online, widen this dict.
        """
        return {
            "version": 1,
            "carrying_set": {
                # JSON keys must be strings; sort the set members so diffs are stable.
                str(cid): sorted(s) for cid, s in self.carrying_set.items()
            },
            # Pooled plastic accumulator — persisted so a shift mid-plastic resumes
            # accumulation (rather than freezing on a truncated window).
            "plastic": {
                "div_sum": self.plastic_div_sum,
                "b_sum": self.plastic_b_sum,
                "count": self.plastic_count,
                "frozen": self.plastic_frozen,
                "frozen_set": sorted(self.plastic_frozen_set),
                # Benign-norm baseline for the norm-clip report (persisted so the frozen
                # reference survives a resume; ephemeral normclip_ctx is NOT persisted).
                "benign_norms": list(self.plastic_benign_norms),
                "benign_norm_ref": self.plastic_benign_norm_ref,
            },
        }

    def from_dict(self, d: dict) -> None:
        """Mutate self from a to_dict() output. Missing keys are treated as
        'no recorded state' — a partial load (e.g. a future version that
        added fields) is not an error here, the loader just gets what it can.
        """
        self.carrying_set = {
            int(cid): set(names)
            for cid, names in d.get("carrying_set", {}).items()
        }
        p = d.get("plastic", {})
        self.plastic_div_sum = dict(p.get("div_sum", {}))
        self.plastic_b_sum = dict(p.get("b_sum", {}))
        self.plastic_count = int(p.get("count", 0))
        self.plastic_frozen = bool(p.get("frozen", False))
        self.plastic_frozen_set = set(p.get("frozen_set", ()))
        self.plastic_benign_norms = [float(v) for v in p.get("benign_norms", [])]
        _ref = p.get("benign_norm_ref", None)
        self.plastic_benign_norm_ref = None if _ref is None else float(_ref)

    def save(self, path) -> None:
        """Write the snapshot to disk atomically.

        Writes to a sibling .tmp and renames into place so a kill between the
        write and the rename leaves the previous snapshot intact — important
        on the 4-hour cluster shift boundary where shutdown can be abrupt.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        tmp.replace(path)

    def load(self, path) -> bool:
        """Populate self from a previously-saved file.

        Returns True on a successful load, False if the file doesn't exist
        (fresh-start case — not an error). Malformed JSON is allowed to raise
        — we'd rather fail loudly than silently lose state on resume.
        """
        path = Path(path)
        if not path.exists():
            return False
        with open(path) as f:
            self.from_dict(json.load(f))
        n = len(self.carrying_set)
        print(f"[AttackState] resumed carrying sets for {n} attacker(s) from {path}")
        return True


# Module-level singleton. Do not reassign — callers import THIS binding.
AttackState = _AttackState()
