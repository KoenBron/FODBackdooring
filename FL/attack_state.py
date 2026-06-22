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


class _AttackState:
    """Singleton-by-convention — do not instantiate. Import the module-level
    AttackState binding below; use reset() to wipe between in-process tests.
    """

    def __init__(self):
        # ---- Persisted across resumes ----

        # client_id -> set of param-name or group-name strings (protocol-defined).
        # LaySelFL stores per-tensor names; ours stores per-group names.
        self.carrying_set: dict[int, set[str]] = {}

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

    def reset(self) -> None:
        """Wipe every field. Used between in-process simulations and by tests."""
        self.carrying_set.clear()
        self.attack_round_idx.clear()
        self.lhat.clear()
        self.benign_anchor.clear()
        self.beta_max.clear()

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
