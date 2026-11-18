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
        # One entry per (round, client) injection, keyed "<round>:<cid>" so a round
        # REPLAYED after a crash-and-resume OVERWRITES its earlier contribution instead
        # of adding a second one. The sums and the count are derived from this dict
        # (see the properties below), so identity is enforced in one place.
        #
        # Why this matters: the freeze picks carriers from mean(div)/mean(b) pooled over
        # these entries. An append-only accumulator let a replayed plastic round reweight
        # that mean, which is enough to move a tensor sitting near the cutoff across it.
        # Each entry is {"div": {name: float}, "b": {name: float}, "n": int}; "n" is 1 for
        # a real injection and >1 only for the aggregate carried in from a pre-keying
        # checkpoint (see from_dict).
        self.plastic_samples: dict[str, dict] = {}
        self.plastic_frozen: bool = False            # has the freeze happened?
        self.plastic_frozen_set: set[str] = set()    # the single shared post-freeze carrying set

        # ---- Benign-norm baseline — PER CLIENT, never pooled ----
        # Both benign anchors' update norms (||theta_ben1 - theta_t|| and
        # ||theta_ben2 - theta_t||) collected over the plastic injection rounds, keyed by
        # the attacker that produced them. At the plastic freeze each client's MEAN becomes
        # ITS OWN exploit-phase reference (the anchors are skipped after the freeze).
        #
        # Deliberately NOT pooled across attackers: pooling assumes the partitions are
        # exchangeable, which holds under IID but breaks the moment partitions go non-IID —
        # a client's benign update norm is a property of ITS partition. Keeping the
        # reference per client means nothing here has to be re-derived at that switch.
        #
        # Sample count per client = 2 x (its plastic injection rounds): 2 for a client that
        # attacks once during the plastic phase, 4 if it attacks twice, and so on. A client
        # that never attacks during the plastic phase gets NO entry (see freeze_benign_refs)
        # — there is no cross-client fallback, by design.
        self.plastic_benign_norms: dict[int, list[float]] = {}  # cid -> [||Δ_ben1||, ||Δ_ben2||, ...]
        self.plastic_benign_ref: dict[int, float] = {}          # cid -> frozen mean (absent until freeze)

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

    def accumulate_plastic(self, div, b, current_round, cid) -> None:
        """Record one (round, client) injection's per-bucket div_t / b_t.

        div and b are name->float dicts over the same (eligible) key set, so the
        derived sums stay aligned across rounds.

        IDEMPOTENT IN (current_round, cid). Calling this twice for the same round and
        client — which is what a crash-and-resume does, since the round in flight is
        re-run — REPLACES the earlier sample rather than pooling both. Without that,
        the replayed round carries double weight in mean(div)/mean(b), and the freeze
        is decided on a reweighted mean. Tensors sitting near the cutoff can cross it
        on that alone, so two runs with identical configs can end up with different
        carrying sets purely because one of them was interrupted.
        """
        key = self._plastic_key(current_round, cid)
        self.plastic_samples[key] = {
            "div": {n: float(v) for n, v in div.items()},
            "b": {n: float(v) for n, v in b.items()},
            "n": 1,
        }

    @staticmethod
    def _plastic_key(current_round, cid) -> str:
        """Identity of one pooled injection. String so it survives a JSON round-trip."""
        return f"{int(current_round)}:{int(cid)}"

    def _plastic_sum(self, field: str) -> dict[str, float]:
        """Elementwise sum of one field ('div' or 'b') across recorded samples."""
        out: dict[str, float] = {}
        for entry in self.plastic_samples.values():
            for n, v in entry[field].items():
                out[n] = out.get(n, 0.0) + v
        return out

    @property
    def plastic_div_sum(self) -> dict[str, float]:
        """name -> Σ div_t over recorded plastic injections. Derived, read-only."""
        return self._plastic_sum("div")

    @property
    def plastic_b_sum(self) -> dict[str, float]:
        """name -> Σ b_t over recorded plastic injections. Derived, read-only."""
        return self._plastic_sum("b")

    @property
    def plastic_count(self) -> int:
        """Pooled injections accumulated. Derived, read-only: a replayed (round, cid)
        overwrites its sample, so this counts DISTINCT injections, not calls."""
        return sum(entry["n"] for entry in self.plastic_samples.values())

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

    def accumulate_benign_norms(self, cid: int, norms) -> None:
        """Collect one plastic round's benign-anchor update norms FOR ONE ATTACKER.

        `norms` is an iterable of that round's anchor norms — normally both of them
        (ben1 and ben2), which is why a single plastic round contributes 2 samples.
        Appended under the client's own key; nothing is mixed across clients.
        """
        bucket = self.plastic_benign_norms.setdefault(int(cid), [])
        for v in norms:
            bucket.append(float(v))

    def freeze_benign_refs(self) -> None:
        """Freeze each attacker's MEAN benign-anchor norm as ITS exploit-phase reference.

        Mean (not median): the samples are repeated measurements of the same quantity for
        one client, so averaging them reduces variance — there is no cross-client
        contamination for a median to be robust against.

        A client with no collected samples gets NO entry: it never attacked during the
        plastic phase, so there is no benign baseline for it and there is deliberately no
        cross-client fallback (that would be exactly the pooling this design removes).
        Callers must handle a missing reference explicitly.
        """
        self.plastic_benign_ref = {
            cid: statistics.fmean(vals)
            for cid, vals in self.plastic_benign_norms.items() if vals
        }

    def get_benign_ref(self, cid: int):
        """This attacker's frozen benign-anchor reference, or None if it has none."""
        return self.plastic_benign_ref.get(int(cid))

    def benign_ref_sample_counts(self) -> dict:
        """{cid: #anchor norms collected} — for the freeze-time coverage report."""
        return {cid: len(v) for cid, v in self.plastic_benign_norms.items()}

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
        self.plastic_samples.clear()
        self.plastic_frozen = False
        self.plastic_frozen_set.clear()
        self.plastic_benign_norms.clear()
        self.plastic_benign_ref.clear()
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
                # Source of truth: one entry per (round, cid), so a resume that replays
                # a plastic round overwrites rather than double-counts.
                "samples": self.plastic_samples,
                # Derived mirrors. Written for readability and for any older reader that
                # still expects them; from_dict PREFERS "samples" and only falls back to
                # these when loading a checkpoint written before the keying change.
                "div_sum": self.plastic_div_sum,
                "b_sum": self.plastic_b_sum,
                "count": self.plastic_count,
                "frozen": self.plastic_frozen,
                "frozen_set": sorted(self.plastic_frozen_set),
                # Per-client benign-norm baseline (persisted so each attacker's frozen
                # reference survives a resume; ephemeral normclip_ctx is NOT persisted).
                # JSON keys must be strings — same convention as carrying_set above.
                "benign_norms_by_cid": {str(c): list(v) for c, v in self.plastic_benign_norms.items()},
                "benign_ref_by_cid": {str(c): v for c, v in self.plastic_benign_ref.items()},
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
        samples = p.get("samples")
        if samples is not None:
            self.plastic_samples = {
                str(k): {
                    "div": {n: float(v) for n, v in e.get("div", {}).items()},
                    "b": {n: float(v) for n, v in e.get("b", {}).items()},
                    "n": int(e.get("n", 1)),
                }
                for k, e in samples.items()
            }
        else:
            # Checkpoint written before the accumulator was keyed: it holds only the
            # aggregate, so the individual injections are unrecoverable. Carry it as one
            # weighted entry, which reproduces the same means and count exactly. Rounds
            # already folded into it cannot be de-duplicated retroactively — only
            # injections recorded after this load get replay protection.
            count = int(p.get("count", 0))
            self.plastic_samples = (
                {
                    "legacy": {
                        "div": {n: float(v) for n, v in p.get("div_sum", {}).items()},
                        "b": {n: float(v) for n, v in p.get("b_sum", {}).items()},
                        "n": count,
                    }
                }
                if count
                else {}
            )
        self.plastic_frozen = bool(p.get("frozen", False))
        self.plastic_frozen_set = set(p.get("frozen_set", ()))
        self.plastic_benign_norms = {
            int(c): [float(v) for v in vals]
            for c, vals in p.get("benign_norms_by_cid", {}).items()
        }
        self.plastic_benign_ref = {
            int(c): float(v) for c, v in p.get("benign_ref_by_cid", {}).items()
        }
        # Legacy (pre-per-client) snapshots stored one POOLED list + a median under
        # "benign_norms"/"benign_norm_ref". Those samples cannot be re-attributed to the
        # clients that produced them, so they are dropped rather than silently reused as
        # if they were per-client. A run resuming mid-plastic simply re-accumulates.
        if "benign_norms_by_cid" not in p and ("benign_norms" in p or "benign_norm_ref" in p):
            print("[AttackState] legacy POOLED benign-norm baseline found and DISCARDED "
                  "(it predates per-client references and cannot be re-attributed). "
                  "Plastic-phase benign norms will re-accumulate from this round.")

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
