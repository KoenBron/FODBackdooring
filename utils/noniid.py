"""Non-IID partitioning by LABEL SKEW at EQUAL partition size.

Every client receives exactly the same NUMBER of images; only the class composition
differs, drawn per client as q_j ~ Dir(alpha * K * marginal) over the K classes — the
dataset's own class marginal as the base measure, so E[q_j] = marginal and alpha ->
infinity is genuine IID (see _concentration). This is deliberately NOT the textbook
Dirichlet partition (which draws p_k ~ Dir(alpha * 1_N) per class and lets client sizes
fall out as a consequence):

  * Under the textbook scheme label skew and QUANTITY skew are coupled, and quantity
    skew moves things this thesis needs held fixed. FedAvg weights by num_samples, so
    the model-replacement factor becomes gamma = 1/(eta*W) with W the attackers' sample
    share — no longer k/(eta*m). With epochs fixed, the number of SGD steps (and hence
    ||Delta||) scales with partition size, so both the attacker's submitted norm and the
    benign norm median move with the draw, and a norm budget calibrated on one run stops
    transferring to the next.
  * Holding |partition| constant leaves gamma = k/(eta*m) exactly correct and keeps the
    ||Delta|| scale comparable across clients. What is left is pure ANGULAR heterogeneity:
    clients specialise toward their over-represented classes, which widens the benign
    cone (the thing cosine clustering and Krum actually consume) without touching the
    norm distribution.

Detection images hold many objects, so each image is filed under ONE proxy class for
partitioning purposes only — see image_label(), the single swappable function. The proxy
never touches training; it only decides which bucket an image is drawn from.

torch-free. numpy appears in exactly one place — the Dirichlet draw and the RNG stream
behind it — because that is the only spot where it removes a real failure mode (sampling
Dir(alpha) correctly). The bookkeeping is plain Python: index pools, largest-remainder
rounding and the inverse-CDF draw are all short loops that numpy would only obscure, and
keeping them dependency-free means the whole allocator is testable without numpy present.
"""
import math

BACKGROUND_CLASS = "_background"

LABEL_RULES = ("max_area", "count_majority", "first", "stochastic_majority")


def _rng(seed, stream):
    """A numpy Generator seeded by (partitioning seed, stream id).

    Separate streams for labelling and allocation, so changing the label rule cannot
    shift the allocator's draws and vice versa. Imported lazily so that merely importing
    this module (e.g. from parse_args for LABEL_RULES) does not require numpy.
    """
    import numpy as np
    return np.random.default_rng([int(seed), int(stream)])


# ---- step 3: the image -> single class proxy (ONE function, swappable) ----

def _ann_area(ann):
    """Annotation area in ORIGINAL image coordinates.

    Prefers COCO's own `area` field and falls back to the bbox rectangle. Deliberately
    not computed on the resized coordinates: which class dominates an image must not
    depend on RESIZE_WIDTH/RESIZE_HEIGHT (under a uniform resize the argmax is unchanged
    anyway, but reading the raw field keeps that independence obvious).
    """
    area = ann.get("area")
    if area is not None and area > 0:
        return float(area)
    bbox = ann.get("bbox")
    if not bbox or len(bbox) < 4:
        return 0.0
    return float(bbox[2]) * float(bbox[3])


def image_label(anns, rule, rng, cat_to_dense, n_classes, p_majority=0.7):
    """The single dense class index (0..n_classes-1, or n_classes for background) this
    image is filed under when partitioning. THE swap point for the proxy.

    rule:
      'max_area'            — sum annotation areas PER CLASS, take the argmax. The default:
                              "which class dominates this image", aggregated per class
                              rather than by picking one object, so five medium boats can
                              outweigh one large person.
      'count_majority'      — most instances of a class. Biased toward images holding many
                              small objects of one class (crowds of `person`).
      'first'               — class of the first-listed annotation. An arbitrary-but-
                              deterministic control: comparing against it measures how much
                              the proxy choice matters at all.
      'stochastic_majority' — 'max_area' with probability p_majority, otherwise a class
                              drawn uniformly from those PRESENT (not from all K — the
                              point is to spread an image over the buckets it could
                              plausibly belong to, not to mislabel it). Softens the proxy
                              when the deterministic rules pile too much mass on one
                              bucket; the only rule that consumes `rng`.

    Ties break on the lowest dense class index — deterministic and RNG-free, so a tie never
    depends on annotation order. An image with no annotations, or only unknown categories,
    is background (its own bucket: dropping such images would silently stop the partition
    from covering the dataset).
    """
    present, weights = [], {}
    for ann in anns:
        k = cat_to_dense.get(ann.get("category_id"))
        if k is None:
            continue
        present.append(k)
        w = 1.0 if rule == "count_majority" else _ann_area(ann)
        weights[k] = weights.get(k, 0.0) + w

    if not present:
        return n_classes
    if rule == "first":
        return present[0]
    if rule == "stochastic_majority" and rng.random() >= p_majority:
        distinct = sorted(set(present))
        return int(distinct[int(rng.integers(len(distinct)))])
    return min(weights, key=lambda k: (-weights[k], k))


def build_image_labels(img_ids, img_ann_map, class_ids, rule, seed, p_majority=0.7):
    """Apply image_label to every image. Returns (labels, class_index).

    labels[i] is the dense bucket of img_ids[i]; class_index maps category_id -> dense
    index. Buckets run 0..K-1 for the K categories plus K for background, so `labels` is a
    total function over the dataset — every image lands in exactly one bucket.
    """
    class_index = {cid: i for i, cid in enumerate(sorted(class_ids))}
    n_classes = len(class_index)
    rng = _rng(seed, 1)
    return [image_label(img_ann_map.get(i, []), rule, rng, class_index, n_classes,
                        p_majority)
            for i in img_ids], class_index


def bucket_names(cat_map, class_index):
    """Dense bucket names — the K category names in dense order, then BACKGROUND_CLASS.

    Length is n_buckets, so it indexes the same axis as labels / requested / counts.
    """
    names = [None] * len(class_index)
    for cat_id, dense in class_index.items():
        names[dense] = str(cat_map.get(cat_id, cat_id))
    return names + [BACKGROUND_CLASS]


def global_marginal(labels, n_buckets):
    """Dataset-wide class proportions — the composition an IID partition is effectively
    drawn from, so an IID run's drift is measured against the right reference."""
    counts = [0] * n_buckets
    for k in labels:
        counts[k] += 1
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * n_buckets


# ---- the allocator ----

def _largest_remainder(proportions, total):
    """Round proportions*total to integers summing EXACTLY to total (Hamilton method).

    Naive rounding does not sum to `total`, which would silently make partitions unequal —
    the one invariant this whole module exists to preserve.
    """
    raw = [p * total for p in proportions]
    counts = [int(math.floor(x)) for x in raw]
    short = total - sum(counts)
    if short < 0:
        raise AssertionError(f"proportions sum above 1 (over-allocated by {-short})")
    # Largest fractional remainder first; index breaks ties, so this is deterministic.
    order = sorted(range(len(raw)), key=lambda k: (-(raw[k] - counts[k]), k))
    for k in order[:short]:
        counts[k] += 1
    return counts


def _fit_attacker_counts(q, size, supply, target_idx, floor):
    """One integer count vector, shared by every attacker: sums to `size`, never exceeds
    the per-attacker supply, and holds at least `floor` images of the target bucket.

    Raises when the floor or the total is unsatisfiable. A silently under-quota attacker
    partition would make the backdoor fail for a reason invisible in the logs.
    """
    counts = _largest_remainder(q, size)

    if target_idx is not None and floor > 0:
        if supply[target_idx] < floor:
            raise ValueError(
                f"non-IID partitioning: the target class has only {supply[target_idx]} "
                f"image(s) available per attacker but the quota demands {floor}. Lower "
                f"--attacker-target-min-fraction / --attacker-target-min-count, or reduce "
                f"the number of attackers."
            )
        deficit = floor - counts[target_idx]
        if deficit > 0:
            counts[target_idx] += deficit
            # Pay for it out of the largest buckets first, disturbing the shape of q as
            # little as possible.
            for k in sorted(range(len(counts)), key=lambda k: (-counts[k], k)):
                if deficit <= 0:
                    break
                if k == target_idx:
                    continue
                take = min(counts[k], deficit)
                counts[k] -= take
                deficit -= take

    # Cap at supply and re-home the overflow into buckets that still have headroom,
    # preferring the buckets q asked for most. Bounded: each pass either places all the
    # overflow or fills at least one more bucket to capacity.
    protected = target_idx if (target_idx is not None and floor > 0) else None
    by_demand = sorted(range(len(counts)), key=lambda k: (-q[k], k))
    for _ in range(len(counts) + 1):
        overflow = sum(max(0, c - s) for c, s in zip(counts, supply))
        if overflow == 0:
            break
        counts = [min(c, s) for c, s in zip(counts, supply)]
        for k in by_demand:
            if overflow <= 0:
                break
            if k == protected:
                continue
            add = min(supply[k] - counts[k], overflow)
            counts[k] += add
            overflow -= add
    if sum(counts) != size:
        raise ValueError(
            f"non-IID partitioning: cannot give every attacker {size} images — only "
            f"{sum(supply)} are available per attacker across all classes. Reduce "
            f"--total-clients or the number of attackers."
        )
    return counts


def _concentration(labels, n_buckets, alpha):
    """The Dirichlet parameter vector: alpha * n_buckets * (global class marginal).

    The BASE MEASURE is the dataset's own class marginal, not a uniform vector. This is
    what makes alpha behave the way the literature's alpha does:

        E[q_j] = marginal   for every alpha,

    so alpha -> infinity converges on the dataset's true class mix — i.e. genuine IID —
    while small alpha skews around it. A flat Dir(alpha * 1_K) would instead ask every
    client for an equal share of every class, which on a skewed detection set (person
    ~47% of images, sofa ~2%) is nowhere near IID: at large alpha every client would
    request ~9% sofa against a 2% supply, and the allocator would spend the whole draw
    fighting the data. Scaling by n_buckets keeps the total concentration at alpha *
    n_buckets, so this reduces EXACTLY to Dir(alpha * 1_K) when the marginal is uniform.

    Empty buckets get a tiny floor rather than 0 — numpy requires strictly positive
    concentrations, and the allocator's availability mask excludes them anyway.
    """
    marginal = global_marginal(labels, n_buckets)
    return [max(float(alpha) * n_buckets * m, 1e-9) for m in marginal]


def _draw_class(rng, weights, available):
    """Draw a bucket from `weights`, restricted to buckets that still have supply.

    Inverse CDF against a single uniform, so exactly one draw is consumed per image and
    the stream stays easy to reason about. A client whose own distribution has no mass
    left on any available bucket falls back to uniform over what remains: it keeps its
    full partition size (the invariant) at the cost of composition fidelity, which is
    exactly what drift_tv in the report measures.
    """
    total = sum(w for w, ok in zip(weights, available) if ok)
    u = rng.random()
    if total <= 0.0:
        remaining = [k for k, ok in enumerate(available) if ok]
        return remaining[min(int(u * len(remaining)), len(remaining) - 1)]
    u *= total
    acc, last = 0.0, -1
    for k, ok in enumerate(available):
        if not ok:
            continue
        acc += weights[k]
        last = k
        if u < acc:
            return k
    return last            # only reachable through float rounding at the very top end


def allocate_label_skew(labels, n_buckets, n_clients, attacker_ids, alpha, seed,
                        target_idx=None, target_min_count=0, target_min_fraction=0.0,
                        match_attackers=False):
    """Partition image indices into n_clients EQUAL-sized, label-skewed lists.

    Returns (client_indices, requested): client_indices[c] is client c's index list, all
    of length size = len(labels) // n_clients, and requested[c] is the composition c was
    drawn — kept so the report can measure realised-vs-requested drift.

    The len(labels) % n_clients leftover images (fewer than n_clients of them) are DROPPED.
    Exactly equal sizes are the point; the IID path keeps np.array_split's +/-1 and is
    untouched.

    match_attackers controls how the malicious clients are drawn, and it is a real
    experimental choice rather than an implementation detail:

      False (default) — attackers are drawn INDEPENDENTLY, exactly like benign clients.
          Each gets its own Dirichlet draw, so colluding clients no longer share a label
          distribution. This matters because a shared distribution produces correlated
          update directions, which is precisely the signal FoolsGold keys on: matched
          attackers hand that defense close to an ideal sybil signature, and it would
          look strong for a reason that is an artifact of the partitioner. Independent
          draws remove that artifact.
          COST: the plastic phase pools div/b ACROSS attackers on an explicit
          exchangeability assumption (see FL/attack_state.py). Independent draws break
          that assumption — the pooled localisation is now averaging over clients with
          genuinely different data. Either accept it and say so, or raise
          --partial-plastic-rounds so each attacker contributes more samples.

      True — attackers share ONE draw and ONE integer count vector, so their label
          compositions are identical by construction (not merely equal in expectation),
          which keeps the pooling assumption exactly true. Retained as the second arm:
          it is the best-case bound for FoolsGold, and reporting both is the honest way
          to separate "the defense works" from "the partitioner helped it".

    Under match_attackers=True the attackers are served FIRST from full pools, which is
    also where the target-class floor is enforced. Under False there is no floor —
    see scripts/partition_report.py, which measures the realised target supply so the
    draw can be screened before a long run.

    Benign clients are then dealt ROUND-ROBIN, one image each per pass, in a per-pass
    reshuffled order. Filling clients one at a time instead would starve whoever is served
    last — they would get leftovers rather than their own draw — and that asymmetry grows
    exactly when supply is tight (small alpha), i.e. in the regime of interest.
    """
    n_total = len(labels)
    size = n_total // n_clients
    if size <= 0:
        raise ValueError(
            f"non-IID partitioning: {n_total} images cannot fill {n_clients} clients.")

    rng = _rng(seed, 2)
    concentration = _concentration(labels, n_buckets, alpha)

    pools = [[] for _ in range(n_buckets)]
    for idx, k in enumerate(labels):
        pools[k].append(idx)
    for pool in pools:
        rng.shuffle(pool)

    # Independent attackers simply fall through to the benign path: one draw each, dealt
    # round-robin with everyone else. The matched path below is skipped entirely.
    attackers = sorted(set(attacker_ids)) if match_attackers else []
    benign = [c for c in range(n_clients) if c not in set(attackers)]
    client_indices = [[] for _ in range(n_clients)]
    requested = [[0.0] * n_buckets for _ in range(n_clients)]

    # ---- attackers: one shared draw, one shared count vector ----
    if attackers:
        q_att = rng.dirichlet(concentration).tolist()
        floor = max(int(target_min_count),
                    int(math.ceil(float(target_min_fraction) * size)))
        supply = [len(p) // len(attackers) for p in pools]
        counts = _fit_attacker_counts(q_att, size, supply, target_idx, floor)
        for cid in attackers:
            requested[cid] = q_att
            for k, take in enumerate(counts):
                if take:
                    client_indices[cid].extend(pools[k][-take:])
                    del pools[k][-take:]

    # ---- benign: round-robin deal ----
    if benign:
        for cid in benign:
            requested[cid] = rng.dirichlet(concentration).tolist()
        available = [len(p) > 0 for p in pools]
        order = list(benign)
        for _ in range(size):
            rng.shuffle(order)
            for cid in order:
                k = _draw_class(rng, requested[cid], available)
                client_indices[cid].append(pools[k].pop())
                if not pools[k]:
                    available[k] = False

    for cid, part in enumerate(client_indices):
        if len(part) != size:
            raise AssertionError(
                f"non-IID partitioning: client {cid} got {len(part)} images, expected {size}")
    return client_indices, requested


# ---- reporting ----

def composition_rows(client_indices, labels, requested, class_names, attacker_ids,
                     target_name=None, contains_target=frozenset()):
    """One row per client for partition_composition.csv.

    drift_tv is the total-variation distance between the composition a client was DRAWN
    and the one it actually RECEIVED, in [0, 1]. It is the number to read first: supply
    runs out at small alpha and every client still gets its full size (the invariant) by
    taking whatever is left, so a large drift means the requested skew was only partly
    realised.

    label_target counts images whose PROXY bucket is the target class; contains_target
    counts images whose annotations actually hold one. The second is what the backdoor
    needs — the first is only the lever the allocator can pull — so both are reported.
    """
    attackers = set(attacker_ids)
    target_bucket = class_names.index(target_name) if target_name in class_names else None
    rows = []
    for cid, part in enumerate(client_indices):
        n = len(part)
        counts = [0] * len(class_names)
        for i in part:
            counts[labels[i]] += 1
        drift = 0.5 * sum(abs(c / n - q) for c, q in zip(counts, requested[cid])) if n else 0.0
        row = {
            "cid": cid,
            "role": "attacker" if cid in attackers else "benign",
            "n_images": n,
            "drift_tv": round(drift, 6),
            "label_target": counts[target_bucket] if target_bucket is not None else 0,
            "contains_target": sum(1 for i in part if i in contains_target),
        }
        for name, c in zip(class_names, counts):
            row[f"cls_{name}"] = c
        rows.append(row)
    return rows


def write_composition_csv(path, rows):
    """Wide CSV: one row per client, one column per class. Written for BOTH partition
    modes, so an IID run gives a directly comparable baseline (its drift is measured
    against the global marginal)."""
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_composition_table(rows, class_names, top_k=3):
    """Compact startup table — the skew and the attacker matching visible in the log
    without opening the CSV. Attacker rows must show identical top classes and counts."""
    print("\n[PARTITION] per-client composition (full detail in partition_composition.csv)")
    print(f"  {'cid':>4}  {'role':<8}  {'n':>6}  {'drift':>6}  {'tgt_lbl':>7}  "
          f"{'tgt_img':>7}  top-{top_k} classes")
    for r in rows:
        counts = sorted(((name, r[f"cls_{name}"]) for name in class_names),
                        key=lambda x: (-x[1], x[0]))
        top = ", ".join(f"{n}={c}" for n, c in counts[:top_k] if c > 0) or "(none)"
        print(f"  {r['cid']:>4}  {r['role']:<8}  {r['n_images']:>6}  {r['drift_tv']:>6.3f}  "
              f"{r['label_target']:>7}  {r['contains_target']:>7}  {top}")
