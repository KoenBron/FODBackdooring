"""Dry-run the partitioner without training — the numbers that decide alpha and the rule.

Run from project root:
    python scripts/partition_report.py
    python scripts/partition_report.py --alphas 0.1,0.5,5.0 --clients 20 --attackers 0,1,2,3

Prints three things:

  1. The LABEL-BUCKET histogram under every proxy rule, side by side. This answers the
     question the rule choice hangs on: does one class swallow the dataset? Under
     'max_area' a person-heavy set may put most images in one bucket, which caps how much
     skew any alpha can actually produce.

  2. For each alpha, the per-client composition table the run would produce — including
     drift_tv, the total-variation distance between the composition a client was DRAWN and
     the one it RECEIVED. Read drift first: supply runs out at small alpha and clients
     take whatever is left (partition SIZE is invariant, composition fidelity is not), so
     a large drift means the requested skew was only partly realised.

  3. The same partition drawn with PLAIN (textbook) per-class Dirichlet, for reference
     only -- it is never used for training. Shows what the equal-size design refuses:
     client sizes become a consequence of the draw, and with them the model-replacement
     factor gamma = k/(eta*m) stops being exact. Note that alpha is NOT comparable
     between the two schemes; compare the TV columns, not the symbol.

Loads annotations only — no images, no torch, no GPU.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.global_settings import BackdoorProperties, FLSettings, ProjectPaths
from utils.noniid import (
    LABEL_RULES,
    _largest_remainder,
    _rng,
    allocate_label_skew,
    bucket_names,
    build_image_labels,
    composition_rows,
    global_marginal,
    print_composition_table,
)


def load_annotations(path):
    """The two maps utils.dataset builds, without constructing the torch Dataset."""
    import json
    from collections import defaultdict
    with open(path) as f:
        raw = json.load(f)
    img_ann_map = defaultdict(list)
    for ann in raw["annotations"]:
        img_ann_map[ann["image_id"]].append(ann)
    cat_map = {c["id"]: c["name"] for c in raw["categories"]}
    img_ids = [img["id"] for img in raw["images"]]
    return img_ids, img_ann_map, cat_map


def bucket_counts(labels, n_buckets):
    counts = [0] * n_buckets
    for k in labels:
        counts[k] += 1
    return counts


def bucket_histogram(labels, names, top=15):
    counts = bucket_counts(labels, len(names))
    order = sorted(range(len(names)), key=lambda i: (-counts[i], names[i]))[:top]
    return [(names[i], counts[i], 100.0 * counts[i] / max(1, len(labels))) for i in order]


# ---- two views of the same partition: proxy-level and object-level ----

def object_vectors(img_ids, img_ann_map, class_index, n_buckets):
    """Per-image OBJECT-count vector over the dense buckets.

    The proxy files each image under one class; this counts every annotation in it. Both
    are reported because they disagree, and the disagreement is not small: filing a
    multi-object image under a single class skews the PROXY distribution far more than it
    skews the OBJECT distribution the detector's loss actually sees. Any heterogeneity
    number quoted at proxy level is therefore an upper bound on the real thing.

    An image with no (known) annotations contributes one unit to the background bucket, so
    the object marginal is defined over the same axis as the proxy marginal.
    """
    vecs = []
    for img_id in img_ids:
        v = [0] * n_buckets
        hit = False
        for ann in img_ann_map.get(img_id, []):
            k = class_index.get(ann.get("category_id"))
            if k is not None:
                v[k] += 1
                hit = True
        if not hit:
            v[n_buckets - 1] += 1
        vecs.append(v)
    return vecs


def total_variation(counts, reference):
    """TV distance between a realised count vector and a reference proportion vector.

    NaN for an empty partition — under the plain-Dirichlet scheme a client really can
    receive zero images, and reporting 0.0 there would read as 'perfectly on target'.
    """
    n = sum(counts)
    if n <= 0:
        return float("nan")
    return 0.5 * sum(abs(c / n - q) for c, q in zip(counts, reference))


def sum_vectors(part, vecs, n_buckets):
    out = [0] * n_buckets
    for i in part:
        v = vecs[i]
        for k in range(n_buckets):
            out[k] += v[k]
    return out


def class_counts(part, labels, n_buckets):
    out = [0] * n_buckets
    for i in part:
        out[labels[i]] += 1
    return out


# ---- the textbook scheme, for comparison only ----

def allocate_plain_dirichlet(labels, n_buckets, n_clients, alpha, seed):
    """TEXTBOOK per-class Dirichlet: p_k ~ Dir(alpha * 1_N), independently for each class k.

    This is NOT what --partition-mode dirichlet does, and it is never used for training.
    It exists so the report can show what the equal-size design gives up and gains: here
    client sizes are a CONSEQUENCE of the draw,

        n_j = sum_k p_kj * n_k,

    so the scheme couples label skew to QUANTITY skew, and only the first is a knob.

    Per-class counts use the same largest-remainder rounding as the real allocator, so they
    sum to the pool size exactly and no client absorbs the rounding remainder of every class.
    Its own RNG stream (3) so running this comparison cannot shift either of the two
    streams the real partitioner uses.
    """
    rng = _rng(seed, 3)
    pools = [[] for _ in range(n_buckets)]
    for i, k in enumerate(labels):
        pools[k].append(i)

    parts = [[] for _ in range(n_clients)]
    for pool in pools:
        if not pool:
            continue
        rng.shuffle(pool)
        p = rng.dirichlet([float(alpha)] * n_clients).tolist()
        counts = _largest_remainder(p, len(pool))
        start = 0
        for j in range(n_clients):
            parts[j].extend(pool[start:start + counts[j]])
            start += counts[j]
    return parts


def gamma_error_ratios(sizes, attacker_ids, clients_per_round, seed, trials=400):
    """Multiplicative error in the model-replacement factor when sizes are unequal.

    The implemented gamma = k/(eta*m) is exactly what an attacker who CANNOT observe other
    clients' dataset sizes would compute — it assumes every participant looks like itself.
    Under unequal sizes the true factor is 1/(eta*W) with W the attackers' SAMPLE share, so

        gamma_true / gamma_assumed  =  W_assumed / W_true ,   W_assumed = m/k

    is the per-round multiplicative error. Rounds are simulated by sampling `clients_per_round`
    participants; rounds with no attacker are skipped (gamma is not computed there).

    Returns the raw per-round ratios so a caller can POOL them across several partition
    draws — a single draw's error is dominated by whether that draw's attackers happened to
    be large or small, which is realisation noise, not a property of alpha.
    """
    rng = _rng(seed, 4)
    n_clients = len(sizes)
    attackers = set(attacker_ids)
    if not attackers or clients_per_round <= 1:
        # k=1 makes every ratio exactly 1.0 (the single participant is the whole sample),
        # which reads as "unequal sizes are harmless". Return nothing so the caller can say
        # the measurement was not made rather than print a meaningless 1.00.
        return []
    ratios = []
    for _ in range(trials):
        sel = rng.choice(n_clients, size=min(clients_per_round, n_clients), replace=False)
        mal = [c for c in sel.tolist() if c in attackers]
        total = sum(sizes[c] for c in sel.tolist())
        mal_total = sum(sizes[c] for c in mal)
        if not mal or total <= 0 or mal_total <= 0:
            continue
        w_true = mal_total / total
        w_assumed = len(mal) / len(sel)
        ratios.append(w_assumed / w_true)
    return ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=str(ProjectPaths.TRAIN_ANNOTATIONS_FILE))
    ap.add_argument("--clients", type=int, default=FLSettings.TOTAL_CLIENTS)
    ap.add_argument("--attackers", default=",".join(map(str, FLSettings.ATTACKER_IDS)))
    ap.add_argument("--alphas", default="0.1,0.5,5.0")
    ap.add_argument("--seed", type=int, default=FLSettings.PARTITIONING_SEED)
    ap.add_argument("--rule", default="max_area", choices=list(LABEL_RULES),
                    help="rule used for the per-alpha allocation tables (the histogram "
                         "section always shows every rule)")
    ap.add_argument("--label-p", type=float, default=FLSettings.DIRICHLET_LABEL_P)
    ap.add_argument("--target-class", default=BackdoorProperties.TARGET_CLASS)
    ap.add_argument("--target-min-fraction", type=float,
                    default=FLSettings.ATTACKER_TARGET_MIN_FRACTION)
    ap.add_argument("--target-min-count", type=int, default=FLSettings.ATTACKER_TARGET_MIN_COUNT)
    # NOT FLSettings.CLIENTS_PER_ROUND: this script never calls apply_settings, so that
    # attribute still holds the module default of 1 — and at k=1 the gamma ratio is
    # identically 1.0 by algebra (w_assumed = 1/1 = w_true), so the section printed
    # "0% of attack rounds off by >1.5x" regardless of how unequal the sizes were, i.e.
    # the exact opposite of the argument it exists to make.
    ap.add_argument("--clients-per-round", type=int, default=9,
                    help="participants per round; only used to simulate the model-replacement "
                         "gamma error in the plain-Dirichlet comparison section. Must be > 1 "
                         "for that number to mean anything.")
    ap.add_argument("--plain-trials", type=int, default=20,
                    help="partition draws averaged over in the plain-Dirichlet comparison. "
                         "A single draw's gamma error and E[q] bias are dominated by whether "
                         "that draw's attackers happened to be large or small.")
    args = ap.parse_args()

    attackers = [int(a) for a in args.attackers.split(",") if a.strip() != ""]
    alphas = [float(a) for a in args.alphas.split(",")]

    img_ids, img_ann_map, cat_map = load_annotations(args.annotations)
    print(f"[dataset] {len(img_ids)} images, {len(cat_map)} categories, "
          f"{args.clients} clients -> {len(img_ids) // args.clients} images each "
          f"({len(img_ids) % args.clients} dropped)")

    # ---- 1. bucket histograms, one column per rule ----
    labels_by_rule = {}
    for rule in LABEL_RULES:
        labels, class_index = build_image_labels(
            img_ids, img_ann_map, cat_map.keys(), rule, args.seed, args.label_p)
        names = bucket_names(cat_map, class_index)
        labels_by_rule[rule] = (labels, names, class_index)
        nonempty = sum(1 for c in bucket_counts(labels, len(names)) if c > 0)
        print(f"\n[rule={rule}] {nonempty}/{len(names)} buckets non-empty; top 15:")
        for name, count, pct in bucket_histogram(labels, names):
            print(f"    {name:<24} {count:>7}  {pct:>5.1f}%")

    # ---- 2. allocation tables, one per alpha ----
    labels, names, class_index = labels_by_rule[args.rule]
    target_idx = names.index(args.target_class) if args.target_class in names else None
    contains_target = set()
    if args.target_class in cat_map.values():
        tgt_id = next(k for k, v in cat_map.items() if v == args.target_class)
        contains_target = {i for i, img_id in enumerate(img_ids)
                           if any(a["category_id"] == tgt_id for a in img_ann_map.get(img_id, []))}
        print(f"\n[target] '{args.target_class}' present in {len(contains_target)} images; "
              f"proxy-labelled '{args.target_class}' in "
              f"{sum(1 for k in labels if k == target_idx)} images (rule={args.rule})")

    K = len(names)
    S_IID = len(labels) // args.clients
    vecs = object_vectors(img_ids, img_ann_map, class_index, K)
    proxy_marginal = global_marginal(labels, K)
    obj_total = [sum(v[k] for v in vecs) for k in range(K)]
    obj_marginal = [c / max(1, sum(obj_total)) for c in obj_total]

    skew_summary = {}
    for alpha in alphas:
        print(f"\n{'=' * 78}\n[alpha={alpha}] rule={args.rule}")
        try:
            client_indices, requested = allocate_label_skew(
                labels=labels, n_buckets=K, n_clients=args.clients,
                attacker_ids=attackers, alpha=alpha, seed=args.seed,
                target_idx=target_idx, target_min_count=args.target_min_count,
                target_min_fraction=args.target_min_fraction)
        except (ValueError, AssertionError) as exc:
            print(f"  FAILED: {exc}")
            continue
        rows = composition_rows(client_indices, labels, requested, names, attackers,
                                target_name=args.target_class,
                                contains_target=contains_target)
        print_composition_table(rows, names)
        drift = [r["drift_tv"] for r in rows]
        # The attackers' single shared draw, printed because the ENTIRE attack
        # configuration rests on it: one Dirichlet sample, no averaging or rejection, so a
        # different partitioning seed is a different experiment. Recorded here so a run can
        # be identified by what its attackers actually held.
        if attackers:
            q_att = requested[attackers[0]]
            top = sorted(range(K), key=lambda k: -q_att[k])[:5]
            print("  attacker shared draw q_att (one sample; the whole attack rests on it): "
                  + ", ".join(f"{names[k]}={q_att[k]:.3f}" for k in top if q_att[k] > 0.005))
        # Heterogeneity against the DATASET marginal, at both levels. drift_tv above is a
        # different quantity (requested vs realised), so the two are printed apart.
        p_tv = [total_variation(class_counts(p, labels, K), proxy_marginal) for p in client_indices]
        o_tv = [total_variation(sum_vectors(p, vecs, K), obj_marginal) for p in client_indices]
        skew_summary[alpha] = (sum(p_tv) / len(p_tv), sum(o_tv) / len(o_tv))
        # Split by role. Attackers draw from FULL pools and get their request almost
        # exactly; benign clients are dealt afterwards and absorb ALL of the supply error.
        # A single pooled mean hides that asymmetry, and the asymmetry matters: it widens
        # the benign cone by more than the intended skew while leaving attacker updates on
        # a cleaner draw, which is precisely the contrast the stealth metrics measure.
        d_att = [r["drift_tv"] for r in rows if r["role"] == "attacker"]
        d_ben = [r["drift_tv"] for r in rows if r["role"] == "benign"]
        print(f"  drift_tv (requested vs realised): mean={sum(drift) / len(drift):.3f} "
              f"max={max(drift):.3f}   (0 = the requested skew was realised exactly)")
        if d_att and d_ben:
            print(f"    by role: attacker mean={sum(d_att) / len(d_att):.3f}  "
                  f"benign mean={sum(d_ben) / len(d_ben):.3f} "
                  f"(max={max(d_ben):.3f})   <- benign absorb the supply error")

        # ATTACKER TARGET SUPPLY vs the IID expectation. Without the quota there is no
        # floor at all -- the attacker's target-class holding is whatever its single
        # Dirichlet draw plus class CO-OCCURRENCE happens to deliver, and on VOC that
        # varies ~5x across seeds at alpha=0.1. Target supply is the most direct driver of
        # backdoor strength there is, so an ASR gap between two alphas can easily be a
        # supply gap instead. Printed against the IID expectation, with a warning, so the
        # seed can be screened BEFORE a 40-hour run rather than diagnosed after it.
        if contains_target and attackers:
            expected = S_IID * len(contains_target) / len(labels)
            att_rows = [r for r in rows if r["role"] == "attacker"]
            got = sum(r["contains_target"] for r in att_rows) / len(att_rows)
            ben_have = [r["contains_target"] for r in rows if r["role"] == "benign"]
            pct = 100.0 * got / expected if expected else float("nan")
            flag = "" if 85.0 <= pct <= 115.0 else "   <-- OFF THE IID EXPECTATION"
            print(f"  attacker target supply: {got:.0f} images containing "
                  f"'{args.target_class}' vs {expected:.0f} expected under IID "
                  f"({pct:.0f}%){flag}")
            print(f"    benign min={min(ben_have)} max={max(ben_have)}; no floor is "
                  f"enforced — this is the draw plus class co-occurrence")
            if not 85.0 <= pct <= 115.0:
                print(f"    {'':4}Screen --seed, or hold supply fixed across conditions; "
                      f"otherwise alpha and target supply move together.")
        print(f"  heterogeneity vs dataset marginal: proxy-level={skew_summary[alpha][0]:.3f}  "
              f"object-level={skew_summary[alpha][1]:.3f}   "
              f"(object-level is what the detector's loss sees)")

    # ---- IID baseline: the NOISE FLOOR every TV above must be read against ----
    # "~0 by construction" is not good enough to calibrate alpha. An IID split of S images
    # still lands off the marginal by pure multinomial sampling noise, and on a 20-class
    # detection set that floor is a few percent — so an object-level TV of 0.16 is 3x the
    # floor, not "nearly IID". Measured, not asserted: real permutations, same S.
    marginal = global_marginal(labels, K)
    iid_seeds = 5
    p_floor, o_floor = [], []
    _r = _rng(args.seed, 5)
    for _ in range(iid_seeds):
        perm = _r.permutation(len(labels)).tolist()
        for j in range(args.clients):
            part = perm[j * S_IID:(j + 1) * S_IID]
            p_floor.append(total_variation(class_counts(part, labels, K), proxy_marginal))
            o_floor.append(total_variation(sum_vectors(part, vecs, K), obj_marginal))
    p_floor = sum(p_floor) / len(p_floor)
    o_floor = sum(o_floor) / len(o_floor)
    print(f"\n{'=' * 78}\n[iid baseline] largest class share = {max(marginal):.3f}")
    print(f"  SAMPLING-NOISE FLOOR of a true IID split ({iid_seeds} permutations, "
          f"{S_IID} images each):")
    print(f"    proxy-level TV = {p_floor:.3f}    object-level TV = {o_floor:.3f}")
    print(f"  Every TV printed above sits on top of this; divide by it to read how much "
          f"skew is real.")
    if skew_summary:
        print(f"  {'alpha':>7} {'proxy TV':>9} {'x floor':>8} {'object TV':>10} {'x floor':>8}")
        for a in sorted(skew_summary):
            pv, ov = skew_summary[a]
            print(f"  {a:>7} {pv:>9.3f} {pv / p_floor:>7.1f}x {ov:>10.3f} "
                  f"{ov / o_floor:>7.1f}x")

    # ---- the textbook scheme, reported for comparison only ----
    print(f"\n{'=' * 78}")
    print("[PLAIN DIRICHLET - reference only, NOT what --partition-mode dirichlet does]")
    print("Textbook per-class split p_k ~ Dir(alpha * 1_N) over clients. Partition sizes are")
    print("a CONSEQUENCE of the draw (n_j = sum_k p_kj n_k), so the scheme couples label skew")
    print("to QUANTITY skew; the implemented mode fixes every partition at exactly S images.")
    print(f"Each row averages {args.plain_trials} independent draws (seeds {args.seed}.."
          f"{args.seed + args.plain_trials - 1}).")
    print("")
    print("  !! ALPHA IS NOT COMPARABLE BETWEEN THE TWO SCHEMES. This draws Dir(alpha*1_N)")
    print("     over N clients (total concentration alpha*N); the implemented mode draws")
    print("     Dir(alpha*K*m) over K classes (total concentration alpha*K). Matching the")
    print("     SYMBOL does not match the severity. Compare the TV columns, which are the")
    print("     same quantity in both, and read off which alpha each scheme needs to reach")
    print("     a given TV -- do not read the bracketed figures as a like-for-like contest.")
    S = len(labels) // args.clients
    for alpha in alphas:
        cvs, mins, maxs, ptvs, otvs, ratios = [], [], [], [], [], []
        empty_total = 0
        comp_sum, comp_n = [0.0] * K, 0
        for t in range(args.plain_trials):
            parts = allocate_plain_dirichlet(labels, K, args.clients, alpha, args.seed + t)
            sizes = [len(p) for p in parts]
            if sum(sizes) != len(labels):
                raise AssertionError("plain-Dirichlet split lost or duplicated images")
            mean_n = sum(sizes) / len(sizes)
            var = sum((n - mean_n) ** 2 for n in sizes) / len(sizes)
            cvs.append((var ** 0.5) / mean_n if mean_n else float("nan"))
            mins.append(min(sizes)); maxs.append(max(sizes))
            empty_total += sum(1 for n in sizes if n == 0)
            for part in parts:
                n = len(part)
                if n == 0:
                    continue                      # composition undefined, not zero
                cc = class_counts(part, labels, K)
                ptvs.append(total_variation(cc, proxy_marginal))
                otvs.append(total_variation(sum_vectors(part, vecs, K), obj_marginal))
                for k in range(K):
                    comp_sum[k] += cc[k] / n
                comp_n += 1
            ratios.extend(gamma_error_ratios(sizes, attackers, args.clients_per_round,
                                             args.seed + t))
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        ref = skew_summary.get(alpha)
        print(f"\n  [alpha={alpha}]")
        print(f"    sizes             CV={mean(cvs):.2f} (mean over draws)  smallest client seen"
              f"={min(mins)}  largest={max(maxs)}")
        print(f"                      [implemented mode: every client exactly {S}, CV=0]")
        if empty_total:
            print(f"    EMPTY clients     {empty_total} across {args.plain_trials} draws "
                  f"({empty_total / args.plain_trials:.1f} per draw) - excluded from the TV means")
        print(f"    proxy-level TV    {mean(ptvs):.3f}"
              + (f"   [implemented mode at ITS alpha={alpha}: {ref[0]:.3f}]" if ref else ""))
        print(f"    object-level TV   {mean(otvs):.3f}"
              + (f"   [implemented mode at ITS alpha={alpha}: {ref[1]:.3f}]" if ref else "")
              + "   <- what the detector's loss sees")
        if comp_n:
            bias = max(abs(comp_sum[k] / comp_n - proxy_marginal[k]) for k in range(K))
            print(f"    E[q] bias         max_k |E[q_k] - m_k| = {bias:.3f}  over {comp_n} client-draws")
            print(f"                      (composition is a RATIO of two correlated random")
            print(f"                       quantities, so its mean is pulled off the marginal;")
            print(f"                       the gap shrinks with the size spread, not with N)")
        if ratios:
            ratios.sort()
            pick = lambda f: ratios[min(len(ratios) - 1, int(f * len(ratios)))]
            off = sum(1 for r in ratios if r > 1.5 or r < 1 / 1.5) / len(ratios)
            print(f"    gamma error       median {pick(0.5):.2f}  IQR [{pick(0.25):.2f}, "
                  f"{pick(0.75):.2f}]  {off:.0%} of attack rounds off by >1.5x "
                  f"(k={args.clients_per_round})")
            print(f"                      (gamma = k/(eta*m) is what an attacker who cannot see")
            print(f"                       other clients' sizes computes; this is the factor it")
            print(f"                       is wrong by. Equal sizes make it exact.)")
        else:
            print(f"    gamma error       not measured (needs --clients-per-round > 1 and "
                  f"at least one attacker)")


if __name__ == "__main__":
    main()
