"""Scoring: per-S1-entity F0.5, macro-averaged, plus blocking diagnostics.

Metric (from the problem statement):
  * F0.5 = 1.25 * P * R / (0.25 * P + R), computed per S1 entity, then averaged
    over ALL S1 entities in the evaluation set.
  * Singleton (no true matches): 1.0 if prediction is empty, else 0.0.
  * Non-singleton with empty prediction: 0.0 (no true positives).
  * An S1 entity missing from the predictions is treated as an empty prediction.

CLI:
  python src/evaluate.py --pred output/matching_results.tsv \
      [--candidates output/candidate_pairs.tsv] [--split artifacts/val_split.tsv]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Iterable, Mapping

from io_utils import ARTIFACTS_DIR, load_ground_truth, load_id_list_file

BETA2 = 0.25  # beta = 0.5


def entity_f05(pred: Iterable[str], truth: Iterable[str]) -> float:
    pred, truth = set(pred or ()), set(truth or ())
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def score(predictions: Mapping[str, Iterable[str]],
          truth: Mapping[str, Iterable[str]],
          entity_ids: Iterable[str] | None = None,
          groups: Mapping[str, str] | None = None) -> dict:
    """Macro F0.5 over `entity_ids` (default: all keys of `truth`).

    `groups` optionally maps S1 id -> label (e.g. country) for a breakdown.
    Returns {'f05', 'n', 'singleton_f05', 'nonsingleton_f05', 'by_group', ...}.
    """
    ids = list(truth.keys()) if entity_ids is None else list(entity_ids)
    if not ids:
        raise ValueError("no entities to score")
    buckets: dict[str, list[float]] = defaultdict(list)
    total = 0.0
    tp = fp = fn = 0
    for s1 in ids:
        t = set(truth.get(s1, ()))
        p = set(predictions.get(s1, ()))
        f = entity_f05(p, t)
        total += f
        buckets["singleton" if not t else "non_singleton"].append(f)
        if groups is not None:
            buckets[f"group:{groups.get(s1, '?')}"].append(f)
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "f05": total / len(ids),
        "n": len(ids),
        "singleton_f05": mean(buckets["singleton"]),
        "n_singleton": len(buckets["singleton"]),
        "nonsingleton_f05": mean(buckets["non_singleton"]),
        "n_nonsingleton": len(buckets["non_singleton"]),
        "pair_precision": tp / (tp + fp) if tp + fp else float("nan"),
        "pair_recall": tp / (tp + fn) if tp + fn else float("nan"),
        "by_group": {k[6:]: (mean(v), len(v)) for k, v in sorted(buckets.items())
                     if k.startswith("group:")},
    }


def blocking_recall(candidates: Mapping[str, Iterable[str]],
                    truth: Mapping[str, Iterable[str]],
                    entity_ids: Iterable[str] | None = None) -> dict:
    """Fraction of true (S1, S2/S3) pairs present in the candidate set.

    Also reports the share of non-singletons whose matches are ALL covered,
    candidate-set size stats, and the oracle F0.5 ceiling (perfect matcher
    restricted to the candidates; singletons scored as correctly empty).
    """
    ids = list(truth.keys()) if entity_ids is None else list(entity_ids)
    true_pairs = found = n_cands = full_cover = n_nonsingle = 0
    oracle = 0.0
    for s1 in ids:
        t = set(truth.get(s1, ()))
        c = set(candidates.get(s1, ()))
        n_cands += len(c)
        hit = len(t & c)
        true_pairs += len(t)
        found += hit
        if t:
            n_nonsingle += 1
            full_cover += hit == len(t)
        oracle += entity_f05(t & c, t)
    return {
        "pair_recall": found / true_pairs if true_pairs else float("nan"),
        "true_pairs": true_pairs,
        "found_pairs": found,
        "entity_full_coverage": full_cover / n_nonsingle if n_nonsingle else float("nan"),
        "total_candidates": n_cands,
        "mean_candidates_per_s1": n_cands / len(ids) if ids else float("nan"),
        "oracle_f05": oracle / len(ids) if ids else float("nan"),
    }


def _load_split_ids(path, which):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df[df["split"] == which]
    return list(df["entity_id"]), dict(zip(df["entity_id"], df["country"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="matching_results.tsv-style file")
    ap.add_argument("--candidates", help="candidate_pairs.tsv-style file")
    ap.add_argument("--truth", default=None, help="ground truth TSV (default: train GT)")
    ap.add_argument("--split", default=str(ARTIFACTS_DIR / "val_split.tsv"),
                    help="split file from split.py; '' to score all entities")
    ap.add_argument("--which", default="val", help="split label to score (val/train)")
    args = ap.parse_args()

    truth = load_ground_truth(args.truth)
    ids, groups = (None, None)
    if args.split:
        ids, groups = _load_split_ids(args.split, args.which)
    print(f"Scoring {len(ids) if ids is not None else len(truth)} S1 entities")

    if args.pred:
        res = score(load_id_list_file(args.pred), truth, ids, groups)
        print(f"Macro F0.5: {res['f05']:.5f}")
        print(f"  singletons     : {res['singleton_f05']:.5f}  (n={res['n_singleton']})")
        print(f"  non-singletons : {res['nonsingleton_f05']:.5f}  (n={res['n_nonsingleton']})")
        print(f"  pair P/R       : {res['pair_precision']:.4f} / {res['pair_recall']:.4f}")
        for g, (f, n) in res["by_group"].items():
            print(f"  country={g:<10}: {f:.5f}  (n={n})")
    if args.candidates:
        br = blocking_recall(load_id_list_file(args.candidates), truth, ids)
        for k, v in br.items():
            print(f"  blocking {k:<24}: {v:.5f}" if isinstance(v, float) else
                  f"  blocking {k:<24}: {v}")


if __name__ == "__main__":
    main()
