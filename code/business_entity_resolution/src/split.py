"""Reproducible 80/20 train/validation split over train S1 entity IDs.

Stratified jointly by country and singleton/non-singleton (seed 42).
Writes artifacts/val_split.tsv with columns:
  entity_id, country, is_singleton, n_matches, split   (split in {train, val})
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from io_utils import ARTIFACTS_DIR, load_ground_truth, load_source

SEED = 42


def make_split(val_frac: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    s1 = load_source("train", 1, usecols=["entity_id", "country"])
    gt = load_ground_truth()
    s1["n_matches"] = s1["entity_id"].map(lambda e: len(gt.get(e, ()))).astype(int)
    s1["is_singleton"] = (s1["n_matches"] == 0).astype(int)
    # Sort first so the result does not depend on file row order.
    s1 = s1.sort_values("entity_id", kind="stable").reset_index(drop=True)
    strata = s1["country"].astype(str) + "|" + s1["is_singleton"].astype(str)
    _, val_idx = train_test_split(s1.index, test_size=val_frac, random_state=seed,
                                  stratify=strata)
    s1["split"] = "train"
    s1.loc[val_idx, "split"] = "val"
    return s1[["entity_id", "country", "is_singleton", "n_matches", "split"]]


def make_query_sets(split_df: pd.DataFrame, n_ft: int, n_s3: int,
                    seed: int = SEED) -> pd.DataFrame:
    """Disjoint S1 query sets used by the later stages.

    val     : the whole validation split (never trained on)
    ft      : TRAIN-split sample used to fine-tune the embedding model
    s3train : a different TRAIN-split sample whose candidates train the Stage-3
              matcher (disjoint from `ft`, so its embedding scores are out-of-sample
              just like on val/test)
    """
    tr = split_df[split_df["split"] == "train"].sample(frac=1.0, random_state=seed)
    val = split_df[split_df["split"] == "val"]
    out = pd.concat([
        val.assign(qset="val"),
        tr.iloc[:n_ft].assign(qset="ft"),
        tr.iloc[n_ft:n_ft + n_s3].assign(qset="s3train"),
    ])
    return out[["entity_id", "country", "is_singleton", "n_matches", "qset"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(ARTIFACTS_DIR / "val_split.tsv"))
    ap.add_argument("--n-ft", type=int, default=300_000)
    ap.add_argument("--n-s3", type=int, default=200_000)
    args = ap.parse_args()

    df = make_split(args.val_frac, args.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    print(pd.crosstab([df["country"], df["is_singleton"]], df["split"], margins=True))

    qs = make_query_sets(df, args.n_ft, args.n_s3, args.seed)
    qs.to_csv(ARTIFACTS_DIR / "query_sets.tsv", sep="\t", index=False)
    print(f"Wrote {ARTIFACTS_DIR / 'query_sets.tsv'}")
    print(pd.crosstab(qs["country"], qs["qset"], margins=True))


if __name__ == "__main__":
    main()
