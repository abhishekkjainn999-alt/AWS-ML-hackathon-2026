"""Cross-check a candidates.parquet with the independent Stage-1 scorer.

Maps integer rows back to entity ids and calls evaluate.blocking_recall.
Usage: python src/check_candidates.py --qset val
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from blocking import BLOCK_DIR, load_norm, query_positions
from evaluate import blocking_recall
from io_utils import load_ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--qset", default="val")
    args = ap.parse_args()

    _, q = query_positions(args.split, args.qset)
    df = pd.read_parquet(BLOCK_DIR / args.split / args.qset / "candidates.parquet",
                         columns=["q_row", "src", "c_row", "label"])
    cands = defaultdict(set)
    for src in (2, 3):
        ids = load_norm(args.split, src, ["entity_id"])["entity_id"].values
        d = df[df["src"] == src]
        s1 = q["entity_id"].values[d["q_row"].values]
        for a, b in zip(s1, ids[d["c_row"].values]):
            cands[a].add(b)
    gt = load_ground_truth()
    truth = {e: gt[e] for e in q["entity_id"]}
    res = blocking_recall(cands, truth)
    for k, v in res.items():
        print(f"{k:<24} {v}")
    print(f"label-column recall     {df['label'].sum() / res['true_pairs']:.6f}")
    print(f"positives / candidates  {df['label'].mean():.4%}")


if __name__ == "__main__":
    main()
