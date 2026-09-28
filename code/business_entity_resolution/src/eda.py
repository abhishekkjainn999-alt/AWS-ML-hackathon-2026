"""Exploratory data analysis for the Business Entity Resolution challenge.

Usage:  python src/eda.py [--seed 42] [--pairs-per-country 15]
Prints everything to stdout (redirect to a file to keep a report).
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from io_utils import load_ground_truth, load_source

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 70)

TEXT_COLS = ["business_name", "business_address", "country"]


def hr(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def load_split(split: str) -> dict[int, pd.DataFrame]:
    return {s: load_source(split, s) for s in (1, 2, 3)}


def basic_profile(split: str, srcs: dict[int, pd.DataFrame]) -> None:
    hr(f"[{split}] Row counts and country distribution")
    for s, df in srcs.items():
        dup = df["entity_id"].duplicated().sum()
        print(f"S{s}: {len(df):>10,} rows  (duplicate entity_ids: {dup})")
    dist = pd.DataFrame({f"S{s}": df["country"].value_counts() for s, df in srcs.items()})
    pct = dist / dist.sum() * 100
    print("\nCounts:\n", dist.fillna(0).astype(int).to_string())
    print("\nPercent:\n", pct.round(2).fillna(0).to_string())

    hr(f"[{split}] Null / empty (whitespace-only) rates per column, %")
    rows = {}
    for s, df in srcs.items():
        rows[f"S{s}"] = {c: (df[c].str.strip() == "").mean() * 100 for c in TEXT_COLS}
    print(pd.DataFrame(rows).T.round(3).to_string())
    print("\nEmpty rate by country (name / address), %:")
    for s, df in srcs.items():
        g = df.groupby("country")[["business_name", "business_address"]].agg(
            lambda x: (x.str.strip() == "").mean() * 100)
        print(f"  S{s}:\n" + g.round(3).to_string().replace("\n", "\n    "))


DEVANAGARI = re.compile(r"[ऀ-ॿ]")
NON_ASCII = re.compile(r"[^\x00-\x7F]")
DIGITS5 = re.compile(r"(?<!\d)\d{5}(?!\d)")
DIGITS6 = re.compile(r"(?<!\d)\d{6}(?!\d)|(?<!\d)\d{3} \d{3}(?!\d)")
LEADING_JUNK = re.compile(r"^\W")


def noise_profile(split: str, srcs: dict[int, pd.DataFrame], n: int, seed: int) -> None:
    """Surface-level text noise indicators per source x country (on a sample)."""
    hr(f"[{split}] Surface noise indicators per source x country (sample of {n:,}/source), %")
    out = []
    for s, df in srcs.items():
        d = df.sample(min(n, len(df)), random_state=seed)
        name, addr = d["business_name"], d["business_address"]
        feat = pd.DataFrame({
            "country": d["country"],
            "name_non_ascii": name.str.contains(NON_ASCII),
            "name_devanagari": name.str.contains(DEVANAGARI),
            "name_leading_punct": name.str.contains(LEADING_JUNK),
            "name_all_upper": name.str.isupper(),
            "name_all_lower": name.str.islower(),
            "addr_all_upper": addr.str.isupper(),
            "addr_5digit": addr.str.contains(DIGITS5),
            "addr_6digit": addr.str.contains(DIGITS6),
            "addr_near": addr.str.contains(r"\b(?:near|opp|opposite|behind)\b", case=False),
        })
        g = feat.groupby("country").mean() * 100
        g.insert(0, "source", f"S{s}")
        out.append(g)
    print(pd.concat(out).reset_index().set_index(["source", "country"]).round(1).to_string())


def match_stats(srcs: dict[int, pd.DataFrame], gt: dict[str, set[str]]) -> pd.DataFrame:
    s1 = srcs[1]
    hr("[train] Ground truth: singletons and match-count distribution")
    s1_ids = set(s1["entity_id"])
    print(f"GT rows: {len(gt):,}   S1 rows: {len(s1):,}   "
          f"S1 ids missing from GT: {len(s1_ids - gt.keys())}   "
          f"GT ids not in S1: {len(gt.keys() - s1_ids)}")

    n_match = s1["entity_id"].map(lambda e: len(gt.get(e, ()))).astype(int)
    print(f"Singletons: {(n_match == 0).mean() * 100:.2f}%")
    bucket = pd.cut(n_match, [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    tab = pd.crosstab(s1["country"], bucket, normalize="index") * 100
    tab.loc["ALL"] = bucket.value_counts(normalize=True).reindex(tab.columns) * 100
    print("\nMatch-count buckets, % of S1 entities (row = S1 country):\n",
          tab.round(2).to_string())
    print("\nFine-grained match-count distribution (all):\n",
          n_match.value_counts().sort_index().to_string())

    # Explode into pairs
    pairs = pd.DataFrame([(s, m) for s, ms in gt.items() for m in ms],
                         columns=["s1_id", "m_id"])
    pairs["src"] = pairs["m_id"].str[:2]

    hr("[train] How matches split between S2 and S3")
    print(f"Total true pairs: {len(pairs):,}")
    print(pairs["src"].value_counts(normalize=True).mul(100).round(2).to_string())
    per = pairs.groupby(["s1_id", "src"]).size().unstack(fill_value=0)
    for col in ("S2", "S3"):
        if col not in per:
            per[col] = 0
    kind = np.select([(per.S2 > 0) & (per.S3 > 0), per.S2 > 0], ["both", "S2 only"], "S3 only")
    print("\nNon-singleton S1 entities by where their matches live (%):\n",
          pd.Series(kind).value_counts(normalize=True).mul(100).round(2).to_string())
    print("\n#S2 matches per non-singleton S1:\n",
          per.S2.clip(upper=5).value_counts().sort_index().to_string())
    print("\n#S3 matches per non-singleton S1:\n",
          per.S3.clip(upper=5).value_counts().sort_index().to_string())

    # Coverage of S2/S3 by GT, and uniqueness of assignment
    hr("[train] S2/S3 records: how many are linked to some S1 entity?")
    m_counts = pairs["m_id"].value_counts()
    print(f"S2/S3 ids linked to >1 S1 entity: {(m_counts > 1).sum():,}")
    for s in (2, 3):
        ids = srcs[s]["entity_id"]
        linked = ids.isin(m_counts.index)
        print(f"S{s}: {linked.mean() * 100:.2f}% of records are matched to an S1 entity "
              f"({(~linked).sum():,} unlinked distractors)")
        print("   linked % by country:",
              (linked.groupby(srcs[s]["country"]).mean() * 100).round(2).to_dict())
    all_m = pd.concat([srcs[2], srcs[3]], ignore_index=True)
    missing = set(pairs["m_id"]) - set(all_m["entity_id"])
    print(f"GT match ids not found in S2/S3 files: {len(missing)}")

    # Attach records to pairs
    left = s1.rename(columns=lambda c: f"s1_{c}" if c != "entity_id" else "s1_id")
    right = all_m.rename(columns=lambda c: f"m_{c}" if c != "entity_id" else "m_id")
    pairs = pairs.merge(left, on="s1_id", how="left").merge(right, on="m_id", how="left")

    hr("[train] Do matched pairs ever have DIFFERENT country labels?")
    diff = pairs[pairs["s1_country"] != pairs["m_country"]]
    print(f"Pairs with differing country: {len(diff):,} / {len(pairs):,} "
          f"({len(diff) / max(len(pairs), 1) * 100:.4f}%)")
    if len(diff):
        print(pd.crosstab(diff["s1_country"], diff["m_country"]).to_string())
        print(diff[["s1_business_name", "s1_country", "m_business_name", "m_country"]]
              .head(10).to_string())
    return pairs


def show_pairs(pairs: pd.DataFrame, k: int, seed: int) -> None:
    hr(f"[train] {k} random matched pairs per S1 country (S1 vs matched S2/S3)")
    for country, g in pairs.groupby("s1_country"):
        print(f"\n--- country = {country} ---")
        for _, r in g.sample(min(k, len(g)), random_state=seed).iterrows():
            print(f"{r.s1_id:<14} N: {r.s1_business_name}\n{'':<14} A: {r.s1_business_address}")
            print(f"{r.m_id:<14} N: {r.m_business_name}\n{'':<14} A: {r.m_business_address}"
                  f"   [{r.m_country}]")
            print("-" * 60)


def pair_similarity(pairs: pd.DataFrame, n: int, seed: int) -> None:
    """How noisy are true pairs? Similarity stats on a sample of matched pairs."""
    hr(f"[train] Similarity of TRUE pairs (sample {n:,}); rapidfuzz scores 0-100")
    d = pairs.sample(min(n, len(pairs)), random_state=seed).copy()
    d["src"] = d["m_id"].str[:2]

    def norm(x: str) -> str:
        return re.sub(r"[^\w]+", " ", x.lower()).strip()

    sn, mn = d["s1_business_name"].map(norm), d["m_business_name"].map(norm)
    sa, ma = d["s1_business_address"].map(norm), d["m_business_address"].map(norm)
    d["name_exact"] = d["s1_business_name"] == d["m_business_name"]
    d["name_norm_exact"] = sn == mn
    d["name_tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(sn, mn)]
    d["name_ratio"] = [fuzz.ratio(a, b) for a, b in zip(sn, mn)]
    d["addr_tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(sa, ma)]
    d["addr_empty"] = d["m_business_address"].str.strip() == ""
    d["name_script_diff"] = (d["s1_business_name"].str.contains(NON_ASCII)
                             != d["m_business_name"].str.contains(NON_ASCII))
    agg = d.groupby(["s1_country", "src"]).agg(
        n=("name_tsr", "size"),
        name_exact_pct=("name_exact", lambda x: x.mean() * 100),
        name_norm_exact_pct=("name_norm_exact", lambda x: x.mean() * 100),
        name_tsr_median=("name_tsr", "median"),
        name_tsr_p10=("name_tsr", lambda x: x.quantile(0.10)),
        name_ratio_median=("name_ratio", "median"),
        addr_tsr_median=("addr_tsr", "median"),
        addr_tsr_p10=("addr_tsr", lambda x: x.quantile(0.10)),
        m_addr_empty_pct=("addr_empty", lambda x: x.mean() * 100),
        script_diff_pct=("name_script_diff", lambda x: x.mean() * 100),
    )
    print(agg.round(1).to_string())
    print("\nHardest true pairs by name (lowest token_set_ratio):")
    hard = d.nsmallest(12, "name_tsr")
    for _, r in hard.iterrows():
        print(f"  [{r.s1_country}] {r.s1_business_name!r:<45} <-> {r.m_business_name!r}")


def test_profile(srcs: dict[int, pd.DataFrame], train_countries: set[str], seed: int) -> None:
    unseen = sorted(set().union(*(set(df["country"]) for df in srcs.values())) - train_countries)
    hr(f"[test] Countries not seen in train: {unseen}")
    for c in unseen:
        for s, df in srcs.items():
            sub = df[df["country"] == c]
            print(f"\n--- S{s}, country = {c}: {len(sub):,} records; 10 random examples ---")
            if len(sub):
                print(sub.sample(min(10, len(sub)), random_state=seed)[
                    ["entity_id", "business_name", "business_address"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs-per-country", type=int, default=15)
    ap.add_argument("--sample", type=int, default=200_000)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    train = load_split("train")
    basic_profile("train", train)
    noise_profile("train", train, args.sample, args.seed)
    gt = load_ground_truth()
    pairs = match_stats(train, gt)
    show_pairs(pairs, args.pairs_per_country, args.seed)
    pair_similarity(pairs, args.sample, args.seed)
    train_countries = set(train[1]["country"])
    del train, pairs, gt

    test = load_split("test")
    basic_profile("test", test)
    noise_profile("test", test, args.sample, args.seed)
    test_profile(test, train_countries, args.seed)


if __name__ == "__main__":
    main()
