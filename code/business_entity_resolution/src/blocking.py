"""Candidate generation: a union of complementary blocking passes.

Every pass is run per (country label, source) — country is used ONLY to restrict
candidates to records carrying the same label — and writes a fixed-width top-K
matrix per source:  artifacts/blocking/{split}/{qset}/{pass}_s{src}.npz
    idx   int32 [n_queries, K]   row index into {split}_s{src}.parquet (-1 = none)
    score float32 [n_queries, K] pass score (higher = more similar)
Rank is the column position.

Passes
  name_tok  rare-token inverted index on name_core (IDF-weighted cosine, frequent tokens skipped)
  addr_tok  rare-token inverted index on the normalised address (same machinery)
  name_tri  char 3-gram TF-IDF cosine on name_core (exact, GPU cuSPARSE SpMM)
  addr_key  exact keys: (number + rare street token), (postcode + rare name token),
            (city + landmark token); oversized blocks skipped
  dense_<tag> multilingual sentence embeddings of "name | address", GPU top-K
  trans_<tag> transitive expansion through strong S2<->S3 dense neighbours

Usage:
  python src/blocking.py run  --split train --qset val --passes name_tok addr_tok name_tri addr_key
  python src/blocking.py run  --split train --qset val --passes dense_base trans_base
  python src/blocking.py eval --split train --qset val --passes ... [--ks 20 40 60 100]
  python src/blocking.py build --split train --qset val --config '{"name_tok":40,...}'
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import (HashingVectorizer, TfidfTransformer,
                                             TfidfVectorizer)

from io_utils import ARTIFACTS_DIR, load_ground_truth, vdir
from search import dense_topk, sparse_topk_cpu, sparse_topk_gpu

NORM_DIR = vdir("norm")
BLOCK_DIR = vdir("blocking")
SOURCES = (2, 3)
POOL_COLS = ["entity_id", "country", "name_core", "addr_norm", "numbers", "postcode",
             "landmark", "city"]
K_MAX = 100
MAX_DF_TOKEN = 2000        # rare-token passes skip tokens in more records than this
MAX_KEY_BLOCK = 50         # addr_key skips keys shared by more corpus records than this


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
def load_norm(split: str, src: int, cols=POOL_COLS) -> pd.DataFrame:
    return pd.read_parquet(NORM_DIR / f"{split}_s{src}.parquet", columns=cols)


def query_positions(split: str, qset: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Row positions (into {split}_s1.parquet) of the S1 entities in `qset`.

    `qset` may be a comma-separated list; the sets are concatenated in that order
    (each block sorted by position) and labelled in column `qset`, so one run can
    share the expensive corpus-side work across several query sets.
    """
    s1 = load_norm(split, 1)
    if qset == "all":
        return np.arange(len(s1)), s1.assign(qset="all")
    qs = pd.read_csv(ARTIFACTS_DIR / "query_sets.tsv", sep="\t", dtype=str)
    index = pd.Index(s1["entity_id"])
    blocks = []
    for name in qset.split(","):
        pos = index.get_indexer(qs.loc[qs["qset"] == name, "entity_id"])
        assert (pos >= 0).all()
        blocks.append((name, np.sort(pos)))
    pos = np.concatenate([p for _, p in blocks])
    q = s1.iloc[pos].reset_index(drop=True)
    q["qset"] = np.concatenate([[n] * len(p) for n, p in blocks])
    return pos, q


def out_dir(split: str, qset: str) -> Path:
    d = BLOCK_DIR / split / qset
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_pass(split, qset, name, src, idx, score):
    np.savez(out_dir(split, qset) / f"{name}_s{src}.npz", idx=idx, score=score.astype(np.float32))


def load_pass(split, qset, name, src):
    """Load one pass; a comma-separated qset returns the row-concatenation."""
    parts = []
    for qs in qset.split(","):
        z = np.load(out_dir(split, qs) / f"{name}_s{src}.npz")
        parts.append((z["idx"], z["score"]))
    if len(parts) == 1:
        return parts[0]
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def by_country(q: pd.DataFrame, c: pd.DataFrame):
    """Yield (label, query row positions, corpus row positions) per shared country label."""
    qg = q.groupby("country").indices
    cg = c.groupby("country").indices
    for label, qi in qg.items():
        yield label, qi, cg.get(label, np.array([], dtype=np.int64))


def _scatter(idx_all, score_all, qi, ci, idx, score):
    """Write per-country results (local corpus idx) back into global matrices."""
    valid = idx >= 0
    g = np.where(valid, ci[np.clip(idx, 0, None)] if len(ci) else -1, -1)
    idx_all[qi] = g
    score_all[qi] = np.where(valid, score, 0)


# ---------------------------------------------------------------------------
# passes
# ---------------------------------------------------------------------------
def _tfidf_word(corpus_text, query_text, max_df=MAX_DF_TOKEN):
    """IDF-weighted, L2-normalised token vectors; tokens in > max_df corpus records are
    zeroed AFTER normalisation (they still count in the norm but never generate
    candidates). Feature hashing (2^26 buckets) avoids a huge Python vocabulary when
    token pairs are indexed."""
    hv = HashingVectorizer(analyzer=str.split, n_features=2 ** 26, alternate_sign=False,
                           norm=None, dtype=np.float32)
    C = hv.transform(corpus_text)
    Q = hv.transform(query_text)
    df = np.bincount(C.indices, minlength=C.shape[1])
    tf = TfidfTransformer(sublinear_tf=True).fit(C)
    C, Q = tf.transform(C).astype(np.float32), tf.transform(Q).astype(np.float32)
    rare = df <= max_df
    C.data *= rare[C.indices]
    Q.data *= rare[Q.indices]
    C.eliminate_zeros()
    Q.eliminate_zeros()
    return Q.tocsr(), C.T.tocsr(), None, df


def with_pairs(texts, max_tokens=6):
    """Add unordered token pairs ('a~b') to each text. Business names are mostly built
    from individually common words; their pairs are rare, order-invariant keys."""
    out = []
    for t in texts:
        toks = list(dict.fromkeys(t.split()))[:max_tokens]
        pairs = [f"{a}~{b}" if a < b else f"{b}~{a}"
                 for i, a in enumerate(toks) for b in toks[i + 1:]]
        out.append(" ".join(toks + pairs))
    return out


def pass_tokens(q, pool, field, k=K_MAX, pairs=False):
    res = {}
    for src, c in pool.items():
        idx_all = np.full((len(q), k), -1, np.int32)
        sc_all = np.zeros((len(q), k), np.float32)
        for label, qi, ci in by_country(q, c):
            if len(ci) == 0:
                continue
            ct, qt = c[field].values[ci], q[field].values[qi]
            if pairs:
                ct, qt = with_pairs(ct), with_pairs(qt)
            Q, CT, _, _ = _tfidf_word(ct, qt)
            idx, sc = sparse_topk_cpu(Q, CT, k)
            _scatter(idx_all, sc_all, qi, ci, idx, sc)
        res[src] = (idx_all, sc_all)
    return res


def pass_trigram(q, pool, field="name_core", k=K_MAX):
    res = {}
    for src, c in pool.items():
        idx_all = np.full((len(q), k), -1, np.int32)
        sc_all = np.zeros((len(q), k), np.float32)
        for label, qi, ci in by_country(q, c):
            if len(ci) == 0:
                continue
            # max_df drops the ~80 ubiquitous trigrams (stop-trigrams): ~30% fewer
            # non-zeros -> proportionally faster SpMM, negligible ranking change.
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True,
                                  min_df=2, max_df=0.03, dtype=np.float32)
            C = vec.fit_transform(c[field].values[ci])
            Q = vec.transform(q[field].values[qi])
            idx, sc = sparse_topk_gpu(Q, C, k)
            _scatter(idx_all, sc_all, qi, ci, idx, sc)
        res[src] = (idx_all, sc_all)
    return res


def _rare_tokens(text_series, idf: dict, n=2, min_len=3, exclude=None):
    out = []
    for t in text_series:
        toks = {x for x in t.split() if len(x) >= min_len and not x.isdigit()}
        if exclude is not None:
            toks -= exclude
        out.append(sorted(toks, key=lambda x: -idf.get(x, 20.0))[:n])
    return out


def _keys(df, name_idf, addr_idf):
    """Return list of key-lists per record (strings)."""
    street = _rare_tokens(df["addr_norm"], addr_idf)
    name = _rare_tokens(df["name_core"], name_idf, min_len=2)
    keys = []
    for nums, st, pc, nm, city, lm in zip(df["numbers"], street, df["postcode"], name,
                                          df["city"], df["landmark"]):
        ks = []
        for num in nums.split()[:3]:
            num = num.lstrip("0") or "0"            # '091' == '91'
            ks += [f"1|{num}|{s}" for s in st]      # house number + rare street token
            ks += [f"4|{num}|{x}" for x in nm]      # house number + rare name token
        for p in pc.split()[:2]:
            ks += [f"2|{p}|{x}" for x in nm]
        if city and lm:
            ks += [f"3|{city}|{x}" for x in lm.split()[:3] if len(x) >= 3]
        keys.append(ks)
    return keys


def _key_frame(keys):
    lens = np.fromiter((len(k) for k in keys), np.int64, len(keys))
    rows = np.repeat(np.arange(len(keys)), lens)
    flat = pd.Series([x for ks in keys for x in ks], dtype=object)
    h = pd.util.hash_pandas_object(flat, index=False).values if len(flat) else np.array([], np.uint64)
    return pd.DataFrame({"h": h, "row": rows})


def pass_addr_key(q, pool, k=K_MAX):
    res = {}
    for src, c in pool.items():
        idx_all = np.full((len(q), k), -1, np.int32)
        sc_all = np.zeros((len(q), k), np.float32)
        for label, qi, ci in by_country(q, c):
            if len(ci) == 0:
                continue
            cc, qq = c.iloc[ci], q.iloc[qi]
            idfs = {}
            for f in ("name_core", "addr_norm"):
                v = TfidfVectorizer(analyzer=str.split, dtype=np.float32)
                v.fit(cc[f].values)
                idfs[f] = dict(zip(v.get_feature_names_out(), v.idf_))
            ck = _key_frame(_keys(cc, idfs["name_core"], idfs["addr_norm"]))
            qk = _key_frame(_keys(qq, idfs["name_core"], idfs["addr_norm"]))
            bsize = ck.groupby("h").size()
            bsize = bsize[bsize <= MAX_KEY_BLOCK]
            ck = ck[ck["h"].isin(bsize.index)]
            m = qk.merge(ck, on="h", suffixes=("_q", "_c"))
            if m.empty:
                continue
            m["w"] = 1.0 / np.log2(1.0 + bsize.reindex(m["h"]).values)
            agg = m.groupby(["row_q", "row_c"], sort=False)["w"].sum().reset_index()
            agg = agg.sort_values(["row_q", "w"], ascending=[True, False])
            agg["rank"] = agg.groupby("row_q").cumcount()
            agg = agg[agg["rank"] < k]
            idx = np.full((len(qi), k), -1, np.int32)
            sc = np.zeros((len(qi), k), np.float32)
            idx[agg["row_q"].values, agg["rank"].values] = agg["row_c"].values
            sc[agg["row_q"].values, agg["rank"].values] = agg["w"].values
            _scatter(idx_all, sc_all, qi, ci, idx, sc)
        res[src] = (idx_all, sc_all)
    return res


def pass_dense(split, qpos, q, pool, tag, k=K_MAX):
    from embed import load_emb
    Qe = load_emb(tag, split, 1)
    res = {}
    for src, c in pool.items():
        Ce = load_emb(tag, split, src)
        idx_all = np.full((len(q), k), -1, np.int32)
        sc_all = np.zeros((len(q), k), np.float32)
        for label, qi, ci in by_country(q, c):
            if len(ci) == 0:
                continue
            idx, sc = dense_topk(np.asarray(Qe[qpos[qi]]), np.asarray(Ce[ci]), k)
            _scatter(idx_all, sc_all, qi, ci, idx, sc)
        res[src] = (idx_all, sc_all)
    return res


def pass_transitive(split, qset, q, pool, tag, n_seed=3, n_nb=10, k=K_MAX):
    """For each query's top-`n_seed` dense candidates in one source, add the top-`n_nb`
    dense neighbours of that record in the other source. Score = s(q,x) * s(x,y)."""
    from embed import load_emb
    dense = {s: load_pass(split, qset, f"dense_{tag}", s) for s in SOURCES}
    res = {}
    for src_to in SOURCES:
        src_from = 5 - src_to  # 2 <-> 3
        seed_idx, seed_sc = dense[src_from][0][:, :n_seed], dense[src_from][1][:, :n_seed]
        uniq = np.unique(seed_idx[seed_idx >= 0])
        E_from, E_to = load_emb(tag, split, src_from), load_emb(tag, split, src_to)
        c_from, c_to = pool[src_from], pool[src_to]
        nb_idx = np.full((len(uniq), n_nb), -1, np.int32)
        nb_sc = np.zeros((len(uniq), n_nb), np.float32)
        u = c_from.iloc[uniq].reset_index(drop=True)
        for label, ui, ci in by_country(u, c_to):
            if len(ci) == 0:
                continue
            idx, sc = dense_topk(np.asarray(E_from[uniq[ui]]), np.asarray(E_to[ci]), n_nb)
            _scatter(nb_idx, nb_sc, ui, ci, idx, sc)
        pos = np.searchsorted(uniq, np.clip(seed_idx, 0, None))
        cand = nb_idx[pos]                                   # [nq, n_seed, n_nb]
        score = seed_sc[:, :, None] * nb_sc[pos]
        score[(seed_idx < 0)[:, :, None].repeat(n_nb, 2) | (cand < 0)] = 0
        cand = cand.reshape(len(q), -1)
        score = score.reshape(len(q), -1)
        # dedupe per row keeping max score, then top-k
        order = np.argsort(-score, axis=1, kind="stable")
        cand = np.take_along_axis(cand, order, 1)
        score = np.take_along_axis(score, order, 1)
        idx_all = np.full((len(q), k), -1, np.int32)
        sc_all = np.zeros((len(q), k), np.float32)
        for r in range(len(q)):
            seen, j = set(), 0
            for cnd, s in zip(cand[r], score[r]):
                if s <= 0 or cnd < 0 or cnd in seen:
                    continue
                seen.add(cnd)
                idx_all[r, j], sc_all[r, j] = cnd, s
                j += 1
                if j == k:
                    break
        res[src_to] = (idx_all, sc_all)
    return res


# ---------------------------------------------------------------------------
# evaluation / assembly
# ---------------------------------------------------------------------------
SHIFT = np.int64(1 << 23)  # > max rows in a source file


def true_pairs(split, qset, q):
    """{src: sorted int64 keys qrow*SHIFT + crow} for the ground-truth pairs of q."""
    gt = load_ground_truth()
    qrow = {e: i for i, e in enumerate(q["entity_id"])}
    out = {}
    for src in SOURCES:
        ids = load_norm(split, src, ["entity_id"])["entity_id"]
        crow = pd.Series(np.arange(len(ids)), index=ids.values)
        qs, cs = [], []
        pref = f"S{src}-"
        for e, ms in gt.items():
            if e in qrow:
                for m in ms:
                    if m.startswith(pref):
                        qs.append(qrow[e])
                        cs.append(m)
        keys = np.asarray(qs, np.int64) * SHIFT + crow.reindex(cs).values.astype(np.int64)
        out[src] = np.sort(keys)
    return out


NO_RANK = np.int16(32767)


def truth_ranks(idx: np.ndarray, truth_keys: np.ndarray) -> np.ndarray:
    """Rank (column) at which each true pair appears in a pass matrix; NO_RANK if absent."""
    nq, k = idx.shape
    m = idx >= 0
    keys = (np.repeat(np.arange(nq, dtype=np.int64), k).reshape(nq, k)[m] * SHIFT
            + idx[m].astype(np.int64))
    ranks = np.broadcast_to(np.arange(k, dtype=np.int16), (nq, k))[m]
    order = np.argsort(keys, kind="stable")
    keys, ranks = keys[order], ranks[order]
    pos = np.searchsorted(keys, truth_keys)
    pos_c = np.minimum(pos, len(keys) - 1)
    found = (pos < len(keys)) & (keys[pos_c] == truth_keys)
    out = np.full(len(truth_keys), NO_RANK, np.int16)
    out[found] = ranks[pos_c[found]]
    return out


def union_sizes(idx_list: list[np.ndarray]) -> np.ndarray:
    """Per-query number of distinct candidates in the union of several [nq, k] matrices."""
    a = np.sort(np.concatenate(idx_list, axis=1), axis=1)
    distinct = (np.diff(a, axis=1) != 0) & (a[:, 1:] >= 0)
    return distinct.sum(1) + (a[:, 0] >= 0)


class RecallEvaluator:
    """Loads each pass once, precomputes true-pair ranks; every (config) is then cheap."""

    def __init__(self, split, qset, passes):
        _, self.q = query_positions(split, qset)
        self.truth = true_pairs(split, qset, self.q)
        self.country = self.q["country"].values
        self.idx, self.rank = {}, {}
        for p in passes:
            for s in SOURCES:
                idx = load_pass(split, qset, p, s)[0]
                self.idx[p, s] = idx
                self.rank[p, s] = truth_ranks(idx, self.truth[s])
        self.tq = {s: (self.truth[s] // SHIFT).astype(np.int64) for s in SOURCES}

    def evaluate(self, config: dict[str, int], sizes=True):
        n_true_q = np.zeros(len(self.q))
        n_hit_q = np.zeros(len(self.q))
        n_cand_q = np.zeros(len(self.q))
        rows, hits = [], 0
        for s in SOURCES:
            hit = np.zeros(len(self.truth[s]), bool)
            for p, k in config.items():
                hit |= self.rank[p, s] < k
            tq = self.tq[s]
            np.add.at(n_true_q, tq, 1)
            np.add.at(n_hit_q, tq[hit], 1)
            hits += hit.sum()
            for lab in np.unique(self.country):
                m = self.country[tq] == lab
                rows.append((lab, f"S{s}", hit[m].mean() if m.any() else np.nan))
            if sizes:
                n_cand_q += union_sizes([self.idx[p, s][:, :k] for p, k in config.items()])
        nons = n_true_q > 0
        r = np.where(nons, n_hit_q / np.maximum(n_true_q, 1), 1.0)
        f = np.where(nons, np.where(r > 0, 1.25 * r / (0.25 + r), 0.0), 1.0)
        summary = {
            "pair_recall": float(hits / sum(len(t) for t in self.truth.values())),
            "entity_full_coverage": float((n_hit_q[nons] == n_true_q[nons]).mean()),
            "mean_candidates_per_s1": float(n_cand_q.mean()) if sizes else float("nan"),
            "oracle_f05": float(f.mean()),
        }
        tab = pd.DataFrame(rows, columns=["country", "src", "recall"])
        return summary, tab.pivot(index="country", columns="src", values="recall")


def _fmt(s, piv=None):
    out = (f"recall={s['pair_recall']:.4f} full={s['entity_full_coverage']:.4f} "
           f"cands/S1={s['mean_candidates_per_s1']:.1f} oracleF={s['oracle_f05']:.4f}")
    if piv is not None:
        out += " | " + " ".join(f"{c}/{sv}={piv.loc[c, sv]:.4f}" for c in piv.index for sv in piv.columns)
    return out


def cmd_eval(args):
    t = time.time()
    ev = RecallEvaluator(args.split, args.qset, args.passes)
    print(f"loaded {len(args.passes)} passes for {len(ev.q):,} {args.qset} queries "
          f"in {time.time() - t:.0f}s")
    report = {}
    print(f"\n=== recall vs K, each pass alone (K per source) ===")
    for p in args.passes:
        for k in (args.ks if not args.configs_only else []):
            s, piv = ev.evaluate({p: k})
            report[f"{p}@{k}"] = s
            print(f"{p:<16} K={k:<4} {_fmt(s)}")
    for k in (args.ks if not args.configs_only else []):
        print(f"\n=== cumulative union, every pass at K={k} (marginal gain per added pass) ===")
        cfg = {}
        for p in args.passes:
            cfg[p] = k
            s, piv = ev.evaluate(cfg)
            report[f"union@{k}+{p}"] = s
            print(f"+{p:<16} {_fmt(s, piv)}")
    if args.config != "{}":
        cfgs = json.loads(args.config)
        for i, cfg in enumerate(cfgs if isinstance(cfgs, list) else [cfgs]):
            s, piv = ev.evaluate(cfg)
            report[f"config{i}"] = {"config": cfg, **s}
            print(f"\n=== config {cfg} ===\n{_fmt(s, piv)}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))


PASS_COLS = {  # corpus columns each pass needs (loaded one source at a time to save RAM)
    "name_tok": ["name_core"], "name_pair": ["name_core"], "addr_tok": ["addr_norm"], "name_tri": ["name_core"],
    "addr_key": ["name_core", "addr_norm", "numbers", "postcode", "landmark", "city"],
}


def cmd_run(args):
    t0 = time.time()
    qpos, q = query_positions(args.split, args.qset)
    print(f"loaded {len(q):,} queries in {time.time() - t0:.0f}s", flush=True)
    timings = {}
    for p in args.passes:
        t = time.time()
        if p.startswith("trans_"):
            pool = {s: load_norm(args.split, s, ["country"]) for s in SOURCES}
            res = pass_transitive(args.split, args.qset, q, pool, p[len("trans_"):], k=args.k)
        else:
            res = {}
            for s in SOURCES:
                cols = ["country"] + PASS_COLS.get(p, [])
                pool = {s: load_norm(args.split, s, cols)}
                if p == "name_tok":
                    r = pass_tokens(q, pool, "name_core")
                elif p == "name_pair":
                    r = pass_tokens(q, pool, "name_core", k=args.k, pairs=True)
                elif p == "addr_tok":
                    r = pass_tokens(q, pool, "addr_norm", k=args.k)
                elif p == "name_tri":
                    r = pass_trigram(q, pool)
                elif p == "addr_key":
                    r = pass_addr_key(q, pool, k=args.k)
                elif p.startswith("dense_"):
                    r = pass_dense(args.split, qpos, q, pool, p[len("dense_"):], k=args.k)
                else:
                    raise ValueError(p)
                res.update(r)
                del pool
        labels = q["qset"].values
        for src, (idx, sc) in res.items():
            for name in args.qset.split(","):
                m = labels == name
                save_pass(args.split, name, p, src, idx[m], sc[m])
        timings[p] = time.time() - t
        print(f"pass {p}: {timings[p]:.0f}s", flush=True)
    tpath = out_dir(args.split, args.qset.replace(",", "+")) / "timings.json"
    old = json.loads(tpath.read_text()) if tpath.exists() else {}
    old.update({p: {"seconds": s, "n_queries": len(q)} for p, s in timings.items()})
    tpath.write_text(json.dumps(old, indent=1))


def build_candidate_chunk(passes, config, src, r0, r1, truth_keys=None) -> pd.DataFrame:
    """Candidate rows for query rows [r0, r1) of one source, with per-pass rank/score.

    Columns: q_row (row in the qset order), s1_row (row in {split}_s1.parquet is
    added by the caller), src, c_row (row in {split}_s{src}.parquet),
    <pass>_rank (int16, -1 = not retrieved by that pass within its K), <pass>_score.
    """
    keys_all, per_pass = [], {}
    for p, k in config.items():
        idx, sc = passes[p]
        sub_i, sub_s = idx[r0:r1, :k], sc[r0:r1, :k]
        m = sub_i >= 0
        rows = np.broadcast_to(np.arange(r0, r1, dtype=np.int64)[:, None], sub_i.shape)[m]
        pk = rows * SHIFT + sub_i[m].astype(np.int64)
        rk = np.broadcast_to(np.arange(sub_i.shape[1], dtype=np.int16), sub_i.shape)[m]
        pk, first = np.unique(pk, return_index=True)   # first occurrence = best rank
        per_pass[p] = (pk, rk[first], sub_s[m][first])
        keys_all.append(pk)
    keys = np.unique(np.concatenate(keys_all))
    cols = {"q_row": (keys // SHIFT).astype(np.int32), "src": np.full(len(keys), src, np.int8),
            "c_row": (keys % SHIFT).astype(np.int32)}
    for p, (pk, rk, s) in per_pass.items():
        loc = np.searchsorted(keys, pk)
        r = np.full(len(keys), -1, np.int16)
        v = np.full(len(keys), np.nan, np.float32)
        r[loc], v[loc] = rk, s
        cols[f"{p}_rank"], cols[f"{p}_score"] = r, v
    if truth_keys is not None:
        pos = np.minimum(np.searchsorted(truth_keys, keys), len(truth_keys) - 1)
        cols["label"] = (truth_keys[pos] == keys).astype(np.int8)
    return pd.DataFrame(cols)


def cmd_build(args):
    """Write the candidate table for a qset as parquet (integer row ids; see README)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = json.loads(args.config)
    t = time.time()
    qpos, q = query_positions(args.split, args.qset)
    truth = true_pairs(args.split, args.qset, q) if args.split == "train" else None
    path = out_dir(args.split, args.qset) / "candidates.parquet"
    writer, n = None, 0
    for src in SOURCES:
        passes = {p: load_pass(args.split, args.qset, p, src) for p in cfg}
        for r0 in range(0, len(q), 100_000):
            df = build_candidate_chunk(passes, cfg, src, r0, min(r0 + 100_000, len(q)),
                                       truth[src] if truth is not None else None)
            df.insert(1, "s1_row", qpos[df["q_row"].values].astype(np.int32))
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            writer = writer or pq.ParquetWriter(path, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n += len(df)
        del passes
    writer.close()
    (out_dir(args.split, args.qset) / "candidates_config.json").write_text(json.dumps(cfg))
    print(f"wrote {path}: {n:,} pairs ({n / len(q):.1f} per S1) in {time.time() - t:.0f}s")


def miss_reason(r) -> str:
    """Heuristic label for why a true pair was not retrieved."""
    from rapidfuzz import fuzz
    n_sim = fuzz.token_set_ratio(r.s1_name_core, r.m_name_core)
    a_sim = fuzz.token_sort_ratio(r.s1_addr, r.m_addr) if r.m_addr else 0
    m_tokens = len(r.m_addr.split())
    has_num = any(ch.isdigit() for ch in r.m_addr)
    if r.m_nonlatin and n_sim < 60:
        return "native-script name, transliteration too far"
    if not r.m_addr:
        return ("empty address + near-identical name shared by many records" if n_sim >= 80
                else "empty address + name variant")
    if n_sim < 50:
        return ("different name (brand/DBA-like), address similar" if a_sim >= 60
                else "different name and weak address")
    if m_tokens <= 4 or not has_num:
        return "partial address (city/state only) + generic name"
    if a_sim < 50:
        return "similar name, address reordered/rewritten"
    return "similar name and address, crowded out by near-duplicates"


def cmd_missed(args):
    cfg = json.loads(args.config)
    ev = RecallEvaluator(args.split, args.qset, list(cfg))
    s1 = load_norm(args.split, 1, ["entity_id", "raw_name", "raw_address", "name_core", "addr_norm"])
    rows = []
    for s in SOURCES:
        hit = np.zeros(len(ev.truth[s]), bool)
        for p, k in cfg.items():
            hit |= ev.rank[p, s] < k
        miss = ev.truth[s][~hit]
        c = load_norm(args.split, s, ["entity_id", "raw_name", "raw_address", "name_core",
                                      "addr_norm", "country"])
        nl = load_norm(args.split, s, ["name_nonlatin"])["name_nonlatin"].values
        for key in miss:
            qi, ci = int(key // SHIFT), int(key % SHIFT)
            e = ev.q.iloc[qi]
            rows.append((e["entity_id"], e["country"], c["entity_id"].iat[ci], qi, ci, s,
                         bool(nl[ci])))
        del c
    df = pd.DataFrame(rows, columns=["s1_id", "country", "m_id", "q_row", "c_row", "src", "m_nonlatin"])
    s1i = s1.set_index("entity_id")
    df["s1_name"] = s1i.loc[df["s1_id"], "raw_name"].values
    df["s1_address"] = s1i.loc[df["s1_id"], "raw_address"].values
    df["s1_name_core"] = s1i.loc[df["s1_id"], "name_core"].values
    df["s1_addr"] = s1i.loc[df["s1_id"], "addr_norm"].values
    for s in SOURCES:
        c = load_norm(args.split, s, ["raw_name", "raw_address", "name_core", "addr_norm"])
        m = (df["src"] == s).values
        for col, src_col in [("m_name", "raw_name"), ("m_address", "raw_address"),
                             ("m_name_core", "name_core"), ("m_addr", "addr_norm")]:
            if col not in df:
                df[col] = ""
            df.loc[m, col] = c[src_col].values[df.loc[m, "c_row"].values]
        del c
    df["reason"] = [miss_reason(r) for r in df.itertuples()]
    print(f"missed true pairs: {len(df):,}")
    print(df["reason"].value_counts(normalize=True).mul(100).round(1).to_string())
    print(pd.crosstab(df["reason"], df["country"], normalize="columns").mul(100).round(1).to_string())
    path = out_dir(args.split, args.qset) / "missed_pairs.parquet"
    df.to_parquet(path, index=False)
    ex = df.sample(frac=1.0, random_state=42).groupby("reason").head(args.n_examples)
    for reason, g in ex.groupby("reason"):
        print(f"\n### {reason}")
        for r in g.itertuples():
            print(f"  [{r.country}] {r.s1_id}: {r.s1_name!r} | {r.s1_address!r}")
            print(f"  {'':>{len(r.country) + 2}} {r.m_id}: {r.m_name!r} | {r.m_address!r}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "eval", "build", "missed"):
        s = sub.add_parser(name)
        s.add_argument("--split", default="train")
        s.add_argument("--qset", default="val")
        s.add_argument("--passes", nargs="+", default=[])
        s.add_argument("--ks", nargs="+", type=int, default=[20, 40, 60, 100])
        s.add_argument("--report", default=None)
        s.add_argument("--config", default="{}")
        s.add_argument("--n-examples", type=int, default=3)
        s.add_argument("--configs-only", action="store_true")
        s.add_argument("--k", type=int, default=K_MAX, help="top-K kept by dense passes")
    args = ap.parse_args()
    {"run": cmd_run, "eval": cmd_eval, "build": cmd_build, "missed": cmd_missed}[args.cmd](args)


if __name__ == "__main__":
    main()
