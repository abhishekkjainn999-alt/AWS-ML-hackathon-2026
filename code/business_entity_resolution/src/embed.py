"""Encode records with a sentence-embedding model; cache float16 .npy per file.

Text = "query: <name> | <address>" (lower-cased, original script kept so the
multilingual model sees native-script names). e5 models expect the "query: "
prefix for symmetric similarity tasks.

Usage: python src/embed.py --model intfloat/multilingual-e5-small --tag base \
           [--splits train] [--sources 1 2 3]
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd
import torch

from io_utils import ARTIFACTS_DIR, load_source

EMB_DIR = ARTIFACTS_DIR / "emb"
MAX_SEQ = 64
_LEAD = re.compile(r"^[\W_]+")
_WS = re.compile(r"\s+")


def embed_text(name: str, address: str) -> str:
    name = _LEAD.sub("", name.strip())
    return _WS.sub(" ", f"query: {name} | {address}".lower()).strip()


def texts_for(df: pd.DataFrame) -> list[str]:
    return [embed_text(n, a) for n, a in zip(df["raw_name"], df["raw_address"])]


def load_model(model: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device="cuda")
    m.max_seq_length = MAX_SEQ
    m.half()
    pool = m[1]
    assert type(pool).__name__ == "Pooling" and pool.pooling_mode == "mean", \
        "fast encode() assumes Transformer + mean pooling"
    return m


@torch.no_grad()
def encode(model, texts: list[str], batch_size=1024) -> np.ndarray:
    """Equivalent to model.encode(normalize_embeddings=True) for Transformer + mean
    Pooling (+ Normalize) models, but ~2x faster: the whole chunk is tokenised in one
    parallel (Rust) call and batches are length-sorted to minimise padding."""
    tok = model.tokenizer
    ids = tok(texts, truncation=True, max_length=MAX_SEQ, padding=False,
              return_attention_mask=False, return_token_type_ids=False)["input_ids"]
    lens = np.fromiter((len(x) for x in ids), np.int64, len(ids))
    order = np.argsort(-lens, kind="stable")
    tm = model[0].auto_model
    dev = next(tm.parameters()).device
    out = np.empty((len(texts), model.get_embedding_dimension()), np.float16)
    pad = tok.pad_token_id
    for b in range(0, len(order), batch_size):
        sel = order[b:b + batch_size]
        L = int(lens[sel[0]])
        arr = np.full((len(sel), L), pad, np.int64)
        for j, i in enumerate(sel):
            arr[j, :lens[i]] = ids[i]
        x = torch.from_numpy(arr).to(dev, non_blocking=True)
        m = torch.from_numpy(np.arange(L)[None, :] < lens[sel][:, None]).to(dev)
        h = tm(input_ids=x, attention_mask=m.long()).last_hidden_state
        mf = m.unsqueeze(-1).to(h.dtype)
        e = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        out[sel] = e.half().cpu().numpy()
    return out


def emb_path(tag: str, split: str, src: int):
    return EMB_DIR / tag / f"{split}_s{src}.npy"


def load_emb(tag: str, split: str, src: int, mmap=True) -> np.ndarray:
    return np.load(emb_path(tag, split, src), mmap_mode="r" if mmap else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--sources", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--chunk", type=int, default=500_000)
    args = ap.parse_args()

    model = load_model(args.model)
    (EMB_DIR / args.tag).mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        for src in args.sources:
            t = time.time()
            # raw text straight from the TSV (same row order as the normalised parquet)
            df = load_source(split, src, usecols=["business_name", "business_address"]).rename(
                columns={"business_name": "raw_name", "business_address": "raw_address"})
            out = np.lib.format.open_memmap(emb_path(args.tag, split, src),
                                            mode="w+", dtype=np.float16,
                                            shape=(len(df), model.get_embedding_dimension()))
            for i in range(0, len(df), args.chunk):
                out[i:i + args.chunk] = encode(model, texts_for(df.iloc[i:i + args.chunk]))
            out.flush()
            del out
            dt = time.time() - t
            print(f"[{args.tag}] {split} S{src}: {len(df):,} rows in {dt:.0f}s "
                  f"({len(df) / dt:.0f}/s)", flush=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
