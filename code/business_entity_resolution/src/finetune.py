"""Fine-tune the multilingual embedding model on TRAIN-split matched pairs.

Data: S1 entities of query set `ft` (TRAIN split only; disjoint from val and from the
Stage-3 training sample `s3train`).
  anchor   = S1 record text
  positive = a matched S2/S3 record text
  negative = a hard negative mined from that entity's blocking candidates
             (random pick among the top-ranked non-matches of the base dense pass
             and the rare-name-token pass)
Plus S2<->S3 positive pairs of the same entity (helps the transitive pass).

Loss: CachedMultipleNegativesRankingLoss (large effective batch on 8 GB VRAM),
in-batch negatives + the mined hard negative; NO_DUPLICATES batch sampler.

Usage: python src/finetune.py [--epochs 1] [--batch 1024]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from blocking import SOURCES, load_norm, load_pass, query_positions
from embed import MAX_SEQ, embed_text
from io_utils import ARTIFACTS_DIR, load_ground_truth

MODEL_DIR = ARTIFACTS_DIR / "models"


def build_triplets(split="train", qset="ft", n_hard=5, seed=42, s23_frac=0.3):
    rng = np.random.default_rng(seed)
    _, q = query_positions(split, qset)
    s1 = load_norm(split, 1, ["entity_id", "raw_name", "raw_address"])
    s1 = s1.set_index("entity_id").loc[q["entity_id"]]
    q_text = [embed_text(n, a) for n, a in zip(s1["raw_name"], s1["raw_address"])]
    gt = load_ground_truth()

    texts, row_of = {}, {}
    for src in SOURCES:
        d = load_norm(split, src, ["entity_id", "raw_name", "raw_address"])
        texts[src] = d
        row_of[src] = pd.Series(np.arange(len(d)), index=d["entity_id"].values)

    # hard negatives per (query, source): top non-matches of dense_base and name_tri
    hard = {}
    for src in SOURCES:
        cands = [load_pass(split, qset, p, src)[0] for p in ("dense_base", "name_tok")]
        hard[src] = cands

    anchors, poss, negs = [], [], []
    for qi, e in enumerate(q["entity_id"]):
        ms = gt.get(e, set())
        if not ms:
            continue
        pos_rows = {s: [int(row_of[s][m]) for m in ms if m.startswith(f"S{s}-")] for s in SOURCES}
        neg_pool = {}
        for s in SOURCES:
            pr = set(pos_rows[s])
            pool = []
            for cand in hard[s]:
                pool += [c for c in cand[qi, :n_hard * 2] if c >= 0 and c not in pr][:n_hard]
            neg_pool[s] = pool
        for s in SOURCES:
            for r in pos_rows[s]:
                np_ = neg_pool[s] or neg_pool[5 - s]
                if not np_:
                    continue
                ns = s if neg_pool[s] else 5 - s
                anchors.append(q_text[qi])
                poss.append((s, r))
                negs.append((ns, int(rng.choice(np_))))
        # S2<->S3 pairs of the same entity
        if pos_rows[2] and pos_rows[3] and neg_pool[3] and rng.random() < s23_frac:
            a, b = int(rng.choice(pos_rows[2])), int(rng.choice(pos_rows[3]))
            anchors.append(("S2", a))
            poss.append((3, b))
            negs.append((3, int(rng.choice(neg_pool[3]))))

    # batch text lookup (anchors are S1 texts or ("S2", row) for S2<->S3 pairs)
    cache = {}
    for src in SOURCES:
        rows = sorted({r for s, r in poss + negs if s == src} |
                      ({r for a in anchors if isinstance(a, tuple) for r in [a[1]]} if src == 2 else set()))
        d = texts[src].iloc[rows]
        cache[src] = dict(zip(rows, (embed_text(n, a) for n, a in zip(d["raw_name"], d["raw_address"]))))
    A = [a if isinstance(a, str) else cache[2][a[1]] for a in anchors]
    P = [cache[s][r] for s, r in poss]
    N = [cache[s][r] for s, r in negs]
    return pd.DataFrame({"anchor": A, "positive": P, "negative": N}).sample(frac=1.0, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default=str(MODEL_DIR / "e5s-ft"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--mini-batch", type=int, default=256)
    ap.add_argument("--train-word-embeddings", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-train", type=int, default=0, help="cap #triplets (0 = all)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    t0 = time.time()
    df = build_triplets()
    if args.max_train:
        df = df.iloc[:args.max_train]
    print(f"triplets: {len(df):,} built in {time.time() - t0:.0f}s", flush=True)
    print(df.head(5).to_string(), flush=True)
    ds = Dataset.from_pandas(df, preserve_index=False)

    model = SentenceTransformer(args.base, device="cuda")
    model.max_seq_length = MAX_SEQ
    if not args.train_word_embeddings:
        # 96M of the 118M params are the 250k-token multilingual vocabulary; freezing it
        # saves ~1.5 GB VRAM (optimizer state + dense grads) and most of the step time.
        model[0].auto_model.embeddings.word_embeddings.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M", flush=True)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(MODEL_DIR / "_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        bf16=torch.cuda.is_bf16_supported(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=42,
    )
    t1 = time.time()
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    model.save(args.out)
    print(f"saved {args.out}; training {time.time() - t1:.0f}s, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
