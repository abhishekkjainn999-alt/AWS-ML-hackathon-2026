# Business Entity Resolution

Pipeline for the ML Challenge 2026 Business Entity Resolution task: for every
Source 1 (reference) entity, find all matching Source 2 / Source 3 records.

Only the provided data is used (no external APIs, geocoding or lookups). Country
is treated as an open set of string labels and is used **only** to restrict
candidates to records with the same label. Model / package licences:
[docs/licenses.md](docs/licenses.md).

## Layout

```
student_resource/
├── dataset/{train,test}/            # provided data (not modified)
├── output/                          # matching_results.tsv, candidate_pairs.tsv
└── code/business_entity_resolution/
    ├── src/
    │   ├── io_utils.py              # paths + TSV loaders/writers (sep="\t", QUOTE_NONE)
    │   ├── eda.py                   # exploratory analysis report
    │   ├── evaluate.py              # macro F0.5 scorer + blocking recall
    │   ├── split.py                 # 80/20 val split + query sets (val / ft / s3train)
    │   ├── normalize.py             # country-agnostic name/address normalisation -> parquet
    │   ├── embed.py                 # sentence embeddings (fast batched encoder) -> .npy
    │   ├── search.py                # GPU/CPU chunked top-K search primitives
    │   ├── blocking.py              # candidate-generation passes, eval, build, missed-pair analysis
    │   ├── finetune.py              # MNRL fine-tuning of the embedding model (TRAIN split only)
    │   └── check_candidates.py      # re-checks a candidates.parquet with evaluate.blocking_recall
    ├── docs/licenses.md
    ├── artifacts/                   # generated (norm/, emb/, blocking/, models/, logs)
    ├── README.md
    └── requirements.txt
```

Paths default to `student_resource/dataset` and `student_resource/output`;
override with `BER_DATA_DIR` / `BER_OUTPUT_DIR`.

## Setup

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r code/business_entity_resolution/requirements.txt
```

Tested on Windows 11, Python 3.12, RTX 5050 Laptop (8 GB VRAM, Blackwell sm_120),
16 GB RAM. Run jobs one at a time: two big jobs together exceed 16 GB and thrash.

## Run (from `student_resource/`)

One command runs everything end to end (each step is checkpointed and skipped if
its output exists):

```bash
python code/business_entity_resolution/src/run_all.py
```

Pipeline: split → normalise → base embeddings → hard-negative passes → fine-tune →
fine-tuned embeddings → blocking passes (val, s3train, and all train S1 for reverse
context) → candidate tables → **pre-ranker** → features → LightGBM matcher → decision
rule tuned on VAL → test inference (`run_test.py`: embed test, block within test
S2/S3, pre-rank, features, predict, decide, write `output/*.tsv`, run
`utils/validate_submission.py`).

### Final model: v5 (v3 retrained on s3train + VAL-50k + 200k extra VAL entities)

The submitted model. Same 78 v3 features and hyperparameters; training data =
s3train-100k + VAL-50k + **200k further VAL entities** (none used for the bi-encoder or
cross-encoder). For the extra entities `v5.py prep` runs the v2 pre-ranker on their saved
Stage-2 candidates, the v2 features, out-of-sample v2 probabilities and cross-encoder scoring
on the same band ([0.005, 0.995] or pre-rank top-3). 7,000,026 pairs; fixed tree count
471 × 7.0M / 2.0M = **1,649 trees**, 1 seed, no early stopping. Test: saved v3 test features
and cross-encoder scores, one-owner rule, **final threshold t = 0.875** (the same
probabilities also give the t = 0.85 file).

```bash
set BER_VERSION=v5
set BER_BASE_VERSION=v2
set BER_INHERIT=norm,blocking,feat_cache
set BER_OUTPUT_DIR=output/variants/v5
set BER_THRESHOLDS=0.85,0.875
python code/business_entity_resolution/src/v5.py prep                 # ~18 min
python code/business_entity_resolution/src/v5.py train --seeds 1      # ~18 min
python code/business_entity_resolution/src/v5.py test                 # ~19 min -> matching_t085.tsv, matching_t0875.tsv
copy output\variants\v5\matching_t0875.tsv output\matching_results.tsv
copy output\variants\v3\candidate_pairs.tsv output\candidate_pairs.tsv
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`v5.py` options (environment): `BER_THRESHOLDS` (comma list; files named `matching_t085`,
`matching_t0875`, …), and for extending the training data `BER_EXTRA_QSET`, `BER_EXTRA_N`,
`BER_EXTRA_EXCLUDE` (qrows `.npy` of entities already used) and `BER_EXTRA_TRAIN` (extra
feature folders). Defaults reproduce v5.

Public leaderboard: v5 at t = 0.875 was selected as final on the public leaderboard over
v5 at t = 0.85 (**0.980947**); v3full 0.979982. `run_all.py` runs v5 as its final step.

### Model v3full (v3 retrained on s3train + VAL-50k; superseded by v5)

After v3 was validated, its LightGBM was retrained with the same 78 features and
hyperparameters on s3train-100k **plus** the VAL-50k entities (3.0M pairs), with a fixed
tree count = v3's best iteration × (new rows / old rows) = 471 × 1.5 = 707 (no held-out set
remains, so no early stopping). Test is predicted from the saved v3 test features and
cross-encoder scores; one-owner rule + t = 0.85.

```bash
set BER_VERSION=v3full
set BER_BASE_VERSION=v2
set BER_INHERIT=norm,blocking,prerank,feat_cache
set BER_OUTPUT_DIR=output/variants/v3full
python code/business_entity_resolution/src/v3full.py      # ~15 min
copy output\variants\v3full\matching_t085.tsv output\matching_results.tsv
copy output\variants\v3\candidate_pairs.tsv  output\candidate_pairs.tsv
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Public leaderboard: 0.979982 (v3full, t = 0.85) vs 0.97949 (v3). Kept for reference
(`src/v3full.py`); the final step of `run_all.py` is now v5.

### Model v3 (v2 + cross-encoder judge)

`src/run_all.py` runs v1 → v2 → v3 end to end (each step checkpointed). The v3 part:

```bash
set BER_VERSION=v3
set BER_BASE_VERSION=v2
set BER_INHERIT=norm,blocking,prerank,feat_cache
set BER_OUTPUT_DIR=output/variants/v3
S=code/business_entity_resolution/src
python $S/xenc.py oof                                  # OOF v2 probs on s3train (scoring band)
python $S/xenc.py build                                # CE training pairs (other 100k s3train entities, 30% augmented)
python $S/xenc.py train                                # cross-encoder, 1 epoch bf16 (~27 min)
python $S/xenc.py score   --split train --qset s3train
python $S/xenc.py score   --split train --qset val
python $S/xenc.py augment --split train --qset s3train
python $S/xenc.py augment --split train --qset val
python $S/train.py && python $S/decide.py              # v3 LightGBM + VAL-50k tuning
python $S/xenc.py test                                 # test CE scoring (~65 min), features, predict, t=0.75/0.80/0.85 files
copy output\variants\v3\matching_t085.tsv output\matching_results.tsv
copy output\variants\v3\candidate_pairs.tsv output\candidate_pairs.tsv
python utils/validate_submission.py --check-ids ...
```

VAL-50k F0.5: v2 0.9828 → **v3 0.9869–0.9871**; public leaderboard **0.97949** (v3, t = 0.85).

### Model v2

v2 = v1 + extended shared normalisation word lists (company-form words incl.
Cie/Ets/Sté, generic suffix words, articles; name tokens in > 1% of the split's S1
records dropped from the core name; bis/ter/`16a`/`1T` house-number suffixes, `st`→saint
before a place name, All./Crs/Qu, cedex dropped) + house-number features (numeric
difference, suffixes, street-key agreement, "same street, different number"). The
fine-tuned encoder, embeddings and dense/transitive passes are reused from v1.

```bash
python code/business_entity_resolution/src/run_all.py          # v1 end to end (provides embeddings, dense passes)
set BER_VERSION=v2
python code/business_entity_resolution/src/normalize.py --workers 16
# hard-link v1 dense_ft/trans_ft pass files into artifacts/blocking_v2/<split>/<qset>/ (see run_v2.py docstring)
python code/business_entity_resolution/src/run_v2.py            # -> output/variants/v2/
python code/business_entity_resolution/src/package.py --team TEAMNAME
```

VAL-50k F0.5: v1 0.9806 → **v2 0.9828** (threshold 0.80, also the best public-leaderboard
threshold). All artifact folders are versioned via `BER_VERSION` (`norm_v2`,
`blocking_v2`, …), so v1 files stay intact.

### Stage 3 (matching)

- **Pre-ranker = last blocking stage** (`prerank.py`): small LightGBM on blocking
  ranks/scores + fine-tuned dense cosine; keeps the top 20 candidates per S1 (plus any
  with p > 0.05). VAL-50k: keeps 99.82% of the true pairs present in the candidates
  (0.9970 of all true pairs). `output/candidate_pairs.tsv` is exactly this reduced set.
  Out-of-fold (2-fold) pre-ranker probabilities are a feature of the final model.
- **Features** (`features.py`, 63, country is not a feature): name similarities
  (ratio / token set / token sort / partial / Jaro-Winkler on full and core name,
  char-3gram TF-IDF cosine, first token, acronym, web stem, core-token Jaccard, lengths,
  script flags), address (token set/sort, TF-IDF cosine, house number / postcode /
  city / learned state equal-conflict-missing, number Jaccard, landmark overlap, empty /
  length), dense cosine, blocking ranks/scores, per-S1 context (ranks and gaps),
  reverse context (per candidate record across all S1 of the split), cluster
  coherence with the S1's top candidate in the other source. ~8 µs per pair.
- **Matcher** (`train.py`): LightGBM, 100k s3train entities (TRAIN split),
  early stopping on 10% held-out entities.
- **Decision** (`decide.py`): one-owner rule, then global threshold vs per-entity
  expected-F0.5 compared on a 50k-entity VAL subset; threshold 0.75 won.

### Individual steps

```bash
S=code/business_entity_resolution/src
python $S/eda.py > code/business_entity_resolution/artifacts/eda_report.txt
python $S/split.py                       # val_split.tsv + query_sets.tsv
python $S/normalize.py --workers 16      # artifacts/norm/*.parquet (~15 min)
python $S/embed.py --tag base --splits train
python $S/blocking.py run --qset ft --passes dense_base name_tok   # hard negatives for fine-tuning
python $S/finetune.py                    # artifacts/models/e5s-ft (~75 min)
python $S/embed.py --model code/business_entity_resolution/artifacts/models/e5s-ft --tag ft --splits train
python $S/blocking.py run --qset val,s3train --passes dense_ft trans_ft addr_key name_pair addr_tok
CFG='{"dense_ft":20,"trans_ft":10,"addr_key":10,"name_pair":10,"addr_tok":10}'   # artifacts/blocking/final_config.json
python $S/blocking.py eval  --qset val --passes dense_ft trans_ft addr_key name_pair addr_tok
python $S/blocking.py build --qset val     --config "$CFG"
python $S/blocking.py build --qset s3train --config "$CFG"
python $S/blocking.py missed --qset val    --config "$CFG"
python $S/check_candidates.py --qset val   # independent re-check with evaluate.blocking_recall
```

Final blocking (VAL, 441k S1): pair recall **0.9987**, entities fully covered 99.55%,
84 candidates / S1, F0.5 ceiling 0.9996.

### Query sets (`artifacts/query_sets.tsv`)

| qset | from | size | used for |
|---|---|---|---|
| `val` | validation split | 441,365 | honest scoring only; never trained on |
| `ft` | TRAIN split | 300,000 | fine-tuning the embedding model (+ hard-negative mining) |
| `s3train` | TRAIN split, disjoint from `ft` | 200,000 | training the Stage-3 matcher (embedding scores are out-of-sample, as on val/test) |

Every blocking pass searches the **whole** S2/S3 pool of the split (all ~10.3M train
records), so distractor density matches the test setting.

### Candidate table (`artifacts/blocking/train/<qset>/candidates.parquet`)

One row per (S1, candidate) pair: `q_row` (row in the qset), `s1_row` (row in
`norm/train_s1.parquet`), `src` (2/3), `c_row` (row in `norm/train_s{src}.parquet`),
per pass `<pass>_rank` (int16, -1 = not retrieved) and `<pass>_score`, and `label`
(train split only).

### Metric

Per-S1-entity F0.5 (`1.25·P·R / (0.25·P + R)`), macro-averaged over all S1
entities. Singletons score 1.0 for an empty prediction and 0.0 otherwise.

## Status

- [x] Project structure, EDA, evaluator, validation split
- [x] Normalisation + candidate generation (Stage 2)
- [x] Matching model (Stage 3): VAL-50k F0.5 0.9806
- [x] Test inference + submission files + `utils/validate_submission.py`
