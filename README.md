# Business Entity Resolution — ML Challenge 2026

Team **Sentinels**. For every Source 1 (reference) business record, find all matching
Source 2 / Source 3 records across noisy, multilingual name/address data (US, India, and
an unseen France test set).

**Final result:** public leaderboard macro F0.5 **0.980947** (model v5, threshold 0.875).

## Approach

1. **Normalisation** — country-agnostic name/address cleaning (legal forms, house numbers, postcodes, transliteration via `anyascii`).
2. **Candidate generation (blocking)** — fine-tuned `multilingual-e5-small` dense retrieval + transitive, address-key, name-pair and rare-token passes. VAL pair recall 0.9987 at ~84 candidates per entity.
3. **Pre-ranker** — small LightGBM keeps the top 20 candidates.
4. **Matcher** — LightGBM over ~78 string, address, embedding, context and cross-encoder features.
5. **Decision** — one-owner rule + global threshold.

No external data or APIs; all models are MIT/Apache licensed (see `code/business_entity_resolution/docs/licenses.md`).

Full details, commands and per-version results: [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).

## Data

The competition dataset is not included. Place it at `dataset/train/` and `dataset/test/`
(or set `BER_DATA_DIR`). Outputs go to `output/` (or `BER_OUTPUT_DIR`).

## Quick start

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_all.py
```
