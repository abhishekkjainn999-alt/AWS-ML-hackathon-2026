# Licences

The challenge rules say every model must be MIT or Apache-2.0 licensed with at most 8B
parameters. All other dependencies are permissively licensed. `unidecode` (GPL) was
replaced by `anyascii` (ISC). No external data or APIs are used at any stage;
pretrained weights are downloaded once from the Hugging Face Hub.

## Models

| Model | Licence | Params | Verified | Use |
|---|---|---|---|---|
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | **MIT** (model-card tag `license:mit`) | 117.7M (counted locally) | HF model API, 2026-09-26 | Dense blocking pass (base), starting point for fine-tuning |
| `artifacts/models/e5s-ft` (our fine-tune of the above) | MIT (derivative) | 117.7M | — | Dense blocking pass (fine-tuned) |
| `artifacts/models/xenc` (v3 cross-encoder: e5s-ft backbone + 1-logit head, fine-tuned on TRAIN-split s3train pairs) | MIT (derivative of intfloat/multilingual-e5-small) | ~118M | — | v3 pair "judge" feature (`src/xenc.py`) |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Apache-2.0 | 117.7M | HF model API, 2026-09-26 | Benchmarked only (half the encode throughput of e5-small); not used |

## Python packages (pinned in `requirements.txt`)

| Package | Version | Licence |
|---|---|---|
| pandas | 3.0.5 | BSD-3-Clause |
| numpy | 2.5.2 | BSD-3-Clause (+0BSD/MIT/Zlib bundled parts) |
| pyarrow | 25.0.1 | Apache-2.0 |
| scipy | 1.18.1 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| rapidfuzz | 3.14.6 | MIT |
| lightgbm | 4.7.0 | MIT |
| jellyfish | 1.2.1 | MIT |
| anyascii | 0.3.3 | ISC |
| torch (CUDA 13.0 build) | 2.14.0 | BSD-3-Clause style / Apache-2.0 (per package metadata) |
| sentence-transformers | 6.1.0 | Apache-2.0 |
| transformers | 5.17.0 | Apache-2.0 |
| datasets | 5.0.1 | Apache-2.0 |
| accelerate | 1.15.0 | Apache-2.0 |
| tokenizers | 0.23.2 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| huggingface-hub | 1.33.0 | Apache-2.0 |
