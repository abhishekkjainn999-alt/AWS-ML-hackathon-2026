"""Shared I/O helpers: paths and TSV loaders.

All data files are tab-separated. Some names/addresses contain literal double
quotes, so we read with ``quoting=csv.QUOTE_NONE`` to keep them verbatim, and
with ``keep_default_na=False`` so empty fields stay as "" rather than NaN
(and strings like "NA"/"None" are not silently turned into missing values).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

# code/business_entity_resolution/src -> student_resource root
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", REPO_ROOT / "dataset"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", REPO_ROOT / "output"))
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
# Pipeline version: BER_VERSION=v2 routes every version-dependent artifact folder
# (norm, blocking, prerank, features, model, ...) to "<name>_v2"; embeddings and the
# fine-tuned model are shared across versions.
VERSION = os.environ.get("BER_VERSION", "")
# A version can inherit folders from a base version (e.g. v3 = v2 norm/blocking/prerank
# + its own features/model): BER_BASE_VERSION=v2 BER_INHERIT=norm,blocking,prerank,feat_cache
BASE_VERSION = os.environ.get("BER_BASE_VERSION", "")
INHERIT = set(filter(None, os.environ.get("BER_INHERIT", "").split(",")))


def vdir(name: str) -> Path:
    v = BASE_VERSION if name in INHERIT else VERSION
    return ARTIFACTS_DIR / (f"{name}_{v}" if v else name)

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
READ_KW = dict(sep="\t", quoting=csv.QUOTE_NONE, keep_default_na=False,
               na_filter=False, dtype=str, encoding="utf-8")


def source_path(split: str, source: int) -> Path:
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def load_source(split: str, source: int, usecols=None) -> pd.DataFrame:
    """Load one source file (split in {'train','test'}, source in {1,2,3})."""
    return pd.read_csv(source_path(split, source), usecols=usecols, **READ_KW)


def load_ground_truth(path: Path | str | None = None) -> dict[str, set[str]]:
    """Return {S1 id: set of matched S2/S3 ids} (empty set for singletons)."""
    path = path or DATA_DIR / "train" / "train_ground_truth.tsv"
    return load_id_list_file(path)


def load_id_list_file(path: Path | str) -> dict[str, set[str]]:
    """Parse any two-column `S1 id <TAB> comma-separated ids` file.

    Works for the ground truth, matching_results.tsv and candidate_pairs.tsv.
    """
    df = pd.read_csv(path, **READ_KW)
    s1_col, ids_col = df.columns[:2]
    return {
        s1: set(filter(None, (x.strip() for x in ids.split(",")))) if ids else set()
        for s1, ids in zip(df[s1_col], df[ids_col])
    }


def write_id_list_file(mapping: dict[str, set[str] | list[str]], path: Path | str,
                       header: tuple[str, str]) -> None:
    """Write a matching_results / candidate_pairs style TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for s1, ids in mapping.items():
            f.write(f"{s1}\t{','.join(dict.fromkeys(ids))}\n")
