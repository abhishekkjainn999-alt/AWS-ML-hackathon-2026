"""Country-agnostic normalisation of names and addresses, cached as parquet.

One function set for every record; no per-country branches. Country labels are
only used to *group* records when learning address vocabularies (state/city)
from the data itself, never to choose a code path.

Pipeline
  phase 1 (per record, multiprocess):
      name_full, name_core, is_web, name_nonlatin,
      addr_norm, addr_comps ("|"-joined normalised comma components),
      numbers, postcode, landmark
  learn (global, from data only):
      * transliteration dictionary: native-script token -> latin token, learned
        by positional alignment of TRAIN-split matched pairs (val untouched)
      * state vocabulary per country label: greedy edge-component cover on S1
      * component -> state map (S1 city votes + unsupervised S2/S3 aliases
        inferred from co-occurring cities, + fuzzy fallback)
  phase 2 (per record, multiprocess):
      remap native-script name tokens, recompute core; derive state and city

Output: artifacts/norm/{split}_s{n}.parquet
Usage:  python src/normalize.py [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii

from io_utils import ARTIFACTS_DIR, load_ground_truth, load_source, vdir

NORM_DIR = vdir("norm")

# ---------------------------------------------------------------------------
# Word lists (one combined, multilingual list each; no country switches)
# ---------------------------------------------------------------------------
LEGAL = """pvt pte private ltd limited llc llp lp plc pllc pc inc incorporated corp
corporation co cos company opc sarl sas sasu eurl sa sci snc scop scm gmbh ag kg bv nv
pty srl spa ab oy as""".split()
TITLES = "mr mrs ms miss dr sri shri shree smt messrs prof".split()
GENERIC = """enterprises enterprise services service group partners holdings the and et
of""".split()
# v2 additions (one shared multilingual list): company-form words, generic suffix words
# seen as noise, and articles / prepositions.
LEGAL_V2 = "cie ets etablissements etablissement ste societe compagnie ei".split()
GENERIC_V2 = """groupe holding participations developpement distribution de du des la le
les d l""".split()
CORE_STOP = frozenset(LEGAL + TITLES + GENERIC + LEGAL_V2 + GENERIC_V2)
HF_CUTOFF = 0.01  # v2: name tokens in > 1% of the split's S1 records are dropped from core

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "boul": "boulevard", "ln": "lane",
    "dr": "drive", "ct": "court", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "sq": "square", "cir": "circle", "trl": "trail", "cres": "crescent",
    "mt": "mount", "ft": "fort", "apt": "apartment", "appt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "r": "rue", "rte": "route",
    "chem": "chemin", "imp": "impasse", "fbg": "faubourg", "pte": "porte",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest", "opp": "opposite",
    "nr": "near", "hsg": "housing", "soc": "society", "ph": "phase", "sec": "sector",
    "dist": "district", "tq": "taluk", "stn": "station", "jn": "junction",
    "jct": "junction", "cplx": "complex", "hosp": "hospital", "univ": "university",
    "mkt": "market", "ngr": "nagar", "clny": "colony", "extn": "extension",
    "ext": "extension",
    # v2
    "all": "allee", "crs": "cours", "qu": "quai",
}
ADDR_DROP = frozenset({"cedex"})
NUM_SUFFIX = frozenset({"bis", "ter", "quater"})
_NUM_SFX = re.compile(r"^(\d+)(bis|ter|quater|[a-z])$")
NUMBER_MARKERS = frozenset("no nos ndeg num".split())
LANDMARK_WORDS = frozenset("near opposite behind beside besides adjacent next".split())
LANDMARK_STOP = frozenset("to the of a an".split())

_NUMSIGN = re.compile(r"(?i)\b(?:n|no)\s*[°º]|[°º#]")
_APOS = re.compile(r"['’`´]")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_PIN_SPLIT = re.compile(r"(?<!\d)(\d{3}) (\d{3})(?!\d)")
_WEB = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.[a-z]{2,6}(?:\.[a-z]{2})?/?$")
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th|e|er|eme|ere)$")
_LOOKALIKE = str.maketrans({"5": "s", "0": "o", "1": "l", "3": "e"})


def _latinish(ch: str) -> bool:
    return ord(ch) <= 0x24F


def is_nonlatin(s: str) -> bool:
    """True if the string contains letters outside the Latin blocks."""
    return any(c.isalpha() and not _latinish(c) for c in s)


def to_ascii_lower(s: str) -> str:
    s = _NUMSIGN.sub(" ", s)
    if not s.isascii():
        s = anyascii(s)
    s = _APOS.sub("", s.lower())
    return s


def merge_single_letters(tokens: list[str]) -> list[str]:
    """'l l c' -> 'llc', 'e u r l' -> 'eurl', 's r arcade' -> 'sr arcade'."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def fix_lookalike(tok: str) -> str:
    """Replace look-alike digits only inside mostly-alphabetic words ('5equoia')."""
    if tok.isalpha() or tok.isdigit() or _ORDINAL.match(tok):
        return tok
    n_alpha = sum(c.isalpha() for c in tok)
    n_digit = len(tok) - n_alpha
    if n_alpha >= 3 and n_digit * 2 <= n_alpha:
        return tok.translate(_LOOKALIKE)
    return tok


def core_tokens(tokens: list[str]) -> list[str]:
    core = [t for t in tokens if t not in CORE_STOP]
    return core or tokens


def normalize_name(raw: str) -> tuple[str, str, bool, bool]:
    """-> (name_full, name_core, is_web, name_nonlatin)."""
    nonlatin = is_nonlatin(raw)
    s = to_ascii_lower(raw.strip())
    is_web = False
    m = _WEB.match(s.strip())
    if m and " " not in s.strip():
        is_web = True
        s = m.group(1)
    s = s.replace("&", " and ")
    tokens = merge_single_letters(_NONALNUM.sub(" ", s).split())
    tokens = [fix_lookalike(t) for t in tokens]
    return " ".join(tokens), " ".join(core_tokens(tokens)), is_web, nonlatin


def _addr_tokens(comp: str) -> tuple[list[str], str]:
    """Tokenise one address component. Returns (tokens, house-number suffix).

    '1T' / '16a' / '119bis' -> number + suffix; 'bis'/'ter'/'quater' after a number are
    suffixes (never 'terrace'); 'st' before an alphabetic token -> 'saint' ('St-Nazaire',
    'St Charles Ave'), else 'street'; 'cedex' dropped.
    """
    raw = _NONALNUM.sub(" ", comp).split()
    toks, sfx = [], ""
    for t in raw:
        m = _NUM_SFX.match(t)
        if m:
            toks.append(m.group(1))
            sfx = sfx or m.group(2)
            continue
        if (t in NUM_SUFFIX or t in ("a", "b", "c", "d")) and toks and toks[-1].isdigit():
            sfx = sfx or t
            continue
        if t in NUMBER_MARKERS or t in ADDR_DROP:
            continue
        toks.append(t)
    toks = merge_single_letters(toks)
    # A one-token component is usually a region/state code ("CT", "MT", "NE"):
    # only expand street-type abbreviations inside multi-token components.
    if len(toks) > 1:
        out = []
        for i, t in enumerate(toks):
            if t == "st":
                nxt = toks[i + 1] if i + 1 < len(toks) else ""
                out.append("saint" if nxt.isalpha() and nxt not in ADDR_ABBR else "street")
            else:
                out.append(ADDR_ABBR.get(t, t))
        toks = out
    return toks, sfx


def normalize_address(raw: str) -> tuple[str, str, str, str, str, str, str]:
    """-> (addr_norm, addr_comps, numbers, postcode, landmark, house_sfx, street)."""
    s = to_ascii_lower(raw)
    s = _PIN_SPLIT.sub(r"\1\2", s)
    comps, landmarks, nums, posts = [], [], [], []
    house_sfx, street = "", ""
    for comp in s.split(","):
        toks, sfx = _addr_tokens(comp)
        if not street and any(t.isdigit() for t in toks):
            # street key: alphabetic tokens of the first component holding a number
            street = " ".join(sorted({t for t in toks if t.isalpha() and len(t) > 1}))
            house_sfx = sfx
        for i, t in enumerate(toks):
            if t in LANDMARK_WORDS:
                lm = [x for x in toks[i + 1:i + 5] if x not in LANDMARK_STOP
                      and x not in LANDMARK_WORDS]
                landmarks.extend(lm)
        for d in _DIGITS.findall(comp):  # all digit runs; postcode-like ones also below
            nums.append(d)
            if len(d) in (5, 6):
                posts.append(d)
        if toks:
            comps.append(" ".join(toks))
    return (" ".join(comps), "|".join(comps), " ".join(dict.fromkeys(nums)),
            " ".join(dict.fromkeys(posts)), " ".join(dict.fromkeys(landmarks)), house_sfx, street)


def _phase1_chunk(args):
    names, addrs = args
    rows = [normalize_name(n) + normalize_address(a) for n, a in zip(names, addrs)]
    return rows


PHASE1_COLS = ["name_full", "name_core", "is_web", "name_nonlatin",
               "addr_norm", "addr_comps", "numbers", "postcode", "landmark", "house_sfx", "street"]


def _chunks(df, size):
    for i in range(0, len(df), size):
        yield (df["business_name"].iloc[i:i + size].tolist(),
               df["business_address"].iloc[i:i + size].tolist())


def phase1(pool, split, src, chunk=50_000) -> pd.DataFrame:
    df = load_source(split, src)
    rows = []
    for part in pool.imap(_phase1_chunk, _chunks(df, chunk)):
        rows.extend(part)
    out = pd.DataFrame(rows, columns=PHASE1_COLS)
    out.insert(0, "entity_id", df["entity_id"].values)
    out.insert(1, "country", df["country"].values)
    out["raw_name"] = df["business_name"].values
    out["raw_address"] = df["business_address"].values
    return out


# ---------------------------------------------------------------------------
# Learned resources
# ---------------------------------------------------------------------------
def learn_translit_dict(s1: pd.DataFrame, others: pd.DataFrame, gt: dict,
                        train_ids: set, min_count=3, min_share=0.5) -> dict:
    """Map transliterated native-script tokens to latin tokens via aligned pairs.

    Uses only TRAIN-split S1 entities. A pair (S1 latin name, S2/S3 native-script
    name) with equal token counts is aligned position by position.
    """
    s1_name = dict(zip(s1["entity_id"], s1["name_full"]))
    nl = others[others["name_nonlatin"]]
    nl_name = dict(zip(nl["entity_id"], nl["name_full"]))
    counts: dict[str, Counter] = defaultdict(Counter)
    for s1_id, ms in gt.items():
        if s1_id not in train_ids:
            continue
        a = s1_name.get(s1_id)
        if a is None:
            continue
        at = a.split()
        for m in ms:
            b = nl_name.get(m)
            if b is None:
                continue
            bt = b.split()
            if len(bt) == len(at):
                for x, y in zip(bt, at):
                    counts[x][y] += 1
    out = {}
    for x, c in counts.items():
        y, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= min_count and n / tot >= min_share and x != y:
            out[x] = y
    return out


def _edge_comps(comps: str) -> tuple[str, str]:
    parts = comps.split("|") if comps else []
    parts = [p for p in parts if p and not any(ch.isdigit() for ch in p)]
    if not parts:
        return "", ""
    return parts[0], parts[-1]


def learn_state_maps(s1: pd.DataFrame, others_sample: pd.DataFrame,
                     cover_target=0.998, min_gain=0.0001) -> dict:
    """Per country label: state vocabulary + component->state map, from data only."""
    from rapidfuzz import fuzz, process

    result = {}
    for country, g in s1.groupby("country"):
        comps_lists = [[p for p in c.split("|") if p and not any(ch.isdigit() for ch in p)]
                       for c in g["addr_comps"]]
        edges = [(c[0], c[-1]) if c else ("", "") for c in comps_lists]
        # greedy cover of records by edge components
        edge_count = Counter()
        for a, b in edges:
            edge_count.update({a, b} - {""})
        covered = np.zeros(len(edges), dtype=bool)
        comp_to_rows = defaultdict(list)
        for i, (a, b) in enumerate(edges):
            for x in {a, b} - {""}:
                comp_to_rows[x].append(i)
        states = []
        n = len(edges)
        cand = [x for x, _ in edge_count.most_common(2000)]
        while covered.mean() < cover_target and cand:
            gains = [(int((~covered[comp_to_rows[x]]).sum()), x) for x in cand]
            gain, best = max(gains)
            if gain / n < min_gain:
                break
            states.append(best)
            covered[comp_to_rows[best]] = True
            cand.remove(best)
        # A chosen component that mostly co-occurs (anywhere in the address) with one
        # earlier, larger chosen component is a city of that state, not a state itself.
        kept = []
        for x in states:
            other = Counter()
            rows_x = comp_to_rows[x]
            for i in rows_x:
                other.update({y for y in comps_lists[i] if y in kept and y != x})
            if not other or other.most_common(1)[0][1] < 0.3 * len(rows_x):
                kept.append(x)
        states = kept
        state_set = set(states)

        # S1: other components vote for the record's state
        votes: dict[str, Counter] = defaultdict(Counter)
        for c in comps_lists:
            st = [x for x in c if x in state_set]
            if len(set(st)) != 1:
                continue
            for x in c:
                if x not in state_set:
                    votes[x][st[0]] += 1
        cmap, ccount = {}, {}
        for x, c in votes.items():
            s, k = c.most_common(1)[0]
            if k >= 3 and k / sum(c.values()) >= 0.7:
                cmap[x], ccount[x] = s, k

        # S2/S3 (unsupervised): components voted by the OTHER co-occurring components
        o = others_sample[others_sample["country"] == country]
        avotes: dict[str, Counter] = defaultdict(Counter)
        edge_unknown = Counter()
        for comps in o["addr_comps"]:
            c = [p for p in comps.split("|") if p and not any(ch.isdigit() for ch in p)]
            if c:
                for x in {c[0], c[-1]}:
                    if x not in state_set and x not in cmap:
                        edge_unknown[x] += 1
            for x in set(c) - state_set:
                inferred = Counter(cmap[y] for y in c if y in cmap and y != x)
                inferred.update({y: 3 for y in c if y in state_set})
                if inferred:
                    avotes[x][inferred.most_common(1)[0][0]] += 1
        amap = {}
        for x, c in avotes.items():
            s, k = c.most_common(1)[0]
            # a component that is a rare S1 city but a frequent S2/S3 state variant
            # (e.g. a town named like a state) follows its dominant usage
            if k >= 10 and k / sum(c.values()) >= 0.8 and \
                    (x not in cmap or (cmap[x] != s and k > 10 * ccount[x])):
                amap[x] = s
        # fuzzy fallback for frequent unknown edge components (typos / variants)
        fz = {}
        rest = [x for x, k in edge_unknown.items() if k >= 5 and x not in amap and len(x) >= 4]
        if rest and states:
            for x in rest:
                hit = process.extractOne(x, states, scorer=fuzz.ratio, score_cutoff=88)
                if hit:
                    fz[x] = hit[0]
        comp_map = {**fz, **cmap, **amap, **{s: s for s in states}}
        result[country] = {"states": states, "comp_map": comp_map,
                           "city_vocab": sorted(set(cmap) - set(amap))}
        print(f"  [{country}] states={len(states)} cover={covered.mean():.4f} "
              f"city_map={len(cmap)} alias_map={len(amap)} fuzzy={len(fz)}")
    return result


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
_G: dict = {}


def _init2(translit, state_maps, hf):
    _G["translit"] = translit
    _G["state_maps"] = {c: (set(m["states"]), m["comp_map"], set(m["city_vocab"]))
                        for c, m in state_maps.items()}
    _G["hf"] = {k: frozenset(v) for k, v in hf.items()}


def core_tokens_v2(toks: list[str], hf: frozenset) -> list[str]:
    """Core = tokens minus the shared stoplist and the split's high-frequency tokens;
    falls back to stoplist-only removal, then to all tokens."""
    core = [t for t in toks if t not in CORE_STOP and t not in hf]
    return core or [t for t in toks if t not in CORE_STOP] or toks


def high_freq_tokens(names, cutoff=HF_CUTOFF) -> list[str]:
    """Name tokens present in more than `cutoff` of the records (same rule everywhere)."""
    df = Counter(t for n in names for t in set(n.split()))
    n = max(len(names), 1)
    return sorted(t for t, c in df.items() if c / n > cutoff and t not in CORE_STOP)


def _phase2_chunk(args):
    names, nonlatin, comps_col, countries, split = args
    tr = _G["translit"]
    hf = _G["hf"].get(split, frozenset())
    out = []
    for name, nl, comps, country in zip(names, nonlatin, comps_col, countries):
        toks = [tr.get(t, t) for t in name.split()] if nl else name.split()
        name = " ".join(toks)
        core = " ".join(core_tokens_v2(toks, hf))
        states, cmap, cities = _G["state_maps"].get(country, (set(), {}, set()))
        parts = [p for p in comps.split("|") if p] if comps else []
        alpha = [p for p in parts if not any(ch.isdigit() for ch in p)]
        direct = [p for p in alpha if p in states]
        if direct:
            state = Counter(direct).most_common(1)[0][0]
        else:
            vc = Counter(cmap[p] for p in alpha if p in cmap)
            state = vc.most_common(1)[0][0] if vc else ""
        city = ""
        for p in reversed(alpha):
            if p in cities and p not in states:
                city = p
                break
        if not city:
            rest = [p for p in alpha if p not in states and p not in cmap]
            city = rest[-1] if rest else ""
        out.append((name, core, state, city))
    return out


def phase2(pool, df: pd.DataFrame, split: str, chunk=50_000) -> pd.DataFrame:
    args = ((df["name_full"].iloc[i:i + chunk].tolist(),
             df["name_nonlatin"].iloc[i:i + chunk].tolist(),
             df["addr_comps"].iloc[i:i + chunk].tolist(),
             df["country"].iloc[i:i + chunk].tolist(), split) for i in range(0, len(df), chunk))
    rows = []
    for part in pool.imap(_phase2_chunk, args):
        rows.extend(part)
    r = pd.DataFrame(rows, columns=["name_full", "core2", "state", "city"])
    df = df.copy()
    df["name_full"] = r["name_full"].values
    df["name_core"] = r["core2"].values
    df["state"] = r["state"].values
    df["city"] = r["city"].values
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = {}
    with Pool(args.workers) as pool:
        for split in ("train", "test"):
            for src in (1, 2, 3):
                t = time.time()
                df = phase1(pool, split, src)
                path = NORM_DIR / f"_p1_{split}_s{src}.parquet"
                df.to_parquet(path, index=False)
                frames[(split, src)] = path
                print(f"phase1 {split} S{src}: {len(df):,} rows in {time.time() - t:.0f}s")
                del df

    # ---- learn transliteration dictionary (TRAIN split only) ----
    t = time.time()
    split_df = pd.read_csv(ARTIFACTS_DIR / "val_split.tsv", sep="\t", dtype=str)
    train_ids = set(split_df.loc[split_df["split"] == "train", "entity_id"])
    cols = ["entity_id", "country", "name_full", "name_nonlatin", "addr_comps"]
    s1 = pd.read_parquet(frames[("train", 1)], columns=cols)
    oth = pd.concat([pd.read_parquet(frames[("train", s)], columns=cols) for s in (2, 3)],
                    ignore_index=True)
    translit = learn_translit_dict(s1, oth, load_ground_truth(), train_ids)
    (NORM_DIR / "translit_dict.json").write_text(
        json.dumps(translit, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"translit dict: {len(translit):,} entries in {time.time() - t:.0f}s")

    # ---- learn state/city maps (unsupervised; train+test) ----
    t = time.time()
    s1_all = pd.concat([s1, pd.read_parquet(frames[("test", 1)], columns=cols)],
                       ignore_index=True)
    oth_test = pd.concat([pd.read_parquet(frames[("test", s)], columns=cols) for s in (2, 3)],
                         ignore_index=True)
    others_sample = pd.concat([oth.sample(min(len(oth), 1_500_000), random_state=0),
                               oth_test.sample(min(len(oth_test), 1_500_000), random_state=0)])
    del oth, oth_test
    state_maps = learn_state_maps(s1_all, others_sample)
    del s1, s1_all, others_sample
    (NORM_DIR / "state_maps.json").write_text(
        json.dumps({c: {"states": m["states"],
                        "comp_map_size": len(m["comp_map"])} for c, m in state_maps.items()},
                   indent=1), encoding="utf-8")
    print(f"state maps learned in {time.time() - t:.0f}s")

    # ---- high-frequency name tokens per split (S1, after transliteration mapping) ----
    hf = {}
    for split in ("train", "test"):
        d = pd.read_parquet(frames[(split, 1)], columns=["name_full", "name_nonlatin"])
        names = [" ".join(translit.get(t, t) for t in n.split()) if nl else n
                 for n, nl in zip(d["name_full"], d["name_nonlatin"])]
        hf[split] = high_freq_tokens(names)
        print(f"high-frequency S1 name tokens ({split}, >{HF_CUTOFF:.0%}): {hf[split]}")
    (NORM_DIR / "high_freq_tokens.json").write_text(json.dumps(hf, indent=1), encoding="utf-8")

    # ---- phase 2 ----
    with Pool(args.workers, initializer=_init2, initargs=(translit, state_maps, hf)) as pool:
        for (split, src), path in frames.items():
            t = time.time()
            df = phase2(pool, pd.read_parquet(path), split)
            df.to_parquet(NORM_DIR / f"{split}_s{src}.parquet", index=False)
            path.unlink()
            print(f"phase2 {split} S{src}: {time.time() - t:.0f}s")
            del df
    print(f"total normalisation time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
