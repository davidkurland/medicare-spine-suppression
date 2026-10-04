"""Shared helpers: fuzzy hospital matching to CMS HCRIS and agreement statistics."""
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Neurosurgical MS-DRG groups (CMS v31-v42). Craniotomy DRGs 020-027 include endovascular intracranial work.
MSDRG = {
    "cranial_open_endo": [20, 21, 22, 23, 24, 25, 26, 27],
    "cranial_shunt": [31, 32, 33],
    # FY2025 (Oct 2024) split fusion into single/multiple-level DRGs 402, 426-430, 447-448, 450-451
    "spine_fusion": [453, 454, 455, 456, 457, 458, 459, 460, 471, 472, 473,
                     402, 426, 427, 428, 429, 430, 447, 448, 450, 451],
    "spine_nonfusion": [28, 29, 30, 490, 491, 518, 519, 520],
    "sens_extracranial_carotid": [34, 35, 36, 37, 38, 39],
    "sens_periph_nerve": [40, 41, 42],
}
MSDRG_GROUP = {k: ("cranial" if k.startswith("cranial") else "spine" if k.startswith("spine") else "sensitivity")
               for k in MSDRG}
DRG_TO_CAT = {d: k for k, v in MSDRG.items() for d in v}
# Keep distinguishing words such as "community", "north", "regional": dropping them collapses
# names like "Community Hospital East" to nothing and produces confident wrong matches.
STOP = {"hospital", "hospitals", "hosp", "center", "ctr", "medical", "med", "health", "healthcare",
        "the", "of", "and", "at", "inc", "llc", "corp", "co"}


def tokens(s) -> set:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return {("st" if t == "saint" else t) for t in s.split() if t not in STOP}


def name_score(a, b) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    jacc = len(ta & tb) / len(ta | tb)
    seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return 0.5 * jacc + 0.5 * seq


def hcris_state(state: str, years=None) -> pd.DataFrame:
    h = pd.read_parquet(ROOT / "data/interim/hcris_hospital_cy.parquet")
    h = h[h.state == state]
    if years is not None:
        h = h[h.year.isin(years)]
    return h


def match_to_hcris(fac: pd.DataFrame, state: str) -> pd.DataFrame:
    """fac: columns id, name, city. Returns id -> ccn with scores and match_ok."""
    h = hcris_state(state).sort_values("year").drop_duplicates("ccn", keep="last")
    h = h[h.fac_type != "PH"].copy()
    h["city_u"] = h.city.str.upper().str.strip()
    # score every (facility, CCN) pair, then assign greedily from the best pair down so each facility
    # takes its best-scoring acceptable CCN; several facilities may share a CCN
    pairs = []
    for r in fac.itertuples(index=False):
        city = (h.city_u == str(r.city).upper().strip()).to_numpy()
        ns = h.name.apply(lambda n: name_score(r.name, n)).to_numpy()
        for j in np.argsort(-(ns + 0.5 * city))[:5]:
            pairs.append((ns[j] + 0.5 * city[j], ns[j], bool(city[j]), r.id, r.name, r.city, j))
    pairs.sort(key=lambda p: -p[0])
    taken_ids, out = set(), []
    for total, ns, same_city, fid, name, city, j in pairs:
        if fid in taken_ids:
            continue
        ccn = h.iloc[j].ccn
        # system brand names ("Indiana University Health ...") make cross-city names look alike,
        # so a match in another city needs a near-identical name
        ok = ns >= 0.4 if same_city else ns >= 0.85
        if not ok:
            continue
        taken_ids.add(fid)
        out.append({"id": fid, "name": name, "city": city, "ccn": ccn, "cms_name": h.iloc[j]["name"],
                    "cms_type": h.iloc[j].fac_type, "name_score": round(ns, 3), "same_city": same_city,
                    "match_ok": True})
    unmatched = fac[~fac.id.isin(taken_ids)].assign(match_ok=False)
    return pd.concat([pd.DataFrame(out), unmatched], ignore_index=True)


def agreement(df: pd.DataFrame, state_col="sp", ref_col="discharges") -> pd.DataFrame:
    """Per-year agreement metrics between a state count column and HCRIS discharges."""
    def f(d):
        lr = np.log(d[state_col] / d[ref_col])
        return pd.Series({
            "hospitals": len(d), "state_total": d[state_col].sum(), "hcris_total": d[ref_col].sum(),
            "agg_ratio": d[state_col].sum() / d[ref_col].sum(),
            "median_ratio": np.exp(lr.median()),
            "within_10pct": (lr.abs() < np.log(1.1)).mean(),
            "within_25pct": (lr.abs() < np.log(1.25)).mean(),
            "off_2x": (lr.abs() > np.log(2)).mean(),
            "corr_log": np.corrcoef(np.log(d[state_col]), np.log(d[ref_col]))[0, 1],
        })
    d = df[(df[state_col] > 0) & (df[ref_col] > 0)]
    return d.groupby("year").apply(f, include_groups=False)
