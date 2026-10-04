"""Indiana neurosurgical cohorts (IDOH PUF aggregates, 2017-2024): cranial vs spine.

1. Two definitions from separate hospital-level tables:
   - MS-DRG table (same groups as CMS)
   - principal ICD-10-PCS table (surgical root operations on brain/cranial nerves vs spinal cord,
     vertebral joints, discs and spinal nerve roots)
2. Like-for-like validation against CMS Medicare FFS by hospital (CCN): Indiana's payer field
   separates Medicare FFS from Medicare Advantage from 2021; earlier years use all Medicare.
Outputs: data/interim/state_compare/IN_neuro_*.csv
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DRG_TO_CAT, MSDRG_GROUP, ROOT

RAW = ROOT / "data/raw/states/IN"
OUT = ROOT / "data/interim/state_compare"

SURG_OPS = set("159BCNQST")          # bypass, destruction, drainage, excision, extirpation, release, repair, reposition, resection
CNS_CRANIAL = set("0123456789ABCDEFGHJKLMNPQRS")
CNS_SPINE = set("TUVWXY")


def pcs_group(code: str):
    c = str(code).upper()
    if len(c) < 4:
        return None
    if c.startswith("00") and c[2] in SURG_OPS:
        return "cranial" if c[3] in CNS_CRANIAL else "spine" if c[3] in CNS_SPINE else None
    if re.match(r"^0RG[0124678A]|^0SG[013]", c):
        return "spine"                                  # vertebral fusion
    if re.match(r"^0R[BT][359B]|^0S[BT][24]|^01N[18BR]|^0PB[34]|^0QB[01]", c):
        return "spine"                                  # discectomy, nerve root release, laminectomy-type excision
    return None


def payer_map(year: int) -> dict:
    f = next(RAW.glob(f"{year}_PAY*"))
    k = pd.read_excel(f, sheet_name=0)
    k = k.set_axis(["code", "desc"] + list(k.columns[2:]), axis=1)
    out = {}
    for r in k.itertuples():
        d = str(r.desc).lower()
        out[int(r.code)] = ("medicare_ffs" if "medicare" in d and "fee" in d else
                            "medicare_adv" if "medicare advantage" in d else
                            "medicare_any" if "medicare" in d else "other")
    return out


def load(kind: str) -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob(f"*_{kind}_IN*.csv")):
        y = int(f.name[:4])
        d = pd.read_csv(f, dtype={"PROCEDURE_1": str})
        d.columns = [c.upper().replace("PAYER1", "PAYOR1") for c in d.columns]
        d["year"] = y
        d["payer"] = d.PAYOR1.map(payer_map(y)).fillna("other")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    ms = load("MSDRG")
    ms["category"] = ms.MSDRG.map(DRG_TO_CAT)
    ms["grp"] = ms.category.map(MSDRG_GROUP)
    pc = load("PROC")
    pc["grp"] = pc.PROCEDURE_1.map(pcs_group)

    # 1. definition comparison (statewide, all payers)
    defs = pd.DataFrame({
        "msdrg": ms[ms.grp.isin(["cranial", "spine"])].groupby(["year", "grp"]).PATS.sum(),
        "pcs_principal": pc.dropna(subset=["grp"]).groupby(["year", "grp"]).PATS.sum(),
    })
    defs["pcs_over_msdrg"] = defs.pcs_principal / defs.msdrg
    defs.to_csv(OUT / "IN_neuro_definitions.csv")
    print(defs.round(3).to_string())
    ms[ms.category.notna()].groupby(["year", "category"]).PATS.sum().unstack().to_csv(OUT / "IN_neuro_msdrg_categories.csv")
    ms[ms.grp.isin(["cranial", "spine"])].groupby(["year", "grp", "payer"]).PATS.sum().unstack().to_csv(OUT / "IN_neuro_payer_mix.csv")

    # 2. validation vs CMS Medicare FFS
    xw = pd.read_csv(OUT / "IN_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok][["id", "ccn"]]
    med = ms[ms.grp.isin(["cranial", "spine"])].copy()
    med["ffs_like"] = np.where(med.year >= 2021, med.payer == "medicare_ffs",
                               med.payer.isin(["medicare_ffs", "medicare_any", "medicare_adv"]))
    med = med[med.ffs_like].rename(columns={"HOSPITAL_ID": "id"}).merge(xw, on="id")
    cells_state = med.groupby(["year", "grp", "ccn", "MSDRG"]).PATS.sum().rename("state_medicare")
    rows = pd.read_parquet(ROOT / "data/interim/cms_neuro_drg_rows.parquet")
    rows = rows[(rows.state == "IN") & rows.category.map(MSDRG_GROUP).isin(["cranial", "spine"])]
    rows = rows.assign(grp=rows.category.map(MSDRG_GROUP)).rename(columns={"drg": "MSDRG"})
    cells_cms = rows.groupby(["year", "grp", "ccn", "MSDRG"]).n.sum().rename("cms_ffs")
    cells = pd.concat([cells_state, cells_cms], axis=1).reset_index()
    cells.to_csv(OUT / "IN_neuro_vs_cms_cells.csv", index=False)

    # cell-matched: exact (hospital, MS-DRG) agreement where CMS publishes the cell
    pub = cells.dropna(subset=["cms_ffs"]).fillna({"state_medicare": 0})
    cm = pub.groupby(["grp", "year"]).apply(lambda d: pd.Series({
        "cells_published": len(d), "cms": d.cms_ffs.sum(), "state": d.state_medicare.sum(),
        "ratio": d.state_medicare.sum() / d.cms_ffs.sum(),
        "cells_within_10pct": (np.abs(d.state_medicare - d.cms_ffs) <= 0.1 * d.cms_ffs).mean(),
        "cells_exact": (d.state_medicare == d.cms_ffs).mean(),
        "spearman_cells": spearmanr(d.cms_ffs, d.state_medicare).statistic if len(d) > 5 else np.nan,
    }), include_groups=False)
    # what CMS suppression hides: state cases in cells CMS does not publish
    hid = cells.fillna({"cms_ffs": 0}).groupby(["grp", "year"]).apply(lambda d: pd.Series({
        "state_cases_unpublished_cells": d.loc[d.cms_ffs == 0, "state_medicare"].sum(),
        "unpublished_share": d.loc[d.cms_ffs == 0, "state_medicare"].sum() / d.state_medicare.sum(),
        "hospitals_with_state_cases": d.loc[d.state_medicare > 0, "ccn"].nunique(),
        "hospitals_with_any_cms_cell": d.loc[d.cms_ffs > 0, "ccn"].nunique(),
    }), include_groups=False)
    cell_summary = cm.join(hid)
    cell_summary.to_csv(OUT / "IN_neuro_cell_validation.csv")
    print(cell_summary.round(3).to_string())

    st = cells.groupby(["year", "grp", "ccn"]).state_medicare.sum(min_count=1)
    cms = cells.groupby(["year", "grp", "ccn"]).cms_ffs.sum(min_count=1)
    v = pd.concat([st, cms], axis=1).reset_index()
    v.to_csv(OUT / "IN_neuro_vs_cms_hospital.csv", index=False)

    def val(d):
        both = d.dropna(subset=["cms_ffs", "state_medicare"])
        a = both.cms_ffs >= both.cms_ffs.quantile(0.75)
        b = both.state_medicare >= both.state_medicare.quantile(0.75)
        return pd.Series({
            "ccn_both": len(both), "cms_total": d.cms_ffs.sum(), "state_total": d.state_medicare.sum(),
            "ratio_matched": both.state_medicare.sum() / both.cms_ffs.sum(),
            "median_hosp_ratio": np.exp(np.log(both.state_medicare / both.cms_ffs).median()),
            "within_10pct": (np.log(both.state_medicare / both.cms_ffs).abs() < np.log(1.1)).mean(),
            "spearman": spearmanr(both.cms_ffs, both.state_medicare).statistic if len(both) > 5 else np.nan,
            "top_quartile_jaccard": (a & b).sum() / max(1, (a | b).sum()),
            "hospitals_hidden_by_cms": (d.cms_ffs.isna() & d.state_medicare.notna()).sum(),
            "state_cases_hidden_by_cms": d.loc[d.cms_ffs.isna(), "state_medicare"].sum(),
            "hidden_share": d.loc[d.cms_ffs.isna(), "state_medicare"].sum() / d.state_medicare.sum(),
        })
    validation = v.groupby(["grp", "year"]).apply(val, include_groups=False)
    validation.to_csv(OUT / "IN_neuro_vs_cms_by_year.csv")
    print(validation.round(3).to_string())
