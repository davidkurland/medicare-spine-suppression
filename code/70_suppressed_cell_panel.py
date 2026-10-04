"""Hospital-level panel of suppressed CMS spine cells.

CMS drops any hospital x MS-DRG x year cell with 1-10 discharges, and a hospital with 0 cases looks the
same as one with 1-10. This builds the cell panel:
  observed     y >= 11 (a published row in the provider file)
  suppressed   y in [0, 10], for every hospital in the file that year and each spine DRG active that year
plus the complete national and state totals (geography files) that the hidden volume must add up to,
and Indiana hospital-level ground truth, where the true 1-10 counts are known.

Outputs: data/interim/suppressed_cells/
  cells.parquet, nat_drg_year.csv, state_drg_year.csv, indiana_truth.csv, panel_summary.json
"""
import importlib
import json
import re

import numpy as np
import pandas as pd

from common import ROOT, MSDRG

l8 = importlib.import_module("08_cms_neuro")
OUT = ROOT / "data/interim/suppressed_cells"
OUT.mkdir(parents=True, exist_ok=True)
SPINE = sorted(set(MSDRG["spine_fusion"] + MSDRG["spine_nonfusion"]))


def geo_totals():
    """National and state complete discharges by spine DRG and year (geography files)."""
    rows = []
    for f in sorted((ROOT / "data/raw/cms_mupinp_geo").glob("*.CSV")):
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        g = pd.read_csv(f, dtype=str, encoding="latin-1", low_memory=False)
        c = {k.lower(): k for k in g.columns}
        g = g[[c["rndrng_prvdr_geo_lvl"], c["rndrng_prvdr_geo_desc"], c["drg_cd"], c["tot_dschrgs"]]]
        g.columns = ["lvl", "geo", "drg", "n"]
        g["drg"] = pd.to_numeric(g.drg, errors="coerce")
        g["n"] = pd.to_numeric(g.n, errors="coerce")
        rows.append(g[g.drg.isin(SPINE)].assign(year=y))
    g = pd.concat(rows, ignore_index=True)
    g["lvl"] = g.lvl.str.lower()
    return g[g.lvl == "national"], g[g.lvl == "state"]


def build():
    d = l8.load()
    d = d[d.year.between(2013, 2024)]
    sp = d[d.drg.isin(SPINE)].copy()
    nat, st = geo_totals()

    # DRGs that exist nationally in that year (the FY2025 fusion DRGs appear only in 2024)
    active = nat[nat.n > 0].groupby("year").drg.apply(lambda s: sorted(set(s))).to_dict()
    infile = d[["year", "ccn", "state"]].drop_duplicates()
    ever_spine = set(sp.ccn)
    pair_ever = set(map(tuple, sp[["ccn", "drg"]].drop_duplicates().to_numpy()))

    frames = []
    for y, drgs in active.items():
        h = infile[infile.year == y]
        cell = h.merge(pd.DataFrame({"drg": drgs}), how="cross")
        frames.append(cell)
    cells = pd.concat(frames, ignore_index=True)
    cells = cells.merge(sp[["year", "ccn", "drg", "n"]], on=["year", "ccn", "drg"], how="left")
    cells = cells.rename(columns={"n": "y"})
    cells["observed"] = cells.y.notna()
    cells["hosp_ever_spine"] = cells.ccn.isin(ever_spine)
    cells["pair_ever_published"] = [ (c, g) in pair_ever for c, g in zip(cells.ccn, cells.drg) ]
    # hospital total published spine volume that year: the strongest signal for what a hidden cell holds
    hv = sp.groupby(["year", "ccn"]).n.sum().rename("hosp_spine_published").reset_index()
    cells = cells.merge(hv, on=["year", "ccn"], how="left").fillna({"hosp_spine_published": 0})
    cells["stratum"] = np.where(cells.pair_ever_published, "A_core",
                         np.where(cells.hosp_ever_spine, "B_other_drg", "C_never_spine"))

    natdy = nat.groupby(["year", "drg"]).n.sum().rename("complete").reset_index()
    pubdy = sp.groupby(["year", "drg"]).n.sum().rename("published").reset_index()
    natdy = natdy.merge(pubdy, on=["year", "drg"], how="left").fillna({"published": 0})
    natdy["hidden"] = natdy.complete - natdy.published
    stdy = st.groupby(["year", "geo", "drg"]).n.sum().rename("complete_state").reset_index()

    cells.to_parquet(OUT / "cells.parquet", index=False)
    natdy.to_csv(OUT / "nat_drg_year.csv", index=False)
    stdy.to_csv(OUT / "state_drg_year.csv", index=False)

    summary = {
        "cells": int(len(cells)), "observed": int(cells.observed.sum()),
        "suppressed": int((~cells.observed).sum()),
        "cells_by_stratum": cells.stratum.value_counts().to_dict(),
        "hidden_national_by_year": natdy.groupby("year").hidden.sum().astype(int).to_dict(),
        "published_by_year": natdy.groupby("year").published.sum().astype(int).to_dict(),
        "spine_drgs": SPINE,
    }
    json.dump(summary, open(OUT / "panel_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "spine_drgs"}, indent=1)[:1200])
    return cells


def indiana_truth():
    """Indiana hospital x MS-DRG x year Medicare counts (state data) beside the published CMS cell."""
    p = ROOT / "data/interim/state_compare/IN_neuro_vs_cms_cells.csv"
    if not p.exists():
        print("no Indiana cell file"); return
    t = pd.read_csv(p, dtype={"ccn": str})
    t = t[t.grp == "spine"].rename(columns={"MSDRG": "drg"})
    t["ccn"] = t.ccn.str.zfill(6)
    t.to_csv(OUT / "indiana_truth.csv", index=False)
    known = t[t.cms_ffs.isna() & t.state_medicare.notna()]
    print("Indiana spine cells:", len(t), "| suppressed cells with state count:", len(known),
          "| state count distribution in suppressed cells:",
          known.state_medicare.describe()[["mean", "50%", "max"]].round(2).to_dict())


if __name__ == "__main__":
    build()
    indiana_truth()
