"""Indiana hospital rank concordance for lower extremity joint replacement (eFigure: linkage sensitivity).

Same construction as the spine and cranial series (10_indiana_neuro.py, 17_supplement_figures.py): state Medicare inpatients
(FFS from 2021, any Medicare before) by CCN and MS-DRG against published CMS FFS rows, restricted to published cells,
summed to the hospital, Spearman rho by year; once with the hand-verified crosswalk, once with the automated one.
The spine series is recomputed the same way as a check on the method.
Output: data/interim/state_compare/IN_joint_rank_concordance.csv
"""
import importlib
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ROOT

inn = importlib.import_module("10_indiana_neuro")
I = ROOT / "data/interim"
SC = I / "state_compare"
JOINT = [469, 470, 521, 522]


def cms_cells(drgs):
    rows = []
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1", low_memory=False)
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        pick = lambda key: next(c for c in d.columns if c.lower() == key.lower())
        d = d[[pick("Rndrng_Prvdr_CCN"), pick("Rndrng_Prvdr_State_Abrvtn"), pick("DRG_Cd"), pick("Tot_Dschrgs")]]
        d.columns = ["ccn", "state", "MSDRG", "cms_ffs"]
        d["MSDRG"] = pd.to_numeric(d.MSDRG, errors="coerce"); d["cms_ffs"] = pd.to_numeric(d.cms_ffs, errors="coerce")
        d = d[(d.state == "IN") & d.MSDRG.isin(drgs)]
        rows.append(d.assign(year=y, ccn=d.ccn.str.zfill(6)).groupby(["year", "ccn", "MSDRG"]).cms_ffs.sum())
    return pd.concat(rows)


def concordance(med, cms, xw):
    x = xw[xw.match_ok == True][["id", "ccn"]].assign(ccn=lambda t: t.ccn.astype(str).str.zfill(6))
    st = med.merge(x, on="id").groupby(["year", "ccn", "MSDRG"]).PATS.sum().rename("state_medicare")
    cells = pd.concat([st, cms], axis=1).reset_index()
    out = {}
    for y, d in cells[cells.year >= 2017].groupby("year"):
        hp = d.dropna(subset=["cms_ffs"]).fillna({"state_medicare": 0}).groupby("ccn")[["cms_ffs", "state_medicare"]].sum()
        out[y] = spearmanr(hp.cms_ffs, hp.state_medicare).statistic
    return pd.Series(out)


if __name__ == "__main__":
    ms = inn.load("MSDRG")
    ms["ffs_like"] = np.where(ms.year >= 2021, ms.payer == "medicare_ffs", ms.payer.isin(["medicare_ffs", "medicare_any", "medicare_adv"]))
    ms = ms[ms.ffs_like].rename(columns={"HOSPITAL_ID": "id"})
    ver = pd.read_csv(SC / "IN_crosswalk.csv", dtype={"ccn": str})
    auto = pd.read_csv(ROOT / "data/manual/IN_crosswalk_automated.csv", dtype={"ccn": str})
    j = ms[ms.MSDRG.isin(JOINT)]
    cj = cms_cells(JOINT)
    res = pd.DataFrame({"rho": concordance(j, cj, ver), "rho_hospital_automated": concordance(j, cj, auto)})
    res = res.rename_axis("year").reset_index().assign(grp="joint replacement")
    res.to_csv(SC / "IN_joint_rank_concordance.csv", index=False)
    print(res.round(3).to_string(index=False))
    # method check: spine automated series should match data/manual/IN_rank_agreement_automated_crosswalk.csv
    from common import DRG_TO_CAT, MSDRG_GROUP
    sp_drgs = [d for d, c in DRG_TO_CAT.items() if MSDRG_GROUP.get(c) == "spine"]
    s = ms[ms.MSDRG.isin(sp_drgs)]
    chk = concordance(s, cms_cells(sp_drgs), auto).rename("recomputed")
    ref = pd.read_csv(ROOT / "data/manual/IN_rank_agreement_automated_crosswalk.csv").query("grp == 'spine'").set_index("year").rho_hospital_automated
    print(pd.concat([chk, ref], axis=1).round(3))
