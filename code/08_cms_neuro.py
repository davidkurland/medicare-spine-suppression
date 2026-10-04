"""National Medicare FFS neurosurgical panel from CMS 'Medicare Inpatient Hospitals by Provider and Service'.

Hospital (CCN) x year x neurosurgical category, 2013-2024, plus state summaries and an estimate of
volume hidden by small-cell suppression (DRG rows with < 11 discharges are not published).
Outputs: data/interim/cms_neuro_hospital_year.parquet, cms_neuro_state_year.csv, cms_neuro_suppression.csv
"""
import re

import pandas as pd

from common import DRG_TO_CAT, MSDRG, MSDRG_GROUP, ROOT

COLS = {"Rndrng_Prvdr_CCN": "ccn", "Rndrng_Prvdr_Org_Name": "name", "Rndrng_Prvdr_State_Abrvtn": "state",
        "DRG_Cd": "drg", "Tot_Dschrgs": "n", "Avg_Tot_Pymt_Amt": "avg_pay"}


def load() -> pd.DataFrame:
    frames = []
    # files mix UTF-8 and cp1252; latin-1 never fails and only garbles a few hospital names
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1", usecols=list(COLS)).rename(columns=COLS)
        d["year"] = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["ccn"] = d.ccn.str.zfill(6)
    d["drg"] = pd.to_numeric(d.drg, errors="coerce")
    d["n"] = pd.to_numeric(d.n, errors="coerce")
    d["avg_pay"] = pd.to_numeric(d.avg_pay, errors="coerce")
    return d


if __name__ == "__main__":
    mup = load()
    mup["category"] = mup.drg.map(DRG_TO_CAT)
    neuro = mup.dropna(subset=["category"])
    neuro = neuro.assign(pay_total=neuro.n * neuro.avg_pay)
    # DRG-level rows let state data be compared only in cells CMS actually publishes (>= 11 discharges)
    neuro[["year", "ccn", "state", "drg", "category", "n"]].to_parquet(
        ROOT / "data/interim/cms_neuro_drg_rows.parquet", index=False)
    hy = (neuro.groupby(["year", "ccn", "category"])
               .agg(name=("name", "first"), state=("state", "first"), discharges=("n", "sum"),
                    drg_rows=("drg", "nunique"), pay_total=("pay_total", "sum"))
               .reset_index())
    hy["avg_pay"] = hy.pay_total / hy.discharges
    hy["group"] = hy.category.map(MSDRG_GROUP)
    hy.to_parquet(ROOT / "data/interim/cms_neuro_hospital_year.parquet", index=False)

    sy = (hy.groupby(["state", "year", "group", "category"])
            .agg(discharges=("discharges", "sum"), hospitals=("ccn", "nunique")).reset_index())
    sy.to_csv(ROOT / "data/interim/cms_neuro_state_year.csv", index=False)

    # Suppression bound: any hospital in the file that year may have 1-10 discharges in a DRG it does not show.
    hosp_year = mup.groupby("year").ccn.nunique()
    shown_cells = neuro.groupby(["year", "category"]).size()
    n_drgs = pd.Series({k: len(v) for k, v in MSDRG.items()})
    sup = shown_cells.rename("shown_cells").reset_index()
    sup["possible_cells"] = sup.apply(lambda r: hosp_year[r.year] * n_drgs[r.category], axis=1)
    sup["hidden_cells"] = sup.possible_cells - sup.shown_cells
    shown = hy.groupby(["year", "category"]).agg(shown=("discharges", "sum"), hospitals_shown=("ccn", "nunique"))
    sup = sup.merge(shown.reset_index(), on=["year", "category"])
    sup["group"] = sup.category.map(MSDRG_GROUP)
    sup["ipps_hospitals"] = sup.year.map(hosp_year)
    sup.to_csv(ROOT / "data/interim/cms_neuro_suppression.csv", index=False)

    pd.set_option("display.width", 220)
    print(hy.groupby(["year", "category"]).discharges.sum().unstack().to_string())
    print(sup.groupby(["year", "group"])[["shown", "hospitals_shown", "ipps_hospitals"]].agg(
        {"shown": "sum", "hospitals_shown": "max", "ipps_hospitals": "first"}).unstack().to_string())
