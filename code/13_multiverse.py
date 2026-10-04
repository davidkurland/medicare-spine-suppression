"""Multiverse analysis: how much do two standard neurosurgical results move under defensible analytic choices?

A. Volume trend (percent change in inpatient cases between two years), cranial vs spine:
   SPARCS all-payer under 4 case definitions x 2 age scopes x 3 windows, plus the same windows in
   CMS Medicare FFS (NY and national) as alternative *sources*.
B. Black-White difference in length of stay (log LOS, % difference), cranial vs spine, across
   definition x ethnicity scope x adults-only x LOS trimming x hospital race-quality restriction x
   year window x adjustment set (unadjusted / case-mix / case-mix + hospital-year fixed effects).
Outputs: data/interim/multiverse_trend.csv, data/interim/multiverse_los.csv
"""
import itertools
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
I = ROOT / "data/interim"

LOAD = f"""
WITH b AS (
  SELECT year, pfi, race, ethnicity, age_group, sex, soi, pay1, los, admit_type,
         TRY_CAST(apr_drg AS INT) AS drg,
         CASE WHEN dx_system = 'CCS' THEN ltrim(proc_code, '0') ELSE proc_code END AS pc,
         dx_system
  FROM read_parquet('{ROOT}/data/interim/sparcs/*/*.parquet', union_by_name=true))
SELECT *,
  CASE WHEN drg IN (20,21,22) THEN 'cranial' WHEN drg IN (23,303,304,310,321) THEN 'spine' END AS drg_grp,
  CASE WHEN dx_system='CCS' AND pc IN ('1','2') THEN 'cranial'
       WHEN dx_system='CCS' AND pc IN ('3','158') THEN 'spine'
       WHEN dx_system='CCSR' AND pc IN ('CNS004','CNS005','CNS007','CNS009','CNS010') THEN 'cranial'
       WHEN dx_system='CCSR' AND pc IN ('MST013','MST016','CNS008','PNS001','PNS002') THEN 'spine' END AS proc_grp
FROM b
WHERE drg IN (20,21,22,23,303,304,310,321)
   OR pc IN ('1','2','3','158','CNS004','CNS005','CNS007','CNS009','CNS010','MST013','MST016','CNS008','PNS001','PNS002')
"""
DEFS = {
    "APR-DRG": lambda d, g: d.drg_grp == g,
    "Procedure": lambda d, g: d.proc_grp == g,
    "Both": lambda d, g: (d.drg_grp == g) & (d.proc_grp == g),
    "Either": lambda d, g: (d.drg_grp == g) | (d.proc_grp == g),
}


def trend(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    windows = [(2013, 2024), (2016, 2024), (2019, 2024)]
    for g, (dn, df_), adults, (y0, y1) in itertools.product(["cranial", "spine"], DEFS.items(), [False, True], windows):
        m = df_(d, g) & (~d.age_group.eq("0 to 17") if adults else True)
        cnt = d.loc[m].groupby("year").size()
        rows.append({"grp": g, "source": "SPARCS all-payer", "definition": dn, "adults_only": adults,
                     "window": f"{y0}-{y1}", "change": cnt.get(y1, np.nan) / cnt.get(y0, np.nan) - 1})
    # SPARCS Medicare payer only, to line up with CMS
    for g, (dn, df_), (y0, y1) in itertools.product(["cranial", "spine"], DEFS.items(), windows):
        m = df_(d, g) & d.pay1.str.match("(?i)^medicare", na=False)
        cnt = d.loc[m].groupby("year").size()
        rows.append({"grp": g, "source": "SPARCS Medicare payer", "definition": dn, "adults_only": True,
                     "window": f"{y0}-{y1}", "change": cnt.get(y1, np.nan) / cnt.get(y0, np.nan) - 1})
    cms = pd.read_parquet(I / "cms_neuro_hospital_year.parquet")
    cms = cms[cms.group.isin(["cranial", "spine"])]
    for scope, sub in [("CMS Medicare FFS, NY", cms[cms.state == "NY"]), ("CMS Medicare FFS, US", cms)]:
        tot = sub.groupby(["group", "year"]).discharges.sum()
        for g, (y0, y1) in itertools.product(["cranial", "spine"], windows):
            rows.append({"grp": g, "source": scope, "definition": "MS-DRG", "adults_only": True,
                         "window": f"{y0}-{y1}", "change": tot[(g, y1)] / tot[(g, y0)] - 1})
    return pd.DataFrame(rows)


def within(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """OLS with absorbed group fixed effects via demeaning."""
    codes, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    def dm(v):
        s = np.bincount(inv, weights=v, minlength=len(codes))
        return v - (s / cnt)[inv]
    yd = dm(y)
    Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    keep = Xd.std(axis=0) > 0
    beta, *_ = np.linalg.lstsq(Xd[:, keep], yd, rcond=None)
    out = np.full(X.shape[1], np.nan)
    out[keep] = beta
    return out


def los_gap(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.race.isin(["White", "Black/African American"]) & d.los.notna() & d.pfi.notna()].copy()
    d["black"] = (d.race == "Black/African American").astype(float)
    d["ylog"] = np.log1p(d.los.astype(float))
    # hospital race-data quality (share Other/Unknown race among all its discharges in the cohort file)
    q = pd.read_parquet(I / "sparcs_quality_by_facility.parquet")
    q["bad_race"] = q.race_other + q.race_unknown
    d = d.merge(q[["year", "pfi", "bad_race"]], on=["year", "pfi"], how="left")
    case_mix = pd.get_dummies(d[["age_group", "sex", "soi", "pay1"]].astype(str), drop_first=True).astype(float)
    yr = pd.get_dummies(d.year.astype(str), prefix="y", drop_first=True).astype(float)
    CM = np.column_stack([case_mix.to_numpy(), yr.to_numpy()])
    hosp_year = (d.pfi.astype(int) * 10000 + d.year).to_numpy()
    rows = []
    grid = itertools.product(["cranial", "spine"], DEFS.items(), [False, True], [False, True], [False, True],
                             [False, True], [(2009, 2024), (2016, 2024)], ["Unadjusted", "Case mix", "Case mix + hospital FE"])
    for g, (dn, df_), nonhisp, adults, trim, goodrace, (y0, y1), adj in grid:
        m = df_(d, g) & d.year.between(y0, y1)
        if nonhisp:
            m &= d.ethnicity.str.contains("Not Span", na=False)
        if adults:
            m &= d.age_group.ne("0 to 17")
        if goodrace:
            m &= d.bad_race < 0.25
        if trim:
            cut = d.loc[m, "los"].quantile(0.99)
            m &= d.los <= cut
        idx = np.flatnonzero(m.to_numpy())
        if len(idx) < 500:
            continue
        y = d.ylog.to_numpy()[idx]
        if adj == "Unadjusted":
            X = d.black.to_numpy()[idx, None]
            b = within(y, X, np.zeros(len(idx)))[0]
        elif adj == "Case mix":
            X = np.column_stack([d.black.to_numpy()[idx], CM[idx]])
            b = within(y, X, np.zeros(len(idx)))[0]
        else:
            X = np.column_stack([d.black.to_numpy()[idx], CM[idx]])
            b = within(y, X, hosp_year[idx])[0]
        rows.append({"grp": g, "definition": dn, "non_hispanic": nonhisp, "adults_only": adults, "trim_los_p99": trim,
                     "good_race_hospitals": goodrace, "window": f"{y0}-{y1}", "adjustment": adj, "n": len(idx),
                     "pct_longer_black": float(np.expm1(b))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = duckdb.connect().execute(LOAD).df()
    t = trend(d)
    t.to_csv(I / "multiverse_trend.csv", index=False)
    print(t.groupby(["grp", "source"]).change.describe().round(3).to_string())
    l = los_gap(d)
    l.to_csv(I / "multiverse_los.csv", index=False)
    print(l.groupby(["grp", "adjustment"]).pct_longer_black.describe().round(3).to_string())
    print(l.groupby(["grp", "definition"]).pct_longer_black.median().round(3).to_string())
