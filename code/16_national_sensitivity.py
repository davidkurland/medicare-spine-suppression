"""National Medicare FFS neurosurgical trends: how much of the 'decline' survives adjustment?

1. Raw published discharges (CMS MUP inpatient by provider & service), cranial vs spine.
2. Per 100,000 traditional Medicare Part A beneficiaries (CMS Medicare Monthly Enrollment, annual rows),
   removing the shift of beneficiaries into Medicare Advantage.
3. Balanced panel: hospitals with a published row in the group every year 2013-2024
   (removes hospitals entering/leaving publication through suppression or closure).
4. Excluding FY2025 fusion MS-DRGs (what a pre-2025 code list would report for 2024).
5. State ecological association: change in Medicare Advantage share vs change in FFS discharges, 2016-2023.
Outputs: data/interim/national_sensitivity.csv, data/interim/state_ma_vs_ffs.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ROOT

I = ROOT / "data/interim"
FY25 = {402, 426, 427, 428, 429, 430, 447, 448, 450, 451}


def enrollment() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "data/raw/cms_enrollment/medicare_monthly_enrollment.csv", dtype=str,
                    usecols=["YEAR", "MONTH", "BENE_GEO_LVL", "BENE_STATE_ABRVTN", "TOT_BENES",
                             "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"])
    e = e[(e.MONTH == "Year") & e.BENE_GEO_LVL.isin(["National", "State"])].copy()
    for c in ["TOT_BENES", "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e["year"] = e.YEAR.astype(int)
    e["state"] = np.where(e.BENE_GEO_LVL == "National", "US", e.BENE_STATE_ABRVTN)
    e["ma_share"] = e.A_MA_AND_OTH_BENES / (e.A_MA_AND_OTH_BENES + e.A_ORGNL_MDCR_BENES)
    return e[["year", "state", "TOT_BENES", "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES", "ma_share"]]


if __name__ == "__main__":
    enr = enrollment()
    hy = pd.read_parquet(I / "cms_neuro_hospital_year.parquet")
    hy = hy[hy.group.isin(["cranial", "spine"])]
    rows = pd.read_parquet(I / "cms_neuro_drg_rows.parquet")
    rows = rows[rows.category.str.startswith(("cranial", "spine"))]
    rows["grp"] = np.where(rows.category.str.startswith("cranial"), "cranial", "spine")

    nat = hy.groupby(["year", "group"]).discharges.sum().unstack()
    us = enr[enr.state == "US"].set_index("year")
    out = pd.DataFrame(index=nat.index)
    for g in ["cranial", "spine"]:
        out[f"{g}_raw"] = nat[g]
        out[f"{g}_per100k_ffs"] = nat[g] / us.A_ORGNL_MDCR_BENES.reindex(nat.index) * 1e5
        hosp_years = hy[hy.group == g].groupby("ccn").year.nunique()
        keep = hosp_years[hosp_years == hy.year.nunique()].index
        out[f"{g}_balanced"] = hy[(hy.group == g) & hy.ccn.isin(keep)].groupby("year").discharges.sum()
        out[f"{g}_balanced_hospitals"] = len(keep)
        out[f"{g}_hospitals_published"] = hy[hy.group == g].groupby("year").ccn.nunique()
    out["spine_no_fy25_drgs"] = rows[(rows.grp == "spine") & ~rows.drg.isin(FY25)].groupby("year").n.sum()
    out["ma_share_us"] = us.ma_share.reindex(out.index)
    out["ffs_part_a_benes"] = us.A_ORGNL_MDCR_BENES.reindex(out.index)
    out = out.reset_index()
    out.to_csv(I / "national_sensitivity.csv", index=False)

    base, last = out.iloc[0], out.iloc[-1]
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    for c in [c for c in out.columns if c.startswith(("cranial", "spine")) and "hospitals" not in c]:
        print(f"{c:28s} {base[c]:>10.1f} -> {last[c]:>10.1f}  ({last[c] / base[c] - 1:+.1%})")

    # state ecological association, 2016 -> 2023
    st = hy.groupby(["state", "group", "year"]).discharges.sum().unstack()
    se = enr[enr.state != "US"].pivot(index="state", columns="year", values="ma_share")
    sf = enr[enr.state != "US"].pivot(index="state", columns="year", values="A_ORGNL_MDCR_BENES")
    rows_out = []
    for g in ["cranial", "spine"]:
        d = st.xs(g, level="group")
        df = pd.DataFrame({"ffs_change": d[2023] / d[2016] - 1,
                           "rate_change": (d[2023] / sf[2023]) / (d[2016] / sf[2016]) - 1,
                           "ma_share_change_pts": (se[2023] - se[2016]) * 100}).dropna()
        df = df[d.loc[df.index, 2016] >= 100]
        df["grp"] = g
        rows_out.append(df.reset_index())
        rho = spearmanr(df.ma_share_change_pts, df.ffs_change)
        print(f"{g}: states={len(df)} rho(MA share change, FFS discharge change)={rho.statistic:.2f} p={rho.pvalue:.3f}")
    pd.concat(rows_out).to_csv(I / "state_ma_vs_ffs.csv", index=False)
