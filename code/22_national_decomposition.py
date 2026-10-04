"""National decomposition of the change in published Medicare FFS neurosurgical discharges.

Using exact FFS totals (geography file) and traditional Medicare Part A enrollment:
  log(published_t / published_0) = log(total_t / total_0)            [all FFS discharges]
                                  + log(share_published_t / share_0) [suppression component]
  log(total_t / total_0)          = log(FFS enrollees_t / _0)         [enrollment component]
                                  + log(rate_t / rate_0)              [per-enrollee change: migration, practice, selection]
Reported for groups and spine families and for 2013-2024, 2013-2019, 2019-2023, 2023-2024.
Also: FY2025 fragmentation test (published cells and hidden share by family, 2022-2024);
Poisson trend models on exact totals with enrollment offset and COVID indicator; state-year panel of
FFS discharges per FFS enrollee on MA share with state and year fixed effects (published vs state totals);
Maryland and Puerto Rico are excluded from the panel.
Outputs: data/interim/national_decomposition.csv, fy2025_fragmentation.csv, national_trend_models.csv, state_panel.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from common import ROOT

I = ROOT / "data/interim"


def enrollment() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "data/raw/cms_enrollment/medicare_monthly_enrollment.csv", dtype=str,
                    usecols=["YEAR", "MONTH", "BENE_GEO_LVL", "BENE_STATE_ABRVTN", "TOT_BENES", "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"])
    e = e[(e.MONTH == "Year") & e.BENE_GEO_LVL.isin(["National", "State"])].copy()
    for c in ["TOT_BENES", "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e["year"] = e.YEAR.astype(int)
    e["st"] = np.where(e.BENE_GEO_LVL == "National", "US", e.BENE_STATE_ABRVTN)
    e["ma_share"] = e.A_MA_AND_OTH_BENES / (e.A_MA_AND_OTH_BENES + e.A_ORGNL_MDCR_BENES)
    return e


def decompose(d, y0, y1):
    a, b = d.loc[y0], d.loc[y1]
    lp = np.log(b.published / a.published)
    lt = np.log(b.total_ffs / a.total_ffs)
    le = np.log(b.ffs_benes / a.ffs_benes)
    return {"window": f"{y0}-{y1}", "published_change": np.expm1(lp), "total_change": np.expm1(lt),
            "rate_change": np.expm1(lt - le), "log_published": lp,
            "suppression_log": lp - lt, "enrollment_log": le, "rate_log": lt - le,
            "suppression_pct_of_change": (lp - lt) / lp * 100, "enrollment_pct_of_change": le / lp * 100,
            "rate_pct_of_change": (lt - le) / lp * 100,
            "hidden_share_start": 1 - a.published / a.total_ffs, "hidden_share_end": 1 - b.published / b.total_ffs}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    nat = pd.read_csv(I / "suppression_exact_national.csv")
    enr = enrollment()
    us = enr[enr.st == "US"].set_index("year")
    rows = []
    for (lvl, s), d in nat.groupby(["level", "stratum"]):
        d = d.set_index("year").join(us.A_ORGNL_MDCR_BENES.rename("ffs_benes"))
        for y0, y1 in [(2013, 2024), (2013, 2023), (2013, 2019), (2019, 2023), (2023, 2024)]:
            rows.append({"level": lvl, "stratum": s, **decompose(d, y0, y1)})
    dec = pd.DataFrame(rows)
    dec.to_csv(I / "national_decomposition.csv", index=False)
    show = dec[dec.stratum.isin(["spine", "cranial", "cervical_fusion", "thoracolumbar_fusion", "deformity_fusion", "spine_nonfusion",
                                 "cranial_open_other", "cranial_thrombectomy_group", "cranial_shunt"])]
    print(show[["stratum", "window", "published_change", "total_change", "rate_change", "suppression_pct_of_change",
                "enrollment_pct_of_change", "rate_pct_of_change", "hidden_share_start", "hidden_share_end"]].round(3).to_string(index=False))

    # FY2025 fragmentation: published cells, published and hidden volume by family, 2022-2024
    # (family map is the same as FAMILY in 21_suppression_exact.py)
    r = pd.read_parquet(I / "cms_neuro_drg_rows.parquet")
    fam = {**{d: "cervical_fusion" for d in (471, 472, 473, 429, 430)}, **{d: "deformity_fusion" for d in (456, 457, 458)},
           **{d: "thoracolumbar_fusion" for d in (453, 454, 455, 459, 460, 402, 426, 427, 428, 447, 448, 450, 451)},
           **{d: "spine_nonfusion" for d in (28, 29, 30, 490, 491, 518, 519, 520)},
           **{d: "cranial_thrombectomy_group" for d in (23, 24)}, **{d: "cranial_open_other" for d in (20, 21, 22, 25, 26, 27)},
           **{d: "cranial_shunt" for d in (31, 32, 33)}}
    r["family"] = r.drg.map(fam)
    cells = r.groupby(["year", "family"]).agg(published_cells=("n", "size"), hospitals=("ccn", "nunique"),
                                              median_cell=("n", "median"), drgs=("drg", "nunique")).reset_index()
    fr = cells.merge(nat[nat.level == "family"].rename(columns={"stratum": "family"}), on=["year", "family"])
    fr = fr[fr.year.isin([2022, 2023, 2024])]
    fr.to_csv(I / "fy2025_fragmentation.csv", index=False)
    print(fr[["family", "year", "drgs", "published_cells", "hospitals", "median_cell", "total_ffs", "published", "hidden_share"]].round(3).to_string(index=False))

    # Poisson trend models on exact totals (and published) with FFS-enrollment offset and COVID indicator
    tm = []
    for grp in ["spine", "cranial"]:
        d = nat[(nat.level == "group") & (nat.stratum == grp)].set_index("year").join(us.A_ORGNL_MDCR_BENES.rename("ffs_benes")).reset_index()
        d["covid"] = d.year.isin([2020, 2021]).astype(int)
        d["t"] = d.year - 2013
        for outcome in ["total_ffs", "published"]:
            for offset in [True, False]:
                m = smf.glm(f"{outcome} ~ t + covid", data=d, family=sm.families.Poisson(),
                            offset=np.log(d.ffs_benes) if offset else None).fit(scale="X2")   # quasi-Poisson SEs
                b, se = m.params["t"], m.bse["t"]
                tm.append({"group": grp, "outcome": outcome, "per_enrollee": offset,
                           "annual_pct_change": np.expm1(b) * 100, "lo": np.expm1(b - 1.96 * se) * 100, "hi": np.expm1(b + 1.96 * se) * 100})
    tm = pd.DataFrame(tm)
    tm.to_csv(I / "national_trend_models.csv", index=False)
    print(tm.round(2).to_string(index=False))

    # State-year panel: FFS discharges per FFS enrollee on MA share, state + year FE, enrollment-weighted
    st = pd.read_csv(I / "suppression_exact_state.csv")
    se = enr[enr.st != "US"][["year", "st", "A_ORGNL_MDCR_BENES", "ma_share"]]
    p = st.merge(se, on=["year", "st"])
    # Maryland (hospitals paid under the state all-payer rate-setting model, not IPPS) and Puerto Rico are left out of the panel
    p = p[(p.st != "MD") & (p.st != "PR")]
    out = []
    for grp in ["spine", "cranial"]:
        d = p[(p.group == grp) & (p.state_total_published_cells > 0)].copy()
        for y in ["state_total_published_cells", "provider_published"]:
            d = d[d[y] > 0]
            d["lograte"] = np.log(d[y] / d.A_ORGNL_MDCR_BENES * 1e5)
            m = smf.wls("lograte ~ ma_share + C(st) + C(year)", data=d, weights=d.A_ORGNL_MDCR_BENES).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d.st)[0]})
            b, se_ = m.params["ma_share"], m.bse["ma_share"]
            out.append({"group": grp, "outcome": y, "n_state_years": len(d),
                        "pct_change_per_10pt_MA": np.expm1(b * 0.10) * 100,
                        "lo": np.expm1((b - 1.96 * se_) * 0.10) * 100, "hi": np.expm1((b + 1.96 * se_) * 0.10) * 100, "p": m.pvalues["ma_share"]})
    panel = pd.DataFrame(out)
    panel.to_csv(I / "state_panel.csv", index=False)
    print(panel.round(3).to_string(index=False))
