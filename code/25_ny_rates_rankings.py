"""New York analyses: per-beneficiary rates, payer proxy, payer-matched rank concordance, and matched-window trends.

1. Per-beneficiary rates 2013-2024: SPARCS Medicare-payer spine/cranial per 100 000 NY Medicare beneficiaries
   (TOT_BENES) and CMS NY FFS per 100 000 NY FFS Part A beneficiaries (A_ORGNL_MDCR_BENES); changes over
   2013-2024, 2013-2019, 2022-2024.
2. Medicare-age proxy: SPARCS 70+ counts by payer category, share of 70+ patients with a non-Medicare primary payer
   (possible MA coded under the carrier), and the share of neurosurgical records with a redacted facility ID.
3. Payer-matched rank concordance with the hospital units of 14_rank_shuffle.py (crosswalk + consolidation):
   rho (a) all-payer vs CMS FFS, (b) Medicare-payer vs CMS FFS, (c) all-payer vs Medicare-payer,
   (d) = (b) among hospitals published by CMS; hospital bootstrap CIs; bootstrap spine-cranial difference in (b);
   county Medicare Advantage share (county enrollment rows) and its association with rank drops.
4. Matched-window trend specifications: every source over identical windows, source comparison kept separate
   from case-definition specifications, age scope labeled explicitly (all ages / adults / 70+).
5. Trend models 2013-2024: Poisson, quasi-Poisson and negative binomial on annual counts with a continuous year
   term, a 2020-2021 COVID indicator and log enrollment offsets; annual percentage change with 95% CI.
6. Figure 3 support: top-15 hospitals by SPARCS Medicare-payer volume in 2023 and 2024.
Case definition throughout (unless stated): APR-DRG cranial 020-022; spine 023/303/304/310/321; CMS MS-DRG groups
from common.MSDRG. CMS data years are calendar years.
Outputs: data/interim/ny_rates_rankings/*.csv and SUMMARY.md
"""
import itertools

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from common import ROOT

I = ROOT / "data/interim"
OUT = I / "ny_rates_rankings"
RNG = np.random.default_rng(20260915)   # fixed seed for reproducible bootstrap intervals
NBOOT = 2000
YEARS = range(2013, 2025)
SRC = f"read_parquet('{ROOT}/data/interim/sparcs/*/*.parquet', union_by_name=true)"

# Neurosurgical cohort with both case definitions (same SQL logic as 09_ny_neuro.py / 13_multiverse.py).
# pfi is cast to BIGINT: DuckDB's perfect-hash aggregate trips over the parquet statistics on the INTEGER column.
LOAD = f"""
WITH b AS (
  SELECT year, CAST(pfi AS BIGINT) AS pfi, facility_name, hospital_county, age_group, pay1,
         TRY_CAST(apr_drg AS INT) AS drg,
         CASE WHEN dx_system = 'CCS' THEN ltrim(proc_code, '0') ELSE proc_code END AS pc, dx_system
  FROM {SRC} WHERE year BETWEEN 2013 AND 2024)
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
# SPARCS Payment Typology 1 -> analysis category. "Commercial" = private, BCBS, managed care unspecified
# (where MA sold by a commercial carrier may be coded); Misc/Other kept apart because it mixes no-fault/workers'
# comp with unclassified plans.
PAYER_CAT = {"Medicare": "medicare", "Medicaid": "medicaid", "Private Health Insurance": "commercial",
             "Blue Cross/Blue Shield": "commercial", "Managed Care, Unspecified": "commercial",
             "Miscellaneous/Other": "misc_other"}
AGES = {"all ages": lambda d: pd.Series(True, index=d.index), "adults 18+": lambda d: d.age_group.ne("0 to 17"),
        "70+": lambda d: d.age_group.eq("70 or Older")}


def pct(a, b):
    return (b / a - 1) * 100


def enrollment():
    """NY state and county annual ('Year' rows) Medicare enrollment."""
    e = pd.read_csv(ROOT / "data/raw/cms_enrollment/medicare_monthly_enrollment.csv", dtype=str)
    e = e[(e.MONTH == "Year") & (e.BENE_STATE_ABRVTN == "NY")].copy()
    num = ["TOT_BENES", "A_TOT_BENES", "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"]
    e[num] = e[num].apply(pd.to_numeric, errors="coerce")
    e["year"] = e.YEAR.astype(int)
    st = e[e.BENE_GEO_LVL == "State"].set_index("year")[num]
    co = e[(e.BENE_GEO_LVL == "County") & (e.BENE_COUNTY_DESC != "Unknown")].copy()
    co["county"] = co.BENE_COUNTY_DESC.str.replace(" County", "", regex=False)
    # MA share of Part A enrollees ("MA and other" includes cost plans, PACE, demonstrations)
    co["county_ma_share"] = co.A_MA_AND_OTH_BENES / co.A_TOT_BENES
    return st, co[["year", "county", "county_ma_share"]]


def norm_county(s: pd.Series) -> pd.Series:
    """SPARCS hospital_county spellings -> CMS enrollment county names."""
    return s.replace({"Manhattan": "New York", "St Lawrence": "St. Lawrence", "Saint Lawrence": "St. Lawrence"})


def cms_ny():
    cms = pd.read_parquet(I / "cms_neuro_hospital_year.parquet")
    return cms[(cms.state == "NY") & cms.group.isin(["cranial", "spine"])]


# ---------------------------------------------------------------- 1. rates
def rates(d, st):
    sp = d[d.drg_grp.notna()]
    med = sp[sp.pay1.str.match("(?i)^medicare", na=False)].groupby(["drg_grp", "year"]).size()
    allp = sp.groupby(["drg_grp", "year"]).size()
    cms = cms_ny().groupby(["group", "year"]).discharges.sum()
    rows = []
    for g, y in itertools.product(["spine", "cranial"], YEARS):
        rows.append({"grp": g, "year": y, "sparcs_all_payer": allp[(g, y)], "sparcs_medicare": med[(g, y)],
                     "cms_ffs_ny": cms[(g, y)], "ny_total_benes": st.loc[y, "TOT_BENES"],
                     "ny_ffs_part_a_benes": st.loc[y, "A_ORGNL_MDCR_BENES"],
                     "ny_ma_share_part_a": st.loc[y, "A_MA_AND_OTH_BENES"] / st.loc[y, "A_TOT_BENES"]})
    r = pd.DataFrame(rows)
    r["sparcs_medicare_per100k_benes"] = r.sparcs_medicare / r.ny_total_benes * 1e5
    r["cms_ffs_per100k_ffs_benes"] = r.cms_ffs_ny / r.ny_ffs_part_a_benes * 1e5
    ch = []
    cols = ["sparcs_all_payer", "sparcs_medicare", "cms_ffs_ny", "ny_total_benes", "ny_ffs_part_a_benes",
            "sparcs_medicare_per100k_benes", "cms_ffs_per100k_ffs_benes"]
    for g, x in r.groupby("grp"):
        x = x.set_index("year")
        for y0, y1 in [(2013, 2024), (2013, 2019), (2022, 2024), (2019, 2024)]:
            ch.append({"grp": g, "window": f"{y0}-{y1}", **{c: round(pct(x.loc[y0, c], x.loc[y1, c]), 1) for c in cols}})
    return r, pd.DataFrame(ch)


# ---------------------------------------------------------------- 2. age 70+ payer mix, redaction
def age70_payer(d):
    sp = d[d.drg_grp.notna()].copy()
    sp["pcat"] = sp.pay1.map(PAYER_CAT).fillna("other")
    a = sp[sp.age_group == "70 or Older"].groupby(["drg_grp", "year", "pcat"]).size().unstack(fill_value=0)
    a = a.reindex(columns=["medicare", "commercial", "medicaid", "misc_other", "other"], fill_value=0)
    a["total_70plus"] = a.sum(axis=1)
    a["share_non_medicare"] = 1 - a.medicare / a.total_70plus
    a["share_commercial"] = a.commercial / a.total_70plus
    a["share_commercial_or_misc"] = (a.commercial + a.misc_other) / a.total_70plus   # possible MA coded under a commercial carrier
    red = sp.groupby(["drg_grp", "year"]).agg(records=("pfi", "size"), pfi_redacted=("pfi", lambda s: s.isna().sum()))
    red["share_pfi_redacted"] = red.pfi_redacted / red.records
    return a.join(red).reset_index().rename(columns={"drg_grp": "grp"})


# ---------------------------------------------------------------- 3. rank concordance
def hospital_units(d):
    """SPARCS PFI -> analysis unit exactly as 14_rank_shuffle.py (CCN via crosswalk, then manual consolidation)."""
    sp = d[d.drg_grp.notna() & d.pfi.notna()].copy()
    sp["medicare"] = sp.pay1.str.match("(?i)^medicare", na=False).astype(int)
    sp["county"] = norm_county(sp.hospital_county)
    sp = sp.groupby(["year", "pfi", "drg_grp"]).agg(
        facility_name=("facility_name", "first"), county=("county", "first"),
        all_payer=("medicare", "size"), medicare=("medicare", "sum")).reset_index().rename(columns={"drg_grp": "grp"})
    xw = pd.read_csv(I / "ny_pfi_ccn_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok == True].drop_duplicates("pfi")[["pfi", "ccn"]]
    sp = sp.merge(xw, on="pfi", how="left")
    sp["unit"] = sp.ccn.fillna("pfi" + sp.pfi.astype(int).astype(str))
    cons = pd.read_csv(ROOT / "data/manual/NY_ccn_consolidation.csv", dtype=str)
    for r in cons.itertuples():
        m = (sp.unit == r.from_unit) & (sp.year >= int(r.start_year))
        sp.loc[m, "unit"] = r.to_ccn
    return sp


def rank_table(sp, co):
    name = sp.sort_values("all_payer").groupby("unit").facility_name.last()
    sp = sp.merge(co, on=["year", "county"], how="left")
    # unit county MA share = case-weighted over the unit's PFIs (consolidated systems span counties)
    sp["w_ma"] = sp.county_ma_share * sp.all_payer
    agg = sp.groupby(["year", "grp", "unit"]).agg(all_payer=("all_payer", "sum"), medicare=("medicare", "sum"),
                                                  w_ma=("w_ma", "sum")).reset_index()
    agg["county_ma_share"] = agg.pop("w_ma") / agg.all_payer
    main_cty = sp.groupby(["unit", "county"]).all_payer.sum().reset_index().sort_values("all_payer") \
                 .groupby("unit").county.last()
    cms = cms_ny().groupby(["year", "group", "ccn"]).discharges.sum().reset_index() \
                  .rename(columns={"group": "grp", "ccn": "unit", "discharges": "cms_ffs"})
    t = agg.merge(cms, on=["year", "grp", "unit"], how="outer")
    t = t[t.year.between(2013, 2024)].copy()
    t["hospital"] = t.unit.map(name)
    t["county"] = t.unit.map(main_cty)
    for c in ["all_payer", "medicare", "cms_ffs"]:
        t[c] = t[c].fillna(0)
    in_file = set()
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        y = 2000 + int(f.name.split("DY")[1][:2])
        ccns = pd.read_csv(f, dtype=str, encoding="latin-1", usecols=["Rndrng_Prvdr_CCN"]).Rndrng_Prvdr_CCN.str.zfill(6)
        in_file.update((y, c) for c in ccns.unique())
    t["cms_status"] = np.select(
        [t.cms_ffs > 0, [(y, u) in in_file for y, u in zip(t.year, t.unit)]],
        ["published", "suppressed (all cells < 11)"], "not in CMS file (exempt, merged or non-IPPS)")
    return t


def rho(x, y):
    return spearmanr(x, y).statistic if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan


def ci(v):
    return np.nanpercentile(v, [2.5, 97.5])


def concordance(t):
    """Analysis set per 14_rank_shuffle.py: hospitals with >= 10 SPARCS all-payer cases in the year/group.
    Ranks within the set: ties 'min', CMS-unpublished hospitals tied at the bottom (CMS count 0)."""
    rows, movers, boots = [], [], {}
    for y in YEARS:
        per = {}
        for g in ["spine", "cranial"]:
            d = t[(t.year == y) & (t.grp == g) & (t.all_payer >= 10)].copy()
            for c in ["all_payer", "medicare", "cms_ffs"]:
                d[f"rank_{c}"] = d[c].rank(ascending=False, method="min")
            d["rank_drop"] = d.rank_cms_ffs - d.rank_medicare          # positive = lower in CMS than in SPARCS Medicare
            d["pct_rank_drop"] = d.rank_drop / len(d)
            per[g] = d
            movers.append(d)
        # cluster bootstrap over hospital units, shared draw across spine and cranial (same hospitals largely
        # appear in both) so the spine-cranial difference is paired
        units = np.array(sorted(set(per["spine"].unit) | set(per["cranial"].unit)))
        draws = RNG.integers(0, len(units), size=(NBOOT, len(units)))
        bstat = {g: {k: np.full(NBOOT, np.nan) for k in "abcd"} for g in per}
        arr = {}
        for g, d in per.items():
            pos = {u: i for i, u in enumerate(d.unit)}
            arr[g] = (np.array([pos.get(u, -1) for u in units]), d.all_payer.to_numpy(), d.medicare.to_numpy(),
                      d.cms_ffs.to_numpy())
        for b in range(NBOOT):
            for g in per:
                ix, a, m, c = arr[g]
                sel = ix[draws[b]]
                sel = sel[sel >= 0]
                bstat[g]["a"][b] = rho(a[sel], c[sel])
                bstat[g]["b"][b] = rho(m[sel], c[sel])
                bstat[g]["c"][b] = rho(a[sel], m[sel])
                pub = sel[c[sel] > 0]
                bstat[g]["d"][b] = rho(m[pub], c[pub])
        boots[y] = bstat
        for g, d in per.items():
            pub = d[d.cms_ffs > 0]
            ma = d.dropna(subset=["county_ma_share"])
            r = {"year": y, "grp": g, "hospitals": len(d), "hospitals_cms_published": len(pub),
                 "rho_a_all_vs_cms": rho(d.all_payer, d.cms_ffs),
                 "rho_b_medicare_vs_cms": rho(d.medicare, d.cms_ffs),
                 "rho_c_all_vs_medicare": rho(d.all_payer, d.medicare),
                 "rho_d_medicare_vs_cms_published": rho(pub.medicare, pub.cms_ffs)}
            for k, col in zip("abcd", ["rho_a_all_vs_cms", "rho_b_medicare_vs_cms", "rho_c_all_vs_medicare",
                                        "rho_d_medicare_vs_cms_published"]):
                r[f"{col}_lo"], r[f"{col}_hi"] = ci(bstat[g][k])
            r["top10_overlap_medicare_cms"] = len(set(d.nlargest(10, "medicare").unit) & set(d.nlargest(10, "cms_ffs").unit))
            r["share_medicare_cases_cms_unpublished"] = d.loc[d.cms_ffs == 0, "medicare"].sum() / d.medicare.sum()
            r["cms_ffs_to_sparcs_medicare_ratio_published"] = pub.cms_ffs.sum() / pub.medicare.sum()
            # MA association: hospital rank drop (CMS rank - SPARCS Medicare rank) vs county MA share, and
            # hospital FFS:SPARCS-Medicare ratio among published hospitals vs county MA share
            r["rho_rankdrop_vs_county_ma"] = rho(ma.rank_drop, ma.county_ma_share)
            r["p_rankdrop_vs_county_ma"] = spearmanr(ma.rank_drop, ma.county_ma_share).pvalue
            pm = pub.dropna(subset=["county_ma_share"])
            pm = pm[pm.medicare > 0]
            r["rho_ffs_ratio_vs_county_ma_published"] = rho(pm.cms_ffs / pm.medicare, pm.county_ma_share)
            r["p_ffs_ratio_vs_county_ma_published"] = spearmanr(pm.cms_ffs / pm.medicare, pm.county_ma_share).pvalue
            rows.append(r)
    conc = pd.DataFrame(rows)
    diff = []
    for y in YEARS:
        dv = boots[y]["spine"]["b"] - boots[y]["cranial"]["b"]
        dd = boots[y]["spine"]["d"] - boots[y]["cranial"]["d"]
        da = boots[y]["spine"]["a"] - boots[y]["cranial"]["a"]
        s, c = conc[(conc.year == y) & (conc.grp == "spine")].iloc[0], conc[(conc.year == y) & (conc.grp == "cranial")].iloc[0]
        diff.append({"year": y,
                     "diff_a_spine_minus_cranial": s.rho_a_all_vs_cms - c.rho_a_all_vs_cms,
                     "diff_a_lo": ci(da)[0], "diff_a_hi": ci(da)[1],
                     "diff_b_spine_minus_cranial": s.rho_b_medicare_vs_cms - c.rho_b_medicare_vs_cms,
                     "diff_b_lo": ci(dv)[0], "diff_b_hi": ci(dv)[1],
                     "diff_d_spine_minus_cranial": s.rho_d_medicare_vs_cms_published - c.rho_d_medicare_vs_cms_published,
                     "diff_d_lo": ci(dd)[0], "diff_d_hi": ci(dd)[1]})
    return conc, pd.DataFrame(diff), pd.concat(movers)


# ---------------------------------------------------------------- 4. matched-window specifications
def trend_specs(d, st):
    windows = [(2013, 2024), (2013, 2019), (2019, 2024), (2022, 2024)]
    series = {}   # (table, grp, source, definition, age_scope, measure) -> year-indexed Series
    for g in ["spine", "cranial"]:
        base = d[d.drg_grp == g]
        med = base.pay1.str.match("(?i)^medicare", na=False)
        for age, f in AGES.items():
            series[("source_comparison", g, "SPARCS all-payer", "APR-DRG", age, "count")] = base[f(base)].groupby("year").size()
        for age in ["all ages", "70+"]:
            s = base[med & AGES[age](base)].groupby("year").size()
            series[("source_comparison", g, "SPARCS Medicare payer", "APR-DRG", age, "count")] = s
            series[("source_comparison", g, "SPARCS Medicare payer", "APR-DRG", age, "per 100k NY Medicare benes")] = \
                s / st.TOT_BENES.reindex(s.index) * 1e5
        cms = cms_ny()
        c = cms[cms.group == g].groupby("year").discharges.sum()
        series[("source_comparison", g, "CMS FFS, NY", "MS-DRG", "all FFS ages", "count")] = c
        series[("source_comparison", g, "CMS FFS, NY", "MS-DRG", "all FFS ages", "per 100k NY FFS Part A benes")] = \
            c / st.A_ORGNL_MDCR_BENES.reindex(c.index) * 1e5
        us = pd.read_csv(I / "national_sensitivity.csv").set_index("year")
        series[("source_comparison", g, "CMS FFS, US", "MS-DRG", "all FFS ages", "count")] = us[f"{g}_raw"]
        series[("source_comparison", g, "CMS FFS, US", "MS-DRG", "all FFS ages", "per 100k US FFS Part A benes")] = \
            us[f"{g}_raw"] / us.ffs_part_a_benes * 1e5
        # case-definition specifications (SPARCS only; data source held fixed within each block)
        for (dn, df_), (src, pay), age in itertools.product(DEFS.items(), [("SPARCS all-payer", False),
                                                                           ("SPARCS Medicare payer", True)], AGES):
            m = df_(d, g) & AGES[age](d)
            if pay:
                m &= d.pay1.str.match("(?i)^medicare", na=False)
            series[("case_definition", g, src, dn, age, "count")] = d[m].groupby("year").size()
    rows = []
    for (tab, g, src, dn, age, meas), s in series.items():
        for y0, y1 in windows:
            rows.append({"table": tab, "grp": g, "source": src, "definition": dn, "age_scope": age, "measure": meas,
                         "window": f"{y0}-{y1}", "start": s.get(y0), "end": s.get(y1),
                         "change_pct": pct(s.get(y0), s.get(y1))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. trend models
def fit_trend(y: pd.Series, offset: pd.Series | None):
    """log E[count] = b0 + b1*(year-2013) + b2*covid(2020-21) + offset. Returns APC and COVID IRR per model."""
    X = pd.DataFrame({"const": 1.0, "year_c": y.index - 2013, "covid": np.isin(y.index, [2020, 2021]).astype(float)},
                     index=y.index)
    off = None if offset is None else np.log(offset.reindex(y.index).astype(float))
    out = []
    pois = sm.GLM(y.astype(float), X, family=sm.families.Poisson(), offset=off).fit()
    quasi = sm.GLM(y.astype(float), X, family=sm.families.Poisson(), offset=off).fit(scale="X2")
    models = [("Poisson", pois), ("Quasi-Poisson (Pearson scale)", quasi)]
    try:
        # alpha is near 0 with 12 annual points, which stalls Newton; start with Nelder-Mead, then refine with BFGS
        nbm = sm.NegativeBinomial(y.astype(float), X, offset=off, loglike_method="nb2")
        start = nbm.fit(method="nm", maxiter=5000, disp=0).params
        nb = nbm.fit(start_params=start, method="bfgs", maxiter=2000, disp=0)
        models.append(("Negative binomial (NB2)", nb))
    except Exception as e:   # noqa: BLE001 - report instead of failing the whole run
        print("NB failed:", e)
    for name, m in models:
        b, (lo, hi) = m.params["year_c"], m.conf_int().loc["year_c"]
        cb, (clo, chi) = m.params["covid"], m.conf_int().loc["covid"]
        r = {"model": name, "apc_pct": np.expm1(b) * 100, "apc_lo": np.expm1(lo) * 100, "apc_hi": np.expm1(hi) * 100,
             "apc_p": m.pvalues["year_c"], "covid_irr": np.exp(cb), "covid_irr_lo": np.exp(clo), "covid_irr_hi": np.exp(chi)}
        if name.startswith("Quasi"):
            r["dispersion"] = m.scale
        if name.startswith("Negative"):
            r["nb_alpha"] = m.params["alpha"]
            r["converged"] = bool(m.mle_retvals.get("converged", False))
        out.append(r)
    return out


def trend_models(rt, st):
    us = pd.read_csv(I / "national_sensitivity.csv").set_index("year")
    rows = []
    for g in ["spine", "cranial"]:
        x = rt[rt.grp == g].set_index("year")
        specs = [("CMS FFS, NY", x.cms_ffs_ny, st.A_ORGNL_MDCR_BENES, "log NY FFS Part A beneficiaries"),
                 ("SPARCS Medicare payer, NY", x.sparcs_medicare, st.TOT_BENES, "log NY total Medicare beneficiaries"),
                 ("SPARCS all-payer, NY", x.sparcs_all_payer, None, "none"),
                 ("SPARCS Medicare payer, NY (no offset)", x.sparcs_medicare, None, "none"),
                 ("CMS FFS, NY (no offset)", x.cms_ffs_ny, None, "none"),
                 ("CMS FFS, US", us.loc[2013:2024, f"{g}_raw"], us.ffs_part_a_benes, "log US FFS Part A beneficiaries"),
                 ("CMS FFS, US (no offset)", us.loc[2013:2024, f"{g}_raw"], None, "none")]
        for src, y, off, lab in specs:
            for r in fit_trend(y.loc[2013:2024], off):
                rows.append({"grp": g, "series": src, "offset": lab, "years": "2013-2024", **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary
def f1(v, d=1):
    return f"{v:.{d}f}"


def headline_results(rt, ch, a70, conc, diff, specs, tm, top):
    """Headline results for the New York analyses (all values computed from the tables above)."""
    c = ch.set_index(["grp", "window"])
    def apc(g, series, model="Negative binomial (NB2)"):
        r = tm[(tm.grp == g) & (tm.series == series) & (tm.model == model)].iloc[0]
        return f"{r.apc_pct:+.2f}% (95% CI, {r.apc_lo:+.2f}% to {r.apc_hi:+.2f}%)"
    def cr(y, g):
        return conc[(conc.year == y) & (conc.grp == g)].iloc[0]
    def rr(r, k):
        return f"{getattr(r, k):.2f} ({getattr(r, k + '_lo'):.2f}-{getattr(r, k + '_hi'):.2f})"
    s24, c24 = cr(2024, "spine"), cr(2024, "cranial")
    dsig_b = diff[(diff.diff_b_lo > 0) | (diff.diff_b_hi < 0)].year.tolist()
    dsig_a = diff[(diff.diff_a_lo > 0) | (diff.diff_a_hi < 0)].year.tolist()
    dsig_d = diff[(diff.diff_d_lo > 0) | (diff.diff_d_hi < 0)].year.tolist()
    d24 = diff[diff.year == 2024].iloc[0]
    sp = conc[conc.grp == "spine"]
    a = a70[a70.grp == "spine"].set_index("year")
    cd = specs[(specs.table == "case_definition") & (specs.window == "2013-2024") & (specs.source == "SPARCS Medicare payer")]
    cd_s = cd[cd.grp == "spine"]
    mx = cd_s.loc[cd_s.change_pct.idxmax()]
    proc_all = cd_s[(cd_s.definition == "Procedure") & (cd_s.age_scope == "all ages")].change_pct.iloc[0]
    sc = specs[(specs.table == "source_comparison") & (specs.window == "2013-2024") & (specs.measure == "count")]
    t23 = top[(top.year == 2023) & (top.grp == "spine")]
    t24 = top[(top.year == 2024) & (top.grp == "spine")]
    nconv = tm[tm.model.str.startswith("Negative") & (tm.converged == False)]
    L = ["## 0. Headline results", "",
         f"1. **Medicare-payer counts and per-beneficiary rates.** SPARCS Medicare-payer spine counts changed "
         f"{c.loc[('spine','2013-2024'),'sparcs_medicare']:+.1f}% (2013-2024) while NY Medicare enrollment changed "
         f"{c.loc[('spine','2013-2024'),'ny_total_benes']:+.1f}%, so SPARCS Medicare-payer spine "
         f"inpatients per 100 000 beneficiaries changed {c.loc[('spine','2013-2024'),'sparcs_medicare_per100k_benes']:+.1f}% "
         f"(2013-2019: {c.loc[('spine','2013-2019'),'sparcs_medicare_per100k_benes']:+.1f}%; 2022-2024: "
         f"{c.loc[('spine','2022-2024'),'sparcs_medicare_per100k_benes']:+.1f}%). The trend model gives an annual change of "
         f"{apc('spine','SPARCS Medicare payer, NY')} per beneficiary, vs {apc('spine','CMS FFS, NY')} for CMS NY FFS per FFS "
         f"beneficiary and {apc('spine','CMS FFS, US')} for national FFS per FFS beneficiary. Without an offset the SPARCS Medicare "
         f"count trend is {apc('spine','SPARCS Medicare payer, NY (no offset)')}. "
         f"Cranial per beneficiary: {c.loc[('cranial','2013-2024'),'sparcs_medicare_per100k_benes']:+.1f}% "
         f"(APC {apc('cranial','SPARCS Medicare payer, NY')}), vs CMS NY FFS per FFS beneficiary "
         f"{c.loc[('cranial','2013-2024'),'cms_ffs_per100k_ffs_benes']:+.1f}%.",
         "",
         f"2. **Spine minus cranial rank concordance.** The paired hospital-bootstrap difference in rho (spine minus "
         f"cranial) has a 95% CI excluding 0 in {len(dsig_a)} of {len(YEARS)} years for all-payer vs CMS ({dsig_a or 'none'}), "
         f"{len(dsig_b)} of {len(YEARS)} for Medicare-payer vs CMS ({dsig_b or 'none'}), and {len(dsig_d)} of {len(YEARS)} for "
         f"the published-only comparison ({dsig_d or 'none'}); the tests are not corrected for multiplicity. In 2024 the Medicare-payer vs "
         f"CMS difference is {d24.diff_b_spine_minus_cranial:+.2f} ({d24.diff_b_lo:+.2f} to {d24.diff_b_hi:+.2f}).",
         "",
         f"3. **Rank concordance in 2024.** All-payer vs CMS (spine): {rr(s24,'rho_a_all_vs_cms')}. "
         f"Payer-matched (SPARCS Medicare vs CMS FFS): spine {rr(s24,'rho_b_medicare_vs_cms')}, cranial "
         f"{rr(c24,'rho_b_medicare_vs_cms')}. SPARCS all-payer vs SPARCS Medicare: spine {rr(s24,'rho_c_all_vs_medicare')}. "
         f"Among CMS-published hospitals only: spine {rr(s24,'rho_d_medicare_vs_cms_published')} (n = {s24.hospitals_cms_published}), "
         f"cranial {rr(c24,'rho_d_medicare_vs_cms_published')} (n = {c24.hospitals_cms_published}). Matching payer scope changes "
         f"spine rho (b minus a) by {(sp.rho_b_medicare_vs_cms - sp.rho_a_all_vs_cms).min():+.2f} to "
         f"{(sp.rho_b_medicare_vs_cms - sp.rho_a_all_vs_cms).max():+.2f} across years.",
         "",
         f"4. **County MA penetration.** Across hospitals, the Spearman correlation between rank drop (CMS rank minus SPARCS "
         f"Medicare rank, zero-count hospitals tied at the bottom) and county MA share has spine rho "
         f"{sp.rho_rankdrop_vs_county_ma.min():.2f} to {sp.rho_rankdrop_vs_county_ma.max():.2f} "
         f"(P < .05 in {(sp.p_rankdrop_vs_county_ma < .05).sum()} of {len(YEARS)} years). Among "
         f"CMS-published hospitals, the correlation between the hospital CMS FFS : SPARCS Medicare ratio and county MA share has spine rho "
         f"{sp.rho_ffs_ratio_vs_county_ma_published.min():.2f} to {sp.rho_ffs_ratio_vs_county_ma_published.max():.2f} "
         f"(P < .05 in {(sp.p_ffs_ratio_vs_county_ma_published < .05).sum()} of {len(YEARS)} years; 2024 rho "
         f"{s24.rho_ffs_ratio_vs_county_ma_published:.2f}).",
         "",
         f"5. **Commercial or miscellaneous primary payer among patients 70+.** Spine 70+ with Private/BCBS/Managed Care "
         f"Unspecified/Misc-Other primary payer: "
         f"{a.loc[2013,'share_commercial_or_misc']*100:.1f}% ({a.loc[2013,'commercial']+a.loc[2013,'misc_other']}/{a.loc[2013,'total_70plus']}) in 2013 "
         f"and {a.loc[2024,'share_commercial_or_misc']*100:.1f}% ({a.loc[2024,'commercial']+a.loc[2024,'misc_other']}/{a.loc[2024,'total_70plus']}) in 2024 "
         f"(range {a.share_commercial_or_misc.min()*100:.1f}%-{a.share_commercial_or_misc.max()*100:.1f}%); commercial alone "
         f"{a.share_commercial.min()*100:.1f}%-{a.share_commercial.max()*100:.1f}%; any non-Medicare {a.share_non_medicare.min()*100:.1f}%-"
         f"{a.share_non_medicare.max()*100:.1f}%. Redacted facility IDs: "
         f"{a70.share_pfi_redacted.max()*100:.2f}% or less of neurosurgical records in every year.",
         "",
         f"6. **Matched-window specifications.** With identical 2013-2024 windows, spine source estimates run from "
         f"{sc[sc.grp=='spine'].change_pct.min():+.1f}% to {sc[sc.grp=='spine'].change_pct.max():+.1f}%, "
         f"and SPARCS Medicare payer (APR-DRG, all ages) is "
         f"{sc[(sc.grp=='spine')&(sc.source=='SPARCS Medicare payer')&(sc.age_scope=='all ages')].change_pct.iloc[0]:+.1f}%. "
         f"Across the SPARCS Medicare-payer case-definition specifications (4 definitions x 3 age scopes) the range is "
         f"{cd_s.change_pct.min():+.1f}% to {cd_s.change_pct.max():+.1f}% (maximum: {mx.definition}, {mx.age_scope}); the "
         f"procedure definition at all ages is {proc_all:+.1f}%. The SPARCS Medicare-payer rows carry no age filter in the "
         "'all ages' scope; 70+ is a separate specification.",
         "",
         f"7. **Top-15 hospitals by SPARCS Medicare volume.** In 2023, {(t23.cms_status == 'published').sum()}/15 top-15 spine "
         f"hospitals were CMS-published; in 2024, {(t24.cms_status != 'published').sum()} were not "
         f"({', '.join(t24[t24.cms_status != 'published'].hospital)}). "
         "Examples (2023 spine, SPARCS Medicare vs CMS FFS): "
         + "; ".join(f"{r.hospital} {int(r.medicare)} vs {int(r.cms_ffs)} (rank {int(r.rank_medicare)} vs {int(r.rank_cms_published)}, county MA {r.county_ma_share*100:.0f}%)"
                     for r in t23.sort_values("cms_minus_medicare_rank", ascending=False).head(4).itertuples()) + ".",
         "",
         "8. **Data notes.** SPARCS public age bands do not allow 65+; the CMS file includes under-65 FFS "
         "beneficiaries; the SPARCS denominator here is total Medicare beneficiaries (all ages). APR-DRG grouper versions change "
         "across SPARCS releases (3M updates) and add definitional drift (no version field exists in the public file). "
         "'MA and other' enrollment includes cost plans, PACE and demonstrations. Trend models use 12 annual points; "
         "Poisson, quasi-Poisson (Pearson scale) and NB2 estimates are in section 5."
         + (f" NB2 did not converge for: {', '.join(nconv.grp + ' ' + nconv.series)}; use quasi-Poisson for those." if len(nconv) else
            " All NB2 fits converged."),
         ""]
    return L


def summary(rt, ch, a70, conc, diff, specs, tm, top, movers):
    L = ["# New York analyses: key numbers", "",
         "Generated by 25_ny_rates_rankings.py. Case definition: APR-DRG (SPARCS) and MS-DRG groups (CMS). CMS data years are "
         "calendar years. Rates use Medicare enrollment 'Year' rows (NY state level).", ""]
    L += headline_results(rt, ch, a70, conc, diff, specs, tm, top)
    L += ["## 1. Per-beneficiary rates", ""]
    for g in ["spine", "cranial"]:
        x = rt[rt.grp == g].set_index("year")
        L.append(f"**{g.title()}.** SPARCS Medicare-payer per 100 000 NY Medicare beneficiaries: "
                 f"{f1(x.loc[2013,'sparcs_medicare_per100k_benes'])} (2013), {f1(x.loc[2019,'sparcs_medicare_per100k_benes'])} (2019), "
                 f"{f1(x.loc[2022,'sparcs_medicare_per100k_benes'])} (2022), {f1(x.loc[2024,'sparcs_medicare_per100k_benes'])} (2024). "
                 f"CMS NY FFS per 100 000 NY FFS Part A beneficiaries: {f1(x.loc[2013,'cms_ffs_per100k_ffs_benes'])}, "
                 f"{f1(x.loc[2019,'cms_ffs_per100k_ffs_benes'])}, {f1(x.loc[2022,'cms_ffs_per100k_ffs_benes'])}, "
                 f"{f1(x.loc[2024,'cms_ffs_per100k_ffs_benes'])}.")
        L.append("")
        L.append("| Window | SPARCS Medicare count | SPARCS Medicare /100k benes | CMS NY FFS count | CMS NY FFS /100k FFS benes | SPARCS all-payer count |")
        L.append("|---|---|---|---|---|---|")
        for r in ch[ch.grp == g].itertuples():
            L.append(f"| {r.window} | {r.sparcs_medicare:+.1f}% | {r.sparcs_medicare_per100k_benes:+.1f}% | {r.cms_ffs_ny:+.1f}% | "
                     f"{r.cms_ffs_per100k_ffs_benes:+.1f}% | {r.sparcs_all_payer:+.1f}% |")
        L.append("")
    e = ch[ch.grp == "spine"].set_index("window")
    L.append(f"NY total Medicare beneficiaries {e.loc['2013-2024','ny_total_benes']:+.1f}% and FFS Part A beneficiaries "
             f"{e.loc['2013-2024','ny_ffs_part_a_benes']:+.1f}% (2013-2024).")
    L += ["", "## 2. Age 70+ payer mix and redacted facility IDs", ""]
    L.append("| Group | Year | 70+ total | Medicare | Commercial/BCBS/MC unspec. | Medicaid | Misc/Other | Other | % non-Medicare | % commercial | % commercial or misc | % PFI redacted (all ages) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in a70.itertuples():
        L.append(f"| {r.grp} | {r.year} | {r.total_70plus} | {r.medicare} | {r.commercial} | {r.medicaid} | {r.misc_other} | {r.other} | "
                 f"{r.share_non_medicare*100:.1f} | {r.share_commercial*100:.1f} | {r.share_commercial_or_misc*100:.1f} | {r.share_pfi_redacted*100:.2f} |")
    L += ["", "## 3. Payer-matched rank concordance (hospitals with >=10 SPARCS all-payer cases; 2000 hospital-bootstrap resamples)", ""]
    L.append("| Year | Group | n | (a) all-payer vs CMS | (b) Medicare vs CMS | (c) all-payer vs Medicare | (d) Medicare vs CMS, CMS-published (n) | rank drop vs county MA, rho (P) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in conc.itertuples():
        L.append(f"| {r.year} | {r.grp} | {r.hospitals} | {r.rho_a_all_vs_cms:.2f} ({r.rho_a_all_vs_cms_lo:.2f}-{r.rho_a_all_vs_cms_hi:.2f}) | "
                 f"{r.rho_b_medicare_vs_cms:.2f} ({r.rho_b_medicare_vs_cms_lo:.2f}-{r.rho_b_medicare_vs_cms_hi:.2f}) | "
                 f"{r.rho_c_all_vs_medicare:.2f} ({r.rho_c_all_vs_medicare_lo:.2f}-{r.rho_c_all_vs_medicare_hi:.2f}) | "
                 f"{r.rho_d_medicare_vs_cms_published:.2f} ({r.rho_d_medicare_vs_cms_published_lo:.2f}-{r.rho_d_medicare_vs_cms_published_hi:.2f}; {r.hospitals_cms_published}) | "
                 f"{r.rho_rankdrop_vs_county_ma:.2f} ({r.p_rankdrop_vs_county_ma:.3f}) |")
    L += ["", "Spine minus cranial difference in rho (paired hospital bootstrap, 95% CI):", "",
          "| Year | (a) all-payer vs CMS | (b) Medicare vs CMS | (d) published only |", "|---|---|---|---|"]
    for r in diff.itertuples():
        L.append(f"| {r.year} | {r.diff_a_spine_minus_cranial:+.2f} ({r.diff_a_lo:+.2f} to {r.diff_a_hi:+.2f}) | "
                 f"{r.diff_b_spine_minus_cranial:+.2f} ({r.diff_b_lo:+.2f} to {r.diff_b_hi:+.2f}) | "
                 f"{r.diff_d_spine_minus_cranial:+.2f} ({r.diff_d_lo:+.2f} to {r.diff_d_hi:+.2f}) |")
    # pooled MA association across years (descriptive)
    mv = movers.dropna(subset=["county_ma_share"])
    L += ["", "Pooled 2013-2024 (hospital-years, descriptive, not independent): Spearman rho between percentile rank drop "
          "(CMS rank minus SPARCS Medicare rank, divided by hospitals) and county MA share: "
          + "; ".join(f"{g} {rho(x.pct_rank_drop, x.county_ma_share):.2f}" for g, x in mv.groupby("grp"))
          + ". Restricted to CMS-published hospitals, rho between the CMS FFS / SPARCS Medicare ratio and county MA share: "
          + "; ".join(f"{g} {rho(x.cms_ffs / x.medicare, x.county_ma_share):.2f}"
                      for g, x in mv[(mv.cms_ffs > 0) & (mv.medicare > 0)].groupby("grp")) + "."]
    L += ["", "## 4. Matched-window source comparison (APR-DRG / MS-DRG; identical windows)", "",
          "| Group | Source | Age scope | Measure | 2013-2024 | 2013-2019 | 2019-2024 | 2022-2024 |", "|---|---|---|---|---|---|---|---|"]
    sc = specs[specs.table == "source_comparison"].pivot_table(index=["grp", "source", "age_scope", "measure"],
                                                               columns="window", values="change_pct", sort=False)
    for k, r in sc.iterrows():
        L.append(f"| {' | '.join(k)} | " + " | ".join(f"{r[w]:+.1f}%" for w in ["2013-2024", "2013-2019", "2019-2024", "2022-2024"]) + " |")
    cd = specs[(specs.table == "case_definition")]
    L += ["", "Case-definition specifications (SPARCS only, 4 definitions x 3 age scopes), range of change by source and window:", "",
          "| Group | Source | 2013-2024 | 2013-2019 | 2019-2024 | 2022-2024 |", "|---|---|---|---|---|---|"]
    for (g, s), x in cd.groupby(["grp", "source"], sort=False):
        L.append(f"| {g} | {s} | " + " | ".join(
            f"{x[x.window == w].change_pct.min():+.1f}% to {x[x.window == w].change_pct.max():+.1f}%"
            for w in ["2013-2024", "2013-2019", "2019-2024", "2022-2024"]) + " |")
    L += ["", "## 5. Trend models, 2013-2024 (year continuous + COVID 2020-2021 indicator)", "",
          "| Group | Series | Offset | Model | APC % (95% CI) | COVID IRR (95% CI) |", "|---|---|---|---|---|---|"]
    for r in tm.itertuples():
        L.append(f"| {r.grp} | {r.series} | {r.offset} | {r.model} | {r.apc_pct:+.2f} ({r.apc_lo:+.2f} to {r.apc_hi:+.2f}) | "
                 f"{r.covid_irr:.2f} ({r.covid_irr_lo:.2f}-{r.covid_irr_hi:.2f}) |")
    L += ["", "## 6. Figure 3 support: top 15 by SPARCS Medicare-payer volume", ""]
    for (y, g), x in top.groupby(["year", "grp"]):
        L += [f"**{y} {g}**", "", "| Medicare rank | Hospital | County | County MA share | SPARCS Medicare | SPARCS all-payer | CMS FFS | CMS rank | CMS status |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in x.itertuples():
            L.append(f"| {int(r.rank_medicare)} | {r.hospital} | {r.county} | {r.county_ma_share*100:.1f}% | {int(r.medicare)} | "
                     f"{int(r.all_payer)} | {int(r.cms_ffs)} | {'' if np.isnan(r.rank_cms_published) else int(r.rank_cms_published)} | {r.cms_status} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    d = duckdb.connect().execute(LOAD).df()
    st, co = enrollment()

    rt, ch = rates(d, st)
    rt.to_csv(OUT / "rates_per_beneficiary.csv", index=False)
    ch.to_csv(OUT / "rates_changes.csv", index=False)
    print(ch.to_string(index=False))

    a70 = age70_payer(d)
    a70.to_csv(OUT / "age70_payer_redaction.csv", index=False)
    print(a70.round(3).to_string(index=False))

    sp = hospital_units(d)
    t = rank_table(sp, co)
    unmatched = sorted(set(sp.county.dropna()) - set(co.county))
    print("SPARCS counties without enrollment match:", unmatched)
    conc, diff, movers = concordance(t)
    conc.to_csv(OUT / "rank_concordance.csv", index=False)
    diff.to_csv(OUT / "rank_rho_difference_spine_cranial.csv", index=False)
    movers.to_csv(OUT / "rank_table.csv", index=False)
    print(conc.round(3).to_string(index=False))
    print(diff.round(3).to_string(index=False))

    # Figure 3 support. CMS rank among CMS-published NY hospitals (all units, no volume floor); status as in 14_rank_shuffle.py.
    t2 = t.copy()
    t2["rank_medicare"] = t2.groupby(["year", "grp"]).medicare.rank(ascending=False, method="min")
    t2["rank_all_payer"] = t2.groupby(["year", "grp"]).all_payer.rank(ascending=False, method="min")
    t2["rank_cms_published"] = t2[t2.cms_ffs > 0].groupby(["year", "grp"]).cms_ffs.rank(ascending=False, method="min")
    t2["cms_minus_medicare_rank"] = t2.rank_cms_published - t2.rank_medicare
    top = (t2[t2.year.isin([2023, 2024])].sort_values("medicare", ascending=False)
           .groupby(["year", "grp"]).head(15).sort_values(["year", "grp", "rank_medicare"]))
    cols = ["year", "grp", "rank_medicare", "hospital", "unit", "county", "county_ma_share", "medicare", "all_payer",
            "rank_all_payer", "cms_ffs", "rank_cms_published", "cms_minus_medicare_rank", "cms_status"]
    top[cols].to_csv(OUT / "top15_medicare_2023_2024.csv", index=False)
    top[top.year == 2023][cols].to_csv(OUT / "top15_medicare_2023.csv", index=False)
    print(top[cols].round(3).to_string(index=False))

    specs = trend_specs(d, st)
    specs.to_csv(OUT / "trend_specs.csv", index=False)

    tm = trend_models(rt, st)
    tm.to_csv(OUT / "trend_models.csv", index=False)
    print(tm.round(3).to_string(index=False))

    (OUT / "SUMMARY.md").write_text(summary(rt, ch, a70, conc, diff, specs, tm, top, movers))
