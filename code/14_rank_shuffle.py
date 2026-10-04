"""Who is a 'top' spine or cranial hospital depends on the data source (New York).

For each hospital (CMS CCN; SPARCS PFIs aggregated through the crosswalk) and year, ranks by:
  SPARCS all-payer volume, SPARCS Medicare-payer volume, CMS Medicare FFS published volume.
Concordance: Spearman rho (bootstrap 95% CI over hospitals), top-10 overlap, and rank displacement
of the top-20 all-payer hospitals in the CMS ranking.
Outputs: data/interim/ny_rank_table.csv, data/interim/ny_rank_concordance.csv
"""
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ROOT

I = ROOT / "data/interim"
RNG = np.random.default_rng(20260914)   # fixed seed for reproducible bootstrap intervals


def boot_rho(a: np.ndarray, b: np.ndarray, n=2000):
    idx = RNG.integers(0, len(a), size=(n, len(a)))
    vals = [spearmanr(a[i], b[i]).statistic for i in idx]
    return np.nanpercentile(vals, [2.5, 97.5])


if __name__ == "__main__":
    sp = duckdb.connect().execute(f"""
      SELECT year, pfi, any_value(facility_name) AS facility_name,
             CASE WHEN TRY_CAST(apr_drg AS INT) IN (20,21,22) THEN 'cranial' ELSE 'spine' END AS grp,
             count(*) AS all_payer, sum(regexp_matches(pay1, '(?i)^medicare')::INT) AS medicare
      FROM read_parquet('{ROOT}/data/interim/sparcs/*/*.parquet', union_by_name=true)
      WHERE TRY_CAST(apr_drg AS INT) IN (20,21,22,23,303,304,310,321) AND pfi IS NOT NULL
      GROUP BY ALL""").df()
    xw = pd.read_csv(I / "ny_pfi_ccn_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok == True].drop_duplicates("pfi")[["pfi", "ccn"]]
    sp = sp.merge(xw, on="pfi", how="left")
    sp["unit"] = sp.ccn.fillna("pfi" + sp.pfi.astype(int).astype(str))
    # consolidations listed in data/manual/NY_ccn_consolidation.csv: campuses CMS reports under a parent system CCN
    cons = pd.read_csv(ROOT / "data/manual/NY_ccn_consolidation.csv", dtype=str)
    for r in cons.itertuples():
        m = (sp.unit == r.from_unit) & (sp.year >= int(r.start_year))
        sp.loc[m, "unit"] = r.to_ccn
    name = sp.sort_values("all_payer").groupby("unit").facility_name.last()
    agg = sp.groupby(["year", "grp", "unit"])[["all_payer", "medicare"]].sum().reset_index()

    cms = pd.read_parquet(I / "cms_neuro_hospital_year.parquet")
    cms = cms[(cms.state == "NY") & cms.group.isin(["cranial", "spine"])]
    cms = cms.groupby(["year", "group", "ccn"]).discharges.sum().reset_index() \
             .rename(columns={"group": "grp", "ccn": "unit", "discharges": "cms_ffs"})
    t = agg.merge(cms, on=["year", "grp", "unit"], how="outer")
    t = t[t.year.between(2013, 2024)]
    t["hospital"] = t.unit.map(name)
    for c in ["all_payer", "medicare", "cms_ffs"]:
        t[c] = t[c].fillna(0)
        t[f"rank_{c}"] = t.groupby(["year", "grp"])[c].rank(ascending=False, method="min")
    t.loc[t.cms_ffs == 0, "rank_cms_ffs"] = np.nan   # not published by CMS
    # Why a hospital has no CMS count: it files under IPPS but every neuro cell is < 11 (suppressed),
    # or its CCN is absent from the whole CMS file that year (PPS-exempt cancer hospital, merged CCN, non-IPPS)
    in_file = set()
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        y = 2000 + int(f.name.split("DY")[1][:2])
        ccns = pd.read_csv(f, dtype=str, encoding="latin-1", usecols=["Rndrng_Prvdr_CCN"]).Rndrng_Prvdr_CCN.str.zfill(6)
        in_file.update((y, c) for c in ccns.unique())
    t["cms_status"] = np.select(
        [t.cms_ffs > 0, [(y, u) in in_file for y, u in zip(t.year, t.unit)]],
        ["published", "suppressed (all cells < 11)"], "not in CMS file (exempt, merged or non-IPPS)")
    t.to_csv(I / "ny_rank_table.csv", index=False)

    rows = []
    for (y, g), d in t.groupby(["year", "grp"]):
        d = d[d.all_payer >= 10]
        if len(d) < 10:
            continue
        a, m, c = d.all_payer.to_numpy(), d.medicare.to_numpy(), d.cms_ffs.to_numpy()
        lo, hi = boot_rho(a, c)
        top_all = set(d.nlargest(10, "all_payer").unit)
        top_cms = set(d.nlargest(10, "cms_ffs").unit)
        top_med = set(d.nlargest(10, "medicare").unit)
        top20 = d.nlargest(20, "all_payer")
        rows.append({"year": y, "grp": g, "hospitals": len(d),
                     "rho_all_vs_cms": spearmanr(a, c).statistic, "rho_lo": lo, "rho_hi": hi,
                     "rho_all_vs_medicare": spearmanr(a, m).statistic,
                     "top10_overlap_all_cms": len(top_all & top_cms), "top10_overlap_all_medicare": len(top_all & top_med),
                     "top20_unpublished_in_cms": int(top20.cms_ffs.eq(0).sum()),
                     "top20_median_abs_rank_shift": float((top20.rank_all_payer - top20.rank_cms_ffs).abs().median()),
                     "share_cases_at_unpublished": d.loc[d.cms_ffs == 0, "all_payer"].sum() / a.sum(),
                     "share_cases_suppressed": d.loc[d.cms_status.str.startswith("suppressed"), "all_payer"].sum() / a.sum(),
                     "share_cases_not_in_file": d.loc[d.cms_status.str.startswith("not in"), "all_payer"].sum() / a.sum()})
    conc = pd.DataFrame(rows)
    conc.to_csv(I / "ny_rank_concordance.csv", index=False)
    pd.set_option("display.width", 250)
    print(conc.round(3).to_string())
    for g in ["spine", "cranial"]:
        d = t[(t.year == 2023) & (t.grp == g)].nlargest(20, "all_payer")
        print(f"\n== 2023 {g}: top 20 by SPARCS all-payer")
        print(d[["hospital", "unit", "all_payer", "rank_all_payer", "medicare", "rank_medicare", "cms_ffs", "rank_cms_ffs"]].to_string(index=False))
