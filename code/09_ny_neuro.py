"""NY SPARCS neurosurgical cohorts: cranial vs spine.

1. Two independent case definitions per year:
   - DRG-based: APR-DRG (cranial 020/021/022; spine 023/303/304/310/321)
   - Procedure-based: principal procedure CCS (<=2017: cranial 1,2; spine 3,158) or
     CCSR (2018+: cranial CNS004/005/007/009/010; spine MST013/MST016/CNS008/PNS001/PNS002)
   Counts, overlap (Jaccard) and the break at the CCS->CCSR switch.
2. Field quality inside each cohort.
3. Hospital concentration (share of cases at hospitals with <25 cases/yr; top-10 share).
4. External validity: SPARCS Medicare-payer discharges by hospital (PFI->CCN) vs CMS Medicare FFS
   MS-DRG counts, cranial vs spine: ratio, rank agreement (Spearman), and volume-tier agreement.
Outputs in data/interim/ny_neuro_*.csv
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/interim"
SRC = f"read_parquet('{ROOT}/data/interim/sparcs/*/*.parquet', union_by_name=true)"

COHORT_SQL = """
  CREATE OR REPLACE TABLE neuro AS
  WITH base AS (
    SELECT *, TRY_CAST(apr_drg AS INT) AS drg,
           CASE WHEN dx_system = 'CCS' THEN ltrim(proc_code, '0') ELSE proc_code END AS pc
    FROM s)
  SELECT *,
    CASE WHEN drg IN (20, 21, 22) THEN 'cranial'
         WHEN drg IN (23, 303, 304, 310, 321) THEN 'spine' END AS drg_group,
    CASE WHEN dx_system = 'CCS' AND pc IN ('1', '2') THEN 'cranial'
         WHEN dx_system = 'CCS' AND pc IN ('3', '158') THEN 'spine'
         WHEN dx_system = 'CCSR' AND pc IN ('CNS004','CNS005','CNS007','CNS009','CNS010') THEN 'cranial'
         WHEN dx_system = 'CCSR' AND pc IN ('MST013','MST016','CNS008','PNS001','PNS002') THEN 'spine' END AS proc_group,
    CASE WHEN drg IN (20, 21) THEN 'craniotomy' WHEN drg = 22 THEN 'shunt'
         WHEN drg IN (303, 304, 321) THEN 'fusion' WHEN drg IN (23, 310) THEN 'nonfusion' END AS drg_sub
  FROM base
  WHERE drg IN (20, 21, 22, 23, 303, 304, 310, 321)
     OR pc IN ('1','2','3','158','CNS004','CNS005','CNS007','CNS009','CNS010','MST013','MST016','CNS008','PNS001','PNS002')
"""

QUALITY = """
  count(*) n,
  avg((race IS NULL OR race ILIKE '%unknown%')::INT) race_unknown,
  avg((race ILIKE 'other%')::INT) race_other,
  avg((ethnicity IS NULL OR ethnicity ILIKE '%unknown%')::INT) eth_unknown,
  avg((pay1 IS NULL OR pay1 ILIKE '%unknown%' OR pay1 ILIKE 'miscellaneous%')::INT) payer_unknown,
  avg((pay1 ILIKE 'medicare%')::INT) medicare,
  avg((pay1 ILIKE 'medicaid%')::INT) medicaid,
  avg((pfi IS NULL)::INT) pfi_redacted,
  avg((zip3 IS NULL)::INT) zip_missing,
  avg((soi >= 3)::INT) soi_3_4,
  avg((costs > charges)::INT) costs_gt_charges,
  avg((admit_type ILIKE 'elective')::INT) elective,
  median(los) median_los, median(costs) median_cost
"""

if __name__ == "__main__":
    con = duckdb.connect()
    con.execute(f"CREATE VIEW s AS SELECT * FROM {SRC}")
    con.execute(COHORT_SQL)

    # 1. definition agreement
    agree = con.execute("""
      WITH g AS (SELECT unnest(['cranial','spine']) grp)
      SELECT year, dx_system, grp,
             sum((drg_group = grp)::INT) drg_n,
             sum((proc_group = grp)::INT) proc_n,
             sum((drg_group = grp AND proc_group = grp)::INT) both_n,
             sum((drg_group = grp OR proc_group = grp)::INT) either_n
      FROM neuro, g GROUP BY ALL ORDER BY grp, year""").df()
    agree["jaccard"] = agree.both_n / agree.either_n
    agree["proc_over_drg"] = agree.proc_n / agree.drg_n
    agree.to_csv(OUT / "ny_neuro_definitions.csv", index=False)
    print(agree.round(3).to_string())

    sub = con.execute("SELECT year, drg_sub, count(*) n FROM neuro WHERE drg_sub IS NOT NULL GROUP BY ALL").df()
    sub.pivot(index="year", columns="drg_sub", values="n").to_csv(OUT / "ny_neuro_drg_subgroups.csv")

    # 2. field quality (DRG-defined cohorts, plus all-discharge reference)
    q = con.execute(f"SELECT year, drg_group AS grp, {QUALITY} FROM neuro WHERE drg_group IS NOT NULL GROUP BY ALL").df()
    ref = con.execute(f"SELECT year, 'all_discharges' AS grp, {QUALITY} FROM s GROUP BY ALL").df()
    quality = pd.concat([q, ref]).sort_values(["grp", "year"])
    quality.to_csv(OUT / "ny_neuro_quality.csv", index=False)
    print(quality.round(3).to_string())

    # 3. concentration
    hosp = con.execute("""SELECT year, drg_group grp, pfi, any_value(facility_name) facility_name, count(*) n
                          FROM neuro WHERE drg_group IS NOT NULL GROUP BY ALL""").df()
    hosp.to_csv(OUT / "ny_neuro_hospital_year.csv", index=False)

    def conc(d):
        d = d.dropna(subset=["pfi"]).sort_values("n", ascending=False)
        tot = d.n.sum()
        return pd.Series({"hospitals": len(d), "cases": tot, "top10_share": d.n.head(10).sum() / tot,
                          "lowvol_hospitals_lt25": (d.n < 25).sum(),
                          "lowvol_case_share": d.loc[d.n < 25, "n"].sum() / tot,
                          "hhi": ((d.n / tot) ** 2).sum()})
    concentration = hosp.groupby(["grp", "year"]).apply(conc, include_groups=False)
    concentration.to_csv(OUT / "ny_neuro_concentration.csv")
    print(concentration.round(3).to_string())

    # 4. validation vs CMS Medicare FFS (MS-DRG) at CCN level
    med = con.execute("""SELECT year, drg_group grp, pfi, count(*) n FROM neuro
                         WHERE drg_group IS NOT NULL AND pay1 ILIKE 'medicare%' AND pfi IS NOT NULL GROUP BY ALL""").df()
    xw = pd.read_csv(OUT / "ny_pfi_ccn_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok][["pfi", "ccn"]].drop_duplicates()
    med = med.merge(xw, on="pfi").groupby(["year", "grp", "ccn"]).n.sum().reset_index(name="sparcs_medicare")
    cms = pd.read_parquet(OUT / "cms_neuro_hospital_year.parquet")
    cms = cms[(cms.state == "NY") & cms.group.isin(["cranial", "spine"])]
    cms = cms.groupby(["year", "group", "ccn"]).discharges.sum().reset_index().rename(columns={"group": "grp", "discharges": "cms_ffs"})
    v = cms.merge(med, on=["year", "grp", "ccn"], how="outer")
    v.to_csv(OUT / "ny_neuro_vs_cms_hospital.csv", index=False)

    def val(d):
        both = d.dropna(subset=["cms_ffs", "sparcs_medicare"])
        rho = spearmanr(both.cms_ffs, both.sparcs_medicare).statistic if len(both) > 5 else np.nan
        # volume tiers: top quartile by each source
        tier_agree = np.nan
        if len(both) > 8:
            a = both.cms_ffs >= both.cms_ffs.quantile(0.75)
            b = both.sparcs_medicare >= both.sparcs_medicare.quantile(0.75)
            tier_agree = (a & b).sum() / max(1, (a | b).sum())
        return pd.Series({
            "ccn_cms": d.cms_ffs.notna().sum(), "ccn_sparcs": d.sparcs_medicare.notna().sum(),
            "ccn_both": len(both), "cms_total": d.cms_ffs.sum(), "sparcs_total_matched": both.sparcs_medicare.sum(),
            "sparcs_total_all": d.sparcs_medicare.sum(),
            "ratio_sparcs_to_cms": both.sparcs_medicare.sum() / both.cms_ffs.sum(),
            "median_hosp_ratio": np.exp(np.log(both.sparcs_medicare / both.cms_ffs).median()),
            "spearman": rho, "top_quartile_jaccard": tier_agree,
            "sparcs_hospitals_hidden_by_cms": (d.cms_ffs.isna() & d.sparcs_medicare.notna()).sum(),
            "sparcs_cases_at_hidden_hospitals": d.loc[d.cms_ffs.isna(), "sparcs_medicare"].sum(),
        })
    validation = v[v.year.between(2013, 2024)].groupby(["grp", "year"]).apply(val, include_groups=False)
    validation.to_csv(OUT / "ny_neuro_vs_cms_by_year.csv")
    print(validation.round(3).to_string())
