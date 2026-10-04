"""Record-level field quality and drift across SPARCS PUF years.

Outputs
  data/interim/sparcs_value_counts.parquet   field x value x year counts (drift catalogue)
  data/interim/sparcs_quality_by_year.csv    headline quality metrics per year
  data/interim/sparcs_quality_by_facility.parquet  facility-year metrics (for variance partitioning)
  data/interim/sparcs_race_icc.csv           share of race-unknown variance between facilities
The facility-year table feeds 13_multiverse.py. The value-count, by-year and race tables are
descriptive records of field quality and drift in the public files; no other script reads them.
"""
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = f"read_parquet('{ROOT}/data/interim/sparcs/*/*.parquet', hive_partitioning=false, union_by_name=true)"
CAT = ["age_group", "sex", "race", "ethnicity", "admit_type", "disposition", "pay1", "med_surg",
       "rom", "abortion_flag", "ed_flag", "dx_system", "hsa"]

METRICS = """
  count(*) AS n,
  avg((pfi IS NULL)::INT) AS pfi_redacted,
  avg((zip3 IS NULL)::INT) AS zip_missing_or_oos,
  avg((zip_raw = 'OOS')::INT) AS zip_oos,
  avg((age_group IS NULL OR age_group = '')::INT) AS age_missing,
  avg((sex NOT IN ('M','F'))::INT) AS sex_unknown,
  avg((race IS NULL OR race ILIKE '%unknown%')::INT) AS race_unknown,
  avg((race ILIKE 'other%')::INT) AS race_other,
  avg((ethnicity IS NULL OR ethnicity ILIKE '%unknown%')::INT) AS eth_unknown,
  avg((pay1 IS NULL OR pay1 ILIKE '%unknown%' OR pay1 ILIKE 'miscellaneous%')::INT) AS payer_unknown,
  avg((soi IS NULL OR soi = 0)::INT) AS soi_ungroupable,
  avg((soi >= 3)::INT) AS soi_major_extreme,
  avg((rom IN ('Major','Extreme'))::INT) AS rom_major_extreme,
  avg((los_raw LIKE '120%')::INT) AS los_topcoded,
  avg(los) AS mean_los,
  avg((charges IS NULL OR charges <= 0)::INT) AS charges_nonpos,
  avg((costs IS NULL OR costs <= 0)::INT) AS costs_nonpos,
  avg((costs > charges)::INT) AS costs_gt_charges,
  median(costs / NULLIF(charges, 0)) AS median_ccr,
  avg((charges % 1000 = 0 AND charges > 0)::INT) AS charges_round_1000,
  avg((ed_flag = 'Y')::INT) AS ed_admit,
  avg((admit_type ILIKE 'newborn')::INT) AS newborn,
  avg((apr_mdc = '15')::INT) AS mdc15_newborn,
  avg((dx_code IS NULL OR dx_code IN ('0', '259', 'XXX000'))::INT) AS dx_unclassified
"""

if __name__ == "__main__":
    con = duckdb.connect()
    con.execute(f"CREATE VIEW s AS SELECT * FROM {SRC}")

    vc = con.execute(" UNION ALL ".join(
        f"SELECT year, '{c}' AS field, CAST({c} AS VARCHAR) AS value, count(*) AS n FROM s GROUP BY ALL" for c in CAT
    )).df()
    vc.to_parquet(ROOT / "data/interim/sparcs_value_counts.parquet", index=False)

    by_year = con.execute(f"SELECT year, {METRICS} FROM s GROUP BY year ORDER BY year").df()
    by_year.to_csv(ROOT / "data/interim/sparcs_quality_by_year.csv", index=False)
    pd.set_option("display.width", 250)
    print(by_year.set_index("year").T.round(4).to_string())

    fac = con.execute(f"SELECT year, pfi, any_value(facility_name) AS facility_name, {METRICS} "
                      "FROM s WHERE pfi IS NOT NULL GROUP BY year, pfi").df()
    fac.to_parquet(ROOT / "data/interim/sparcs_quality_by_facility.parquet", index=False)

    # Race-unknown: how much of the record-level variance lies between facilities (per year)?
    icc = con.execute("""
      WITH f AS (SELECT year, pfi, count(*) n,
                        avg((race IS NULL OR race ILIKE '%unknown%' OR race ILIKE 'other%')::INT) p
                 FROM s WHERE pfi IS NOT NULL GROUP BY ALL),
           t AS (SELECT year, sum(n*p)/sum(n) pbar, sum(n) N FROM f GROUP BY year)
      SELECT f.year, t.pbar AS p_unknown_or_other,
             sum(f.n * power(f.p - t.pbar, 2)) / (t.pbar * (1 - t.pbar) * t.N) AS between_facility_share,
             quantile_cont(f.p, 0.1) AS p10_facility, quantile_cont(f.p, 0.5) AS p50_facility,
             quantile_cont(f.p, 0.9) AS p90_facility
      FROM f JOIN t USING (year) GROUP BY f.year, t.pbar, t.N ORDER BY f.year
    """).df()
    icc.to_csv(ROOT / "data/interim/sparcs_race_icc.csv", index=False)
    print(icc.round(3).to_string())
