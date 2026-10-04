"""Texas PUDF 2016-2019: record-level inpatient spinal fusion cohort for the TEAM selection simulation.

Cohort: Medicare FFS (FIRST_PAYMENT_SRC MA/MB) and Medicare Advantage (16) discharges in cervical fusion MS-DRGs
471-473 and noncervical fusion MS-DRGs 453-460 (pre-FY2025 grouper), all ages recorded; analysis restricts later.
Fields: year, hospital, payer, age band, sex, MS-DRG, APR severity of illness (1-4), APR risk of mortality,
procedure family from the principal ICD-10-PCS code (7th character approach/column for fusion roots), number of
secondary diagnoses and number present on admission, pipe-delimited principal+secondary diagnosis codes (for V28 HCC
construction), discharge status, length of stay, total charges.
Procedure families:
  cervical_anterior   0RG1/0RG2/0RG4 with 7th character 0 (ACDF-type)
  cervical_posterior  0RG0/0RG1/0RG2/0RG4 with 7th character 1 or J (posterior cervical fusion, CPT 22600-type)
  lumbar_anterior     0SG0/0SG1/0SG3 with 7th character 0 (ALIF-type, CPT 22558)
  lumbar_posterior    0SG0/0SG1/0SG3 with 7th character 1 or J (posterolateral/TLIF/PLIF)
  thoracic_other      other 0RG fusion roots
Restricted data (Texas DUA): tx_fusion_records.parquet is record-level, stays local under data/interim (git-ignored)
and must not be shared; only aggregates with cells >10 are reported.
Output: data/interim/texas/tx_fusion_records.parquet
"""
import importlib

import duckdb
import pandas as pd

from common import ROOT

tx = importlib.import_module("26_texas")
OUT = ROOT / "data/interim/texas/tx_fusion_records.parquet"

FAMILY = """
  CASE
    WHEN regexp_matches(pc, '^0RG[124]..0') THEN 'cervical_anterior'
    WHEN regexp_matches(pc, '^0RG[0124]..[1J]') THEN 'cervical_posterior'
    WHEN regexp_matches(pc, '^0SG[013]..0') THEN 'lumbar_anterior'
    WHEN regexp_matches(pc, '^0SG[013]..[1J]') THEN 'lumbar_posterior'
    WHEN regexp_matches(pc, '^0RG') THEN 'thoracic_other'
    ELSE 'other'
  END"""

if __name__ == "__main__":
    con = duckdb.connect()
    diag = [f"OTH_DIAG_CODE_{i}" for i in range(1, 25)]
    poa = [f"POA_OTH_DIAG_CODE_{i}" for i in range(1, 25)]
    n_dx = " + ".join(f"CASE WHEN nullif(trim({c}), '') IS NOT NULL THEN 1 ELSE 0 END" for c in diag)
    n_poa = " + ".join(f"CASE WHEN upper(trim({c})) = 'Y' THEN 1 ELSE 0 END" for c in poa)
    dx_all = "concat_ws('|', upper(trim(PRINC_DIAG_CODE)), " + ", ".join(f"upper(trim({c}))" for c in diag) + ")"
    surg = [f"OTH_SURG_PROC_CODE_{i}" for i in range(1, 25)]
    pcs_all = "concat_ws('|', upper(trim(PRINC_SURG_PROC_CODE)), " + ", ".join(f"upper(trim({c}))" for c in surg) + ")"
    parts = []
    for f in tx.base_files():
        cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_csv('{f}', delim='\t', header=true, all_varchar=true, sample_size=2000, quote='', escape='', strict_mode=false, null_padding=true)").fetchall()}
        poa_expr = n_poa if all(c in cols for c in poa) else "NULL"
        q = f"""
          WITH b AS (
            SELECT left(DISCHARGE, 4)::INT AS year, THCIC_ID AS hosp, FIRST_PAYMENT_SRC AS pay,
                   TRY_CAST(PAT_AGE AS INT) AS age_code, SEX_CODE AS sex, TRY_CAST(MS_DRG AS INT) AS msdrg,
                   TRY_CAST(ILLNESS_SEVERITY AS INT) AS soi, TRY_CAST(RISK_MORTALITY AS INT) AS rom,
                   upper(trim(PRINC_SURG_PROC_CODE)) AS pc, TRY_CAST(LENGTH_OF_STAY AS DOUBLE) AS los,
                   TRY_CAST(TOTAL_CHARGES AS DOUBLE) AS charges, ({n_dx}) AS n_dx, ({poa_expr}) AS n_poa,
                   {dx_all} AS dx, {pcs_all} AS pcs, TRY_CAST(PAT_STATUS AS VARCHAR) AS pat_status
            FROM read_csv('{f}', delim='\t', header=true, all_varchar=true, quote='', escape='', strict_mode=false, null_padding=true))
          SELECT year, hosp, CASE WHEN pay IN ('MA','MB') THEN 'ffs' WHEN pay = '16' THEN 'ma' END AS payer,
                 age_code, sex, msdrg, soi, rom, {FAMILY} AS family, los, charges, n_dx, n_poa, dx, pcs, pat_status
          FROM b
          WHERE pay IN ('MA','MB','16') AND msdrg IN (453,454,455,456,457,458,459,460,471,472,473) AND year BETWEEN 2016 AND 2019"""
        parts.append(con.execute(q).df())
        print("read", f.split("/")[-1], len(parts[-1]), flush=True)
    d = pd.concat(parts, ignore_index=True)
    d = d.sort_values(list(d.columns), kind="mergesort", na_position="last").reset_index(drop=True)  # fixed row order
    d.to_parquet(OUT, index=False)
    print(len(d), d.groupby(["payer", "family"]).size().to_string())
