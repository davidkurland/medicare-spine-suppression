"""Harmonise the 16 SPARCS PUF years into one canonical schema (DuckDB, lazy over parquet).

Column names drift across releases (facility_id vs permanent_facility_id, age vs age_group,
gender vs sex, source_of_payment vs payment_typology, CCS vs CCSR, zip3 vs zip). This script maps
every year onto one schema, records which source column fed each field, and writes:
  data/interim/sparcs/year=YYYY/*.parquet
  data/interim/sparcs_schema_map.csv   (year x canonical field -> source column / missing)
"""
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/sparcs"
OUT = ROOT / "data/interim/sparcs"

# canonical -> candidate normalised source names, in priority order
CANON = {
    "hsa": ["health_service_area", "hospital_service_area"],
    "hospital_county": ["hospital_county"],
    "opcert": ["operating_certificate_number"],
    "pfi": ["permanent_facility_id", "facility_id", "facility_id_"],
    "facility_name": ["facility_name"],
    "age_group": ["age_group", "age"],
    "zip3": ["zip_code_3_digits", "zip_code"],
    "sex": ["gender", "sex"],
    "race": ["race"],
    "ethnicity": ["ethnicity"],
    "los_raw": ["length_of_stay"],
    "admit_type": ["type_of_admission"],
    "disposition": ["patient_disposition"],
    "discharge_year": ["discharge_year"],
    "dx_code": ["ccsr_diagnosis_code", "ccs_diagnosis_code"],
    "dx_desc": ["ccsr_diagnosis_description", "ccs_diagnosis_description"],
    "proc_code": ["ccsr_procedure_code", "ccs_procedure_code"],
    "apr_drg": ["apr_drg_code"],
    "apr_mdc": ["apr_mdc_code"],
    "soi": ["apr_severity_of_illness_code"],
    "rom": ["apr_risk_of_mortality"],
    "med_surg": ["apr_medical_surgical_description", "apr_medical_surgical"],
    "pay1": ["payment_typology_1", "source_of_payment_1"],
    "pay2": ["payment_typology_2", "source_of_payment_2"],
    "birth_weight": ["birth_weight"],
    "abortion_flag": ["abortion_edit_indicator"],
    "ed_flag": ["emergency_department_indicator"],
    "charges_raw": ["total_charges"],
    "costs_raw": ["total_costs"],
}


def norm(c: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", c.lower())).strip("_")


def build(year: int, con) -> list:
    src = RAW / f"sparcs_{year}.parquet"
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{src}'").fetchall()]
    lookup = {norm(c): c for c in cols}
    sel, mapping = [], []
    for canon, cands in CANON.items():
        hit = next((lookup[c] for c in cands if c in lookup), None)
        mapping.append({"year": year, "field": canon, "source": hit})
        sel.append(f'"{hit}" AS {canon}' if hit else f"NULL::VARCHAR AS {canon}")
    dx_system = "'CCSR'" if any(c.startswith("ccsr") for c in lookup) else "'CCS'"
    money = lambda c: f"TRY_CAST(regexp_replace({c}, '[$,]', '', 'g') AS DOUBLE)"
    sql = f"""
      SELECT {year} AS year, {dx_system} AS dx_system,
             * EXCLUDE (los_raw, charges_raw, costs_raw, pfi, soi, zip3, age_group),
             -- 2024 relabelled bands ("0 to 17" -> "0-17"); keep the pre-2024 spelling so years line up
             replace(age_group, '-', ' to ') AS age_group,
             TRY_CAST(ltrim(pfi, '0') AS INTEGER) AS pfi,
             TRY_CAST(soi AS INTEGER) AS soi,
             zip3 AS zip_raw,
             CASE WHEN regexp_matches(zip3, '^[0-9]{{3}}') THEN left(zip3, 3) ELSE NULL END AS zip3,
             los_raw,
             CASE WHEN los_raw LIKE '120%+%' THEN 120 ELSE TRY_CAST(los_raw AS INTEGER) END AS los,
             {money('charges_raw')} AS charges,
             {money('costs_raw')} AS costs
      FROM (SELECT {', '.join(sel)} FROM '{src}')
    """
    out = OUT / f"year={year}"
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{out / 'part.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
    return mapping


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or sorted(int(p.stem.split("_")[1]) for p in RAW.glob("sparcs_*.parquet"))
    con = duckdb.connect()
    maps = []
    for y in years:
        maps += build(y, con)
        print(y, "done", flush=True)
    mp = ROOT / "data/interim/sparcs_schema_map.csv"
    new = pd.DataFrame(maps)
    if mp.exists():
        new = pd.concat([pd.read_csv(mp)[lambda d: ~d.year.isin(years)], new])
    new.sort_values(["year", "field"]).to_csv(mp, index=False)
