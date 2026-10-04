"""Build a national hospital x calendar-year reference panel from CMS HCRIS cost report PUFs.

Cost reports cover hospital fiscal years (often partial or overlapping), so each report's
counts are allocated to calendar years in proportion to days of overlap.
Output:
  data/interim/hcris_hospital_cy.parquet  (ccn, year, state, type, discharges, days, beds, ...)
  data/interim/hcris_state_cy.parquet     (state, year, n_hospitals, discharges)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hcris"
OUT = ROOT / "data" / "interim"
OUT.mkdir(parents=True, exist_ok=True)

KEEP = {
    "rpt_rec_num": "rpt", "Provider CCN": "ccn", "Hospital Name": "name", "Street Address": "address",
    "City": "city", "State Code": "state", "Zip Code": "zip", "County": "county",
    "Rural Versus Urban": "rural_urban", "CCN Facility Type": "fac_type", "Provider Type": "provider_type",
    "Type of Control": "control", "Fiscal Year Begin Date": "fy_begin", "Fiscal Year End Date": "fy_end",
    "Number of Beds": "beds", "Total Days (V + XVIII + XIX + Unknown)": "days",
    "Total Discharges (V + XVIII + XIX + Unknown)": "discharges",
    "Total Discharges Title XVIII": "discharges_medicare",
    "Total Discharges Title XIX": "discharges_medicaid",
    "Hospital Total Discharges (V + XVIII + XIX + Unknown) For Adults & Peds": "discharges_adult_peds",
    "Inpatient Total Charges": "ip_charges", "Number of Interns and Residents (FTE)": "resident_fte",
}
COUNTS = ["discharges", "discharges_medicare", "discharges_medicaid", "discharges_adult_peds", "days", "ip_charges"]


def load() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("CostReport_*_Final.csv")):
        d = pd.read_csv(f, usecols=list(KEEP), dtype=str).rename(columns=KEEP)
        d["file_year"] = int(f.stem.split("_")[1])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("file_year").drop_duplicates("rpt", keep="last")
    for c in COUNTS + ["beds", "resident_fte"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fy_begin"] = pd.to_datetime(df["fy_begin"], format="%m/%d/%Y", errors="coerce")
    df["fy_end"] = pd.to_datetime(df["fy_end"], format="%m/%d/%Y", errors="coerce")
    df["ccn"] = df["ccn"].str.zfill(6)
    df["zip5"] = df["zip"].str[:5]
    return df.dropna(subset=["fy_begin", "fy_end"])


def calendarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples(index=False):
        span = (r.fy_end - r.fy_begin).days + 1
        if span <= 0:
            continue
        for y in range(r.fy_begin.year, r.fy_end.year + 1):
            lo = max(r.fy_begin, pd.Timestamp(y, 1, 1))
            hi = min(r.fy_end, pd.Timestamp(y, 12, 31))
            overlap = (hi - lo).days + 1
            if overlap > 0:
                rows.append((r.ccn, y, r.rpt, overlap / span, overlap))
    alloc = pd.DataFrame(rows, columns=["ccn", "year", "rpt", "w", "days_covered"])
    m = alloc.merge(df, on=["ccn", "rpt"])
    for c in COUNTS:
        m[c] = m[c] * m["w"]
    meta = ["name", "address", "city", "state", "zip5", "county", "rural_urban", "fac_type",
            "provider_type", "control", "beds", "resident_fte"]
    agg = (m.sort_values("fy_end")
             .groupby(["ccn", "year"])
             .agg(**{c: (c, "sum") for c in COUNTS},
                  days_covered=("days_covered", "sum"),
                  n_reports=("rpt", "nunique"),
                  **{c: (c, "last") for c in meta})
             .reset_index())
    agg["coverage_frac"] = agg["days_covered"] / np.where(agg["year"] % 4 == 0, 366, 365)
    return agg


if __name__ == "__main__":
    df = load()
    print(f"{len(df):,} unique cost reports, {df.ccn.nunique():,} CCNs")
    hosp = calendarize(df)
    hosp.to_parquet(OUT / "hcris_hospital_cy.parquet", index=False)
    full = hosp[(hosp.coverage_frac > 0.95) & hosp.year.between(2011, 2023)]
    state = (full.groupby(["state", "year", "fac_type"])
                 .agg(n_hosp=("ccn", "nunique"), discharges=("discharges", "sum"))
                 .reset_index())
    state.to_parquet(OUT / "hcris_state_cy.parquet", index=False)
    print(hosp.groupby("year").agg(n=("ccn", "nunique"), full=("coverage_frac", lambda s: (s > 0.95).mean()),
                                   dis=("discharges", "sum")).tail(15))
