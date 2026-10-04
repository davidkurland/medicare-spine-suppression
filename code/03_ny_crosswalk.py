"""Crosswalk NY SPARCS permanent facility IDs (PFI) to CMS CCNs.

Strategy: PFI -> NYSDOH registry (street address, zip, name) -> HCRIS NY hospitals. Every NY
HCRIS hospital is a candidate for each PFI (psychiatric hospitals only for facilities with a psychiatric name); candidates are scored by name similarity plus
street-address agreement, with smaller bonuses for matching zip5 and city. Several PFIs can
share one CCN (multi-campus systems file one cost report), so the comparison is made at the
CCN level.
Inputs: data/raw/ny_ivw2-k53g.csv (health.data.ny.gov hospital inpatient discharges facility
aggregate, Socrata dataset ivw2-k53g); data/raw/ny_vn5v-hh5r.csv (NYSDOH health facility
registry, Socrata dataset vn5v-hh5r); data/interim/hcris_hospital_cy.parquet (02_hcris_national.py).
Output: data/interim/ny_pfi_ccn_crosswalk.csv (with scores, for manual review)
"""
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOP = {"hospital", "center", "medical", "the", "of", "inc", "and", "health", "campus", "division",
        "memorial", "community", "system", "llc", "corp", "ny", "new", "york", "at", "st", "saint"}
STREET_STOP = {"street", "st", "avenue", "ave", "road", "rd", "drive", "dr", "boulevard", "blvd",
               "place", "pl", "east", "west", "north", "south", "e", "w", "n", "s", "lane", "ln"}


def tokens(s, stop=STOP) -> set:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return {t for t in s.split() if t not in stop}


def name_score(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    jacc = len(ta & tb) / max(1, len(ta | tb))
    seq = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return 0.5 * jacc + 0.5 * seq


def addr_score(a: str, b: str) -> float:
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    na, nb = re.findall(r"^\s*(\d+)", a), re.findall(r"^\s*(\d+)", b)
    ta, tb = tokens(a, STREET_STOP), tokens(b, STREET_STOP)
    ta = {t for t in ta if not t.isdigit()}
    tb = {t for t in tb if not t.isdigit()}
    street = bool(ta & tb)
    num = bool(na and nb and na[0] == nb[0])
    return 1.0 if (num and street) else 0.5 if (num or street) else 0.0


if __name__ == "__main__":
    fa = pd.read_csv(ROOT / "data/raw/ny_ivw2-k53g.csv")
    pfi = (fa.groupby("Facility ID")
             .agg(sparcs_name=("Hospital Name", "last"), first=("Discharge Year", "min"),
                  last=("Discharge Year", "max"), discharges=("Number of Discharges", "sum"))
             .reset_index().rename(columns={"Facility ID": "pfi"}))
    reg = pd.read_csv(ROOT / "data/raw/ny_vn5v-hh5r.csv", dtype=str)
    reg = reg.assign(pfi=pd.to_numeric(reg["Facility ID"], errors="coerce"))
    reg = reg[["pfi", "Facility Name", "Facility Address 1", "Facility Zip Code", "Facility City", "Short Description"]]
    reg.columns = ["pfi", "reg_name", "reg_address", "zip5", "city", "reg_type"]
    reg["zip5"] = reg["zip5"].str[:5]
    reg["city"] = reg["city"].str.upper().str.strip()
    pfi = pfi.merge(reg.drop_duplicates("pfi"), on="pfi", how="left")

    h = pd.read_parquet(ROOT / "data/interim/hcris_hospital_cy.parquet")
    # keep every CCN ever seen: hospitals get new CCNs (e.g. conversion to critical access status)
    h = h[h.state == "NY"].sort_values("year").drop_duplicates("ccn", keep="last")
    h = h[["ccn", "name", "address", "city", "zip5", "fac_type"]].rename(
        columns={"name": "cms_name", "city": "cms_city", "address": "cms_address"})
    h["cms_city"] = h["cms_city"].str.upper().str.strip()

    rows = []
    for r in pfi.itertuples(index=False):
        psych = bool(re.search(r"psych|behavioral|mental", str(r.sparcs_name), re.I))
        pool = h if psych else h[h.fac_type != "PH"]
        # statewide candidates: zip codes often disagree between the NY registry and HCRIS
        cand = pool
        names = [r.sparcs_name] + ([r.reg_name] if isinstance(r.reg_name, str) else [])
        ns = cand.apply(lambda c: max(name_score(n, c.cms_name) for n in names), axis=1)
        ad = cand.cms_address.apply(lambda a: addr_score(r.reg_address, a))
        geo = 0.25 * (cand.zip5 == r.zip5) + 0.1 * (cand.cms_city == r.city)
        total = ns + 0.6 * ad + geo
        best = cand.loc[total.idxmax()]
        rows.append({**r._asdict(), "ccn": best.ccn, "cms_name": best.cms_name, "cms_address": best.cms_address,
                     "cms_zip": best.zip5, "cms_type": best.fac_type, "name_score": round(ns[total.idxmax()], 3),
                     "addr_score": ad[total.idxmax()], "total": round(total.max(), 3)})
        # secondary CCNs for the same site (CCN changes over time): same street address + similar name
        extra = cand[(ad == 1.0) & (ns >= 0.4) & (cand.ccn != best.ccn)]
        for c in extra.itertuples():
            rows.append({**rows[-1], "ccn": c.ccn, "cms_name": c.cms_name, "cms_address": c.cms_address,
                         "cms_zip": c.zip5, "cms_type": c.fac_type, "secondary": True})
    xw = pd.DataFrame(rows)
    xw["secondary"] = xw.get("secondary", False).fillna(False)
    xw["same_zip"] = xw.zip5 == xw.cms_zip
    xw["match_ok"] = (xw.addr_score == 1.0) | (xw.name_score >= 0.55) | ((xw.name_score >= 0.4) & xw.same_zip)
    xw.to_csv(ROOT / "data/interim/ny_pfi_ccn_crosswalk.csv", index=False)
    print("secondary CCN links:", xw.secondary.sum())
    xw = xw[~xw.secondary]
    share = xw.loc[xw.match_ok, "discharges"].sum() / xw.discharges.sum()
    print(f"{len(xw)} PFIs; accepted {xw.match_ok.sum()} ({share:.1%} of discharges)")
    cols = ["pfi", "sparcs_name", "reg_address", "cms_name", "cms_address", "name_score", "addr_score", "match_ok", "last", "discharges"]
    print(xw[~xw.match_ok].sort_values("discharges", ascending=False)[cols].head(40).to_string())
