"""Rechecks of two 2026 provider-file series found by the structured search (eMethods 12).

Dinesh et al, Spine 2026 (PMIDs 42531462 and 42585156) queried the CMS Medicare Physician & Other Practitioners
by Provider and Service file for ACDF (CPT 22551) and lumbar laminectomy (CPT 63047) billed by neurosurgeons and
orthopaedic surgeons, 2013 to 2023, and reported utilization per 10 000 FFS beneficiaries falling 22.9%
(5.63 to 4.34) and 18.1% (24.14 to 19.78). That file removes any provider-code cell under 11 beneficiaries.
We total the published rows for the same codes and specialties and set them against the complete national
Part B count for the same codes (PSPS, primary-surgeon services, all specialties), both per FFS Part B enrollee.

Output: data/interim/dinesh_recheck.csv
"""
import importlib
import time

import pandas as pd
import requests

from common import ROOT

r80 = importlib.import_module("80_lopez_acdf_miller_recheck")
s23, s79 = r80.s23, r80.s79
API = "https://data.cms.gov/data-api/v1/dataset/{}/data"
CODES = {"22551": "ACDF (22551)", "63047": "Lumbar laminectomy (63047)"}
REPORTED = {"22551": (5.63, 4.34), "63047": (24.14, 19.78)}
YEARS = (2013, 2023)


def distributions(kind="by Provider and Service"):
    """CMS lists each data year as its own catalog entry."""
    j = requests.get("https://data.cms.gov/data.json", timeout=120).json()
    out = {}
    for d in j["dataset"]:
        t = d.get("title", "")
        for y in YEARS:
            if t.startswith(f"Medicare Physician & Other Practitioners - {kind} : {y}"):
                u = next(x.get("accessURL") or "" for x in d["distribution"] if "data-api" in (x.get("accessURL") or ""))
                out[y] = u.split("/dataset/")[1].split("/")[0]
    return out


def pull(dist, code, **flt):
    rows, offset = [], 0
    while True:
        r = requests.get(API.format(dist), timeout=180, params={"filter[HCPCS_Cd]": code, "size": 5000, "offset": offset, **{f"filter[{k}]": v for k, v in flt.items()}})
        r.raise_for_status()
        b = r.json()
        rows.extend(b)
        if len(b) < 5000:
            return rows
        offset += 5000
        time.sleep(0.2)


if __name__ == "__main__":
    import duckdb
    con = duckdb.connect()
    f0 = sorted((ROOT / "data/raw/cms_psps").glob("*.csv"))[0]
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_csv('{f0}', all_varchar=true, header=true, sample_size=2000)").fetchall()]
    hc = s23.colmap(cols)["HCPCS_CD"]
    enr = s79.ffs_enrollment().B_ORGNL_MDCR_BENES
    enr_any = s79.ffs_enrollment().ORGNL_MDCR_BENES      # Original Medicare, Part A or Part B: the study's denominator
    dist, geo = distributions(), distributions("by Geography and Service")
    print("distributions:", dist, geo, flush=True)
    out = []
    for code, lab in CODES.items():
        ps = r80.psps_by_setting(f'"{hc}" = \'{code}\'', lab).groupby("year").services.sum()
        for y in YEARS:
            rows = pd.DataFrame(pull(dist[y], code))
            spec = rows.Rndrng_Prvdr_Type.str.lower()
            sp = rows[spec.str.contains("neurosurg") | spec.str.contains("orthop")]
            nat = pd.DataFrame(pull(geo[y], code, Rndrng_Prvdr_Geo_Lvl="National"))   # the file family's own national total, suppressed cells included
            out.append({"procedure": lab, "year": y, "national_services_geography_file": nat.Tot_Srvcs.astype(float).sum(), "published_providers": len(rows), "published_services_all_specialties": rows.Tot_Srvcs.astype(float).sum(),
                        "published_surgeons_ns_ortho": len(sp), "published_services_ns_ortho": sp.Tot_Srvcs.astype(float).sum(),
                        "complete_services_psps": ps.loc[y], "ffs_partb_enrollees": enr.loc[y], "ffs_any_part_enrollees": enr_any.loc[y]})
            print(out[-1], flush=True)
    d = pd.DataFrame(out)
    d["published_per_10k"] = d.published_services_ns_ortho / d.ffs_partb_enrollees * 1e4
    d["complete_per_10k"] = d.complete_services_psps / d.ffs_partb_enrollees * 1e4
    d["published_share_of_complete"] = d.published_services_ns_ortho / d.complete_services_psps
    d["published_per_10k_study_denominator"] = d.published_services_ns_ortho / d.ffs_any_part_enrollees * 1e4
    d["complete_per_10k_study_denominator"] = d.complete_services_psps / d.ffs_any_part_enrollees * 1e4
    d["national_per_10k"] = d.national_services_geography_file / d.ffs_partb_enrollees * 1e4
    d["reported_per_10k"] = [REPORTED[c][i] for c in CODES for i in (0, 1)]
    d.to_csv(ROOT / "data/interim/dinesh_recheck.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.round(3).to_string(index=False))
    for lab, g in d.groupby("procedure"):
        a, b = g.iloc[0], g.iloc[1]
        print(f"\n{lab}: reported {b.reported_per_10k / a.reported_per_10k - 1:+.1%}; published rows per enrollee {b.published_per_10k / a.published_per_10k - 1:+.1%}; "
              f"complete (PSPS) per enrollee {b.complete_per_10k / a.complete_per_10k - 1:+.1%}; on the study's denominator: published {b.published_per_10k_study_denominator / a.published_per_10k_study_denominator - 1:+.1%}, complete {b.complete_per_10k_study_denominator / a.complete_per_10k_study_denominator - 1:+.1%}; national geography file per enrollee {b.national_per_10k / a.national_per_10k - 1:+.1%}; published count {b.published_services_ns_ortho / a.published_services_ns_ortho - 1:+.1%}; "
              f"complete count {b.complete_services_psps / a.complete_services_psps - 1:+.1%}; surgeons {b.published_surgeons_ns_ortho / a.published_surgeons_ns_ortho - 1:+.1%}")
