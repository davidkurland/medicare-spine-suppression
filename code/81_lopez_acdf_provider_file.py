"""Reproduce Lopez et al's ACDF series from the Part B provider-level file it was built on.

Lopez CD, Boddapati V, Lombardi JM, et al. Recent trends in Medicare utilization and reimbursement for
anterior cervical discectomy and fusion. Spine J. 2020;20(11):1737-1743. Reported a 24.2% rise in annual
ACDF volume, 38 877 in 2012 to 48 290 in 2017, from the Physician and Other Supplier public use file.
That file removes any provider-HCPCS cell under 11 services, so its coverage moves with volume.

Source: CMS Medicare Physician & Other Practitioners by Provider and Service, data.cms.gov API. The API
series begins in 2013, 1 year after the study window opens, the same limitation as in the Lopez
discectomy recheck (96_published_paper_checks.py).

Output: data/interim/lopez_acdf_provider_file.csv
"""
import time

import pandas as pd
import requests

from common import ROOT

CODES = ["22551", "22552", "22554", "22585"]
DIST = {2013: "0d5c0f9c-ff36-4f8d-9d2f-4b0e5a1a5b07", 2014: "f63b48ae-946e-48f7-9f56-327a68da4e0b",
        2015: "f8cdb11a-d5f7-4fbe-aac4-05abc8ee2c83", 2016: "7918e22a-fbfb-4a07-9f59-f8aab2b757d4",
        2017: "85bf3c9c-2244-490d-ad7d-c34e4c28f8ea"}
API = "https://data.cms.gov/data-api/v1/dataset/{}/data"


def resolve_2013():
    """Look up the 2013 distribution id in the CMS catalog; it replaces the stored 2013 id in DIST when found."""
    j = requests.get("https://data.cms.gov/data.json", timeout=90).json()
    ds = next(d for d in j["dataset"]
              if d.get("title") == "Medicare Physician & Other Practitioners - by Provider and Service")
    for x in ds["distribution"]:
        u = x.get("accessURL") or x.get("downloadURL") or ""
        if "data-api" in u and str(x.get("title", "")).find("2013") >= 0:
            return u.split("/dataset/")[1].split("/")[0]
    return None


def pull(dist, code):
    rows, offset = [], 0
    while True:
        r = requests.get(API.format(dist), timeout=120,
                         params={"filter[HCPCS_Cd]": code, "size": 5000, "offset": offset})
        r.raise_for_status()
        b = r.json()
        if not b:
            break
        rows.extend(b)
        if len(b) < 5000:
            break
        offset += 5000
        time.sleep(0.2)
    return rows


if __name__ == "__main__":
    d13 = resolve_2013()
    if d13:
        DIST[2013] = d13
    out = []
    for year, dist in sorted(DIST.items()):
        for code in CODES:
            rows = pull(dist, code)
            svc = sum(float(x.get("Tot_Srvcs") or 0) for x in rows)
            ben = sum(float(x.get("Tot_Benes") or 0) for x in rows)
            out.append({"year": year, "cpt": code, "providers_published": len(rows),
                        "services": svc, "beneficiaries": ben})
            print(f"{year} {code}: {len(rows):>5} published providers, {svc:>10,.0f} services", flush=True)
    d = pd.DataFrame(out)
    d.to_csv(ROOT / "data/interim/lopez_acdf_provider_file.csv", index=False)

    t = d.groupby("year")[["providers_published", "services"]].sum()
    print("\nACDF in the Part B provider file (published rows only)\n")
    print(t.round(0).to_string())
    a, b = t.index.min(), t.index.max()
    print(f"\n  {a} -> {b}: published services {100*(t.services[b]/t.services[a]-1):+.1f}%,"
          f" published providers {100*(t.providers_published[b]/t.providers_published[a]-1):+.1f}%")
    print("  Lopez reported +24.2% (2012 to 2017) on this file.")
