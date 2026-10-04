"""Inpatient-Only (IPO) list status by year from CMS OPPS final rule Addendum E (CY2019-CY2024).

Source files: data/raw/cms_opps_addenda/opps_addenda_<year>.zip (CMS OPPS final rule addenda, CY2019-CY2024),
extracted to addendum_e/<year>/*.txt (tab-delimited, Latin-1). A code is on the IPO list in a year if it appears
in that year's Addendum E.
Output: data/interim/ipo_status_by_year.csv (cpt x year, on_ipo) and printed wide table for study codes.
"""
import glob
import re

import pandas as pd

from common import ROOT

RAW = ROOT / "data/raw/cms_opps_addenda/addendum_e"
STUDY = ["22551", "22552", "22554", "22856", "22558", "22595", "22600", "22612", "22630", "22632", "22633", "22634",
         "63045", "63047", "63050", "63051", "63030", "62223", "61510", "61512", "61518", "61154", "61312", "61645", "63265"]


def ipo_codes(year):
    files = glob.glob(str(RAW / str(year) / "*.txt"))
    if not files:
        return None
    text = open(files[0], "rb").read().decode("latin-1")
    return {m.group(1) for m in re.finditer(r"^\s*\"?([0-9]{4}[0-9T]|[A-Z][0-9]{4})\"?\t", text, flags=re.M)}


if __name__ == "__main__":
    rows = []
    for y in range(2019, 2025):
        codes = ipo_codes(y)
        if codes is None:
            print("missing", y)
            continue
        print(y, "IPO codes:", len(codes))
        rows += [{"cpt": c, "year": y, "on_ipo": c in codes} for c in STUDY]
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data/interim/ipo_status_by_year.csv", index=False)
    print(d.pivot_table(index="cpt", columns="year", values="on_ipo", aggfunc="first").replace({True: "IPO", False: "."}).to_string())
