"""National Medicare FFS discharges and average payments by MS-DRG, from CMS 'Medicare Inpatient Hospitals by Geography
and Service' (the National row of each year's file).

national_drg_totals.csv: year, drg, n (total discharges), Avg_Mdcr_Pymt_Amt, for the spine fusion and nonfusion MS-DRGs
(453-460 through FY2024, 402, 426-430, 447-448 and 450-451 from FY2025, 471-473, 518-520) and craniotomy 28-30, 2013-2024.
national_drg_payments.csv: year, drg, discharges, avg_medicare_payment, avg_total_payment for MS-DRGs 453-460 and 471-473
in 2019, 2022 and 2023, the payment years the TEAM simulation draws on.
Outputs: data/interim/national_drg_totals.csv, data/interim/national_drg_payments.csv
"""
import re

import pandas as pd

from common import ROOT

GEO = ROOT / "data/raw/cms_mupinp_geo"
I = ROOT / "data/interim"
TOTAL_DRGS = [28, 29, 30, 402, 426, 427, 428, 429, 430, 447, 448, 450, 451, 453, 454, 455, 456, 457, 458, 459, 460,
              471, 472, 473, 518, 519, 520]
PAY_DRGS = [453, 454, 455, 456, 457, 458, 459, 460, 471, 472, 473]
PAY_YEARS = [2019, 2022, 2023]


def national():
    rows = []
    for f in sorted(GEO.glob("MUP_*.CSV")) + sorted(GEO.glob("MUP_*.csv")):
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        d = d[d.Rndrng_Prvdr_Geo_Lvl.str.strip() == "National"]
        rows.append(pd.DataFrame({"year": y, "drg": pd.to_numeric(d.DRG_Cd), "n": pd.to_numeric(d.Tot_Dschrgs),
                                  "Avg_Mdcr_Pymt_Amt": pd.to_numeric(d.Avg_Mdcr_Pymt_Amt),
                                  "Avg_Tot_Pymt_Amt": pd.to_numeric(d.Avg_Tot_Pymt_Amt)}))
    return pd.concat(rows).drop_duplicates()


if __name__ == "__main__":
    nat = national()
    tot = nat[nat.drg.isin(TOTAL_DRGS)].sort_values(["year", "drg"])
    tot[["year", "drg", "n", "Avg_Mdcr_Pymt_Amt"]].to_csv(I / "national_drg_totals.csv", index=False)
    pay = nat[nat.drg.isin(PAY_DRGS) & nat.year.isin(PAY_YEARS)].sort_values(["year", "drg"])
    pay = pay.rename(columns={"n": "discharges", "Avg_Mdcr_Pymt_Amt": "avg_medicare_payment", "Avg_Tot_Pymt_Amt": "avg_total_payment"})
    pay[["year", "drg", "discharges", "avg_medicare_payment", "avg_total_payment"]].to_csv(I / "national_drg_payments.csv", index=False)
    print(len(tot), "total rows;", len(pay), "payment rows")
