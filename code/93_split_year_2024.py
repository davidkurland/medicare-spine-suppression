"""Calendar year 2024 in the CMS inpatient files spans 2 MS-DRG groupers.

The FY2025 grouper took effect for discharges on or after October 1, 2024, so the 2024 data year holds about 9 months
under the outgoing fusion MS-DRGs (453-455, 459-460) and about 3 months under the FY2025 MS-DRGs (402, 426-430, 447-448,
450-451). Each hospital's fusion volume for the year is therefore divided between 2 sets of cells, and each cell is
compared with the 11-discharge threshold on a part-year count. This script reports published and complete discharges
and the hidden share for each set of rows in 2023 and 2024.
Output: data/interim/split_year_2024.csv
"""
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
GROUPS = [("Fusion MS-DRGs retired by FY2025 (453-455, 459-460)", (453, 454, 455, 459, 460)),
          ("FY2025 fusion MS-DRGs (402, 426-430, 447-448, 450-451)", (402, 426, 427, 428, 429, 430, 447, 448, 450, 451)),
          ("Fusion MS-DRGs unchanged by FY2025 (456-458, 471-473)", (456, 457, 458, 471, 472, 473)),
          ("Nonfusion spine MS-DRGs (028-030, 518-520)", (28, 29, 30, 518, 519, 520))]

if __name__ == "__main__":
    r = pd.read_parquet(I / "cms_neuro_drg_rows.parquet")
    t = pd.read_csv(I / "national_drg_totals.csv")
    rows = []
    for y in (2023, 2024):
        for lab, drgs in GROUPS + [("All spine", tuple(d for _, g in GROUPS for d in g))]:
            p = r[(r.year == y) & r.drg.isin(drgs)]
            tot = t[(t.year == y) & t.drg.isin(drgs)].n.sum()
            if tot == 0:
                continue
            rows.append({"year": y, "rows": lab, "msdrgs_with_discharges": t[(t.year == y) & t.drg.isin(drgs) & (t.n > 0)].drg.nunique(),
                         "complete": tot, "published": p.n.sum(), "hidden": tot - p.n.sum(), "hidden_share": 1 - p.n.sum() / tot,
                         "published_cells": len(p), "median_published_cell": p.n.median()})
    out = pd.DataFrame(rows)
    a = out.set_index(["year", "rows"])
    old, new = GROUPS[0][0], GROUPS[1][0]
    out["share_of_2024_split_volume"] = [r_.complete / (a.loc[(2024, old), "complete"] + a.loc[(2024, new), "complete"]) if (r_.year == 2024 and r_.rows in (old, new)) else float("nan") for r_ in out.itertuples()]
    out.to_csv(I / "split_year_2024.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    nat = pd.read_csv(I / "suppression_exact_national.csv")
    s = nat[(nat.level == "group") & (nat.stratum == "spine") & nat.year.isin([2023, 2024])]
    print(s[["year", "total_ffs", "published"]].to_string(index=False))
