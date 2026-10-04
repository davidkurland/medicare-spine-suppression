"""Where are Indiana's Medicare neurosurgical inpatients in CMS's public data? (2021-2024)

Indiana's payer field separates Medicare FFS from Medicare Advantage, and its MS-DRG counts use the
same grouper as CMS, so every state-recorded Medicare inpatient can be assigned to one of:
  published    - FFS case in a hospital x MS-DRG cell CMS publishes
  suppressed   - FFS case in a cell CMS does not publish (< 11 discharges)
  unlinked     - FFS case at a hospital with no CMS link in our crosswalk
  advantage    - Medicare Advantage enrollee (never in the FFS file)
plus the CMS-reported count in published cells, for comparison with 'published'.
Output: data/interim/IN_medicare_decomposition.csv
"""
import pandas as pd

from common import ROOT

SC = ROOT / "data/interim/state_compare"

if __name__ == "__main__":
    pm = pd.read_csv(SC / "IN_neuro_payer_mix.csv")
    cells = pd.read_csv(SC / "IN_neuro_vs_cms_cells.csv", dtype={"ccn": str})
    rows = []
    for (y, g), c in cells[cells.year >= 2021].groupby(["year", "grp"]):
        p = pm[(pm.year == y) & (pm.grp == g)].iloc[0]
        linked = c.state_medicare.sum()
        pub = c[c.cms_ffs.notna()]
        r = {"year": y, "grp": g,
             "advantage": p.medicare_adv,
             "published": pub.state_medicare.fillna(0).sum(),
             "suppressed": c.loc[c.cms_ffs.isna(), "state_medicare"].sum(),
             "unlinked": p.medicare_ffs - linked,
             "cms_reported": pub.cms_ffs.sum()}
        r["total_medicare"] = r["advantage"] + p.medicare_ffs
        for k in ["advantage", "published", "suppressed", "unlinked", "cms_reported"]:
            r[f"{k}_share"] = r[k] / r["total_medicare"]
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/interim/IN_medicare_decomposition.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
