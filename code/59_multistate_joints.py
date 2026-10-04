"""Joint replacement and cranial comparator lines for the multistate comparison: all-payer Medicare (HCUPnet SID,
FFS + MA) vs the CMS public FFS files, with the same MS-DRGs in both sources (joint replacement 469 and 470; cranial
23 and 25-27, with and without 23), so the definition is identical on each side.

Same balanced-panel rule as 29_multistate.py: states with continuous unsuppressed HCUPnet coverage across the window
(2015, the ICD-10 transition year, is not required). Reports the change in all-payer Medicare discharges, complete CMS
FFS discharges, and published CMS rows, and the number of states where published volume fell further than all-payer.
Hypothesis: joint replacement's inpatient decline is real, so published and complete should both track the all-payer
fall closely, unlike spine fusion.
Inputs: data/raw/hcupnet/hcupnet_joints_long.csv and hcupnet_cranial_long.csv (from 58_hcupnet_joints.py), CMS
geography and provider files.
Outputs: data/interim/multistate_joints_panel.csv, multistate_cranial_panel.csv, multistate_joints_summary.csv
"""
import importlib
import re

import pandas as pd

from common import ROOT

ms = importlib.import_module("29_multistate")
hs = importlib.import_module("52_hidden_share_team_episodes")
I = ROOT / "data/interim"
LINES = {   # name: (DRGs used identically in HCUPnet and CMS, HCUPnet file)
    "joint replacement (469, 470)": ([469, 470], "hcupnet_joints_long.csv"),
    "cranial (23, 25-27)": ([23, 25, 26, 27], "hcupnet_cranial_long.csv"),
    "cranial without thrombectomy-heavy 23 (25-27)": ([25, 26, 27], "hcupnet_cranial_long.csv"),
}


def load_hcup_joints(fname="hcupnet_joints_long.csv", drgs=(469, 470)):
    h = pd.read_csv(ROOT / "data/raw/hcupnet" / fname, dtype=str)
    h = h[h["Diagnoses/Procedures"].str.split(":").str[0].str.strip().astype(int).isin(drgs)]
    h = h[(h.Characteristicsgroup == "Expected Payer") & (h["Characteristic Levels"] == "Medicare")]
    h["value"] = pd.to_numeric(h["Avg. value_rd"].str.replace(",", ""), errors="coerce")
    h["suppressed"] = h.Suppresion.fillna("").str.contains(r"\*")
    h["year"] = pd.to_numeric(h["Year of Year"], errors="coerce").astype("Int64")
    h["st"] = h.State.map(ms.ABBR)
    return h.groupby(["st", "year"]).agg(hcup_medicare=("value", lambda v: v.sum(min_count=len(v))),
                                         any_suppressed=("suppressed", "any")).reset_index()


def cms_state_joints(DRGS=(469, 470)):
    geo = []
    for f in sorted((ROOT / "data/raw/cms_mupinp_geo").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        d["year"] = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        geo.append(d[d.Rndrng_Prvdr_Geo_Lvl == "State"])
    g = pd.concat(geo)
    g["drg"] = pd.to_numeric(g.DRG_Cd, errors="coerce"); g["n"] = pd.to_numeric(g.Tot_Dschrgs, errors="coerce")
    g["st"] = g.Rndrng_Prvdr_Geo_Desc.map(ms.ABBR)
    tot = g[g.drg.isin(DRGS)].groupby(["st", "year"]).n.sum().rename("cms_ffs_total")
    rows = []
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1", low_memory=False)
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        st, drg, n = hs.col(d, "State_Abrvtn"), hs.col(d, "DRG_Cd"), hs.col(d, "Tot_Dschrgs")
        d = d[[st, drg, n]].rename(columns={st: "st", drg: "drg", n: "n"})
        d["drg"] = pd.to_numeric(d.drg, errors="coerce"); d["n"] = pd.to_numeric(d.n, errors="coerce")
        rows.append(d[d.drg.isin(DRGS)].groupby("st").n.sum().rename("cms_ffs_published").reset_index().assign(year=y))
    pub = pd.concat(rows).set_index(["st", "year"]).cms_ffs_published
    return pd.concat([tot, pub], axis=1).fillna(0).reset_index()


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    panels, rows = [], []
    for name, (drgs, fname) in LINES.items():
        if not (ROOT / "data/raw/hcupnet" / fname).exists():
            print("skip (HCUPnet file not found):", name); continue
        panel = load_hcup_joints(fname, drgs).merge(cms_state_joints(drgs), on=["st", "year"], how="left").merge(ms.enrollment(), on=["st", "year"], how="left")
        panel = panel[~panel.any_suppressed & panel.hcup_medicare.notna()].assign(line=name)
        panels.append(panel)
        for y0, y1 in [(2016, 2023), (2019, 2023)]:
            have = panel.groupby("st").year.apply(lambda s_: set(range(y0, y1 + 1)) - {2015} <= set(s_))
            states = sorted(have[have].index)
            b = panel[panel.st.isin(states) & panel.year.isin([y0, y1])].groupby("year")[["hcup_medicare", "cms_ffs_total", "cms_ffs_published"]].sum()
            chg = lambda c: b.loc[y1, c] / b.loc[y0, c] - 1
            ps = panel[panel.st.isin(states) & panel.year.isin([y0, y1])].pivot_table(index="st", columns="year", values=["hcup_medicare", "cms_ffs_published"])
            st_h = ps["hcup_medicare"][y1] / ps["hcup_medicare"][y0] - 1
            st_p = ps["cms_ffs_published"][y1] / ps["cms_ffs_published"][y0] - 1
            rows.append({"category": name, "window": f"{y0}-{y1}", "n_states": len(states),
                         "hcup_medicare_change": chg("hcup_medicare"), "cms_total_change": chg("cms_ffs_total"),
                         "cms_published_change": chg("cms_ffs_published"),
                         "states_where_published_falls_more_than_hcup": int((st_p < st_h).sum()),
                         "median_state_gap_pts": float(((st_h - st_p) * 100).median())})
    allp = pd.concat(panels)
    allp[allp.line.str.startswith("joint")].to_csv(I / "multistate_joints_panel.csv", index=False)
    allp[allp.line.str.startswith("cranial (23")].to_csv(I / "multistate_cranial_panel.csv", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(I / "multistate_joints_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
