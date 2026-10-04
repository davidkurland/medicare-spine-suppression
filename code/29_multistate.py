"""Multistate comparison: all-payer state data (HCUPnet SID, Medicare = FFS + MA) vs CMS public FFS files.

For each state and year:
  hcup_medicare       HCUPnet Medicare discharges (FFS + MA), by category
  cms_ffs_total       exact CMS FFS discharges (geography file, state totals of published state x DRG cells)
  cms_ffs_published   CMS provider-level published rows
  benes_total / benes_ffs from CMS enrollment (Part A)
Comparisons use matched categories:
  spine fusion:   CCSR MST013 (all-listed, 2016+)  vs  CMS fusion MS-DRG family (all fusion DRGs)
  DRG 460:        MS-DRG 460 in both sources (definition identical)
  cranial and joint replacement comparisons are built in 59_multistate_joints.py
Only state-years present and unsuppressed in HCUPnet are used; states with continuous 2016-2023 coverage form
the balanced panel. Outputs: data/interim/multistate_panel.csv, multistate_summary.csv
"""
import re

import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
ABBR = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
        "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}
FUSION = {453, 454, 455, 456, 457, 458, 459, 460, 471, 472, 473, 402, 426, 427, 428, 429, 430, 447, 448, 450, 451}


def load_hcup() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "data/raw/hcupnet/hcupnet_long.csv", dtype=str)
    h = h[(h.Characteristicsgroup == "Expected Payer") & (h["Characteristic Levels"] == "Medicare")]
    h["value"] = pd.to_numeric(h["Avg. value_rd"].str.replace(",", ""), errors="coerce")
    h["suppressed"] = h.Suppresion.fillna("").str.contains(r"\*")
    h["year"] = pd.to_numeric(h["Year of Year"], errors="coerce").astype("Int64")
    h["st"] = h.State.map(ABBR)
    h["code"] = h["Diagnoses/Procedures"].str.split(":").str[0].str.strip()
    return h[["st", "year", "classification", "code", "value", "suppressed"]]


def enrollment() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "data/raw/cms_enrollment/medicare_monthly_enrollment.csv", dtype=str,
                    usecols=["YEAR", "MONTH", "BENE_GEO_LVL", "BENE_STATE_ABRVTN", "TOT_BENES", "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"])
    e = e[(e.MONTH == "Year") & (e.BENE_GEO_LVL == "State")].copy()
    for c in ["TOT_BENES", "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e["year"] = e.YEAR.astype(int)
    e["st"] = e.BENE_STATE_ABRVTN
    e["ma_share"] = e.A_MA_AND_OTH_BENES / (e.A_MA_AND_OTH_BENES + e.A_ORGNL_MDCR_BENES)
    return e[["st", "year", "TOT_BENES", "A_ORGNL_MDCR_BENES", "ma_share"]]


def cms_state() -> pd.DataFrame:
    """State FFS totals by comparison category from the geography file and provider rows."""
    geo = []
    for f in sorted((ROOT / "data/raw/cms_mupinp_geo").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        d["year"] = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        geo.append(d[d.Rndrng_Prvdr_Geo_Lvl == "State"])
    g = pd.concat(geo)
    g["drg"] = pd.to_numeric(g.DRG_Cd, errors="coerce")
    g["n"] = pd.to_numeric(g.Tot_Dschrgs, errors="coerce")
    g["st"] = g.Rndrng_Prvdr_Geo_Desc.map(ABBR)
    prov = pd.read_parquet(I / "cms_neuro_drg_rows.parquet").rename(columns={"state": "st"})
    out = []
    for cat, drgs in {"fusion": FUSION, "drg460": {460}}.items():
        t = g[g.drg.isin(drgs)].groupby(["st", "year"]).n.sum().rename("cms_ffs_total")
        p = prov[prov.drg.isin(drgs)].groupby(["st", "year"]).n.sum().rename("cms_ffs_published")
        out.append(pd.concat([t, p], axis=1).fillna(0).reset_index().assign(category=cat))
    return pd.concat(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    h = load_hcup()
    hc = pd.concat([
        h[(h.classification == "CCSR") & (h.code == "MST013")].assign(category="fusion"),
        h[(h.classification == "MS-DRG") & (h.code == "460")].assign(category="drg460"),
    ])
    hc = hc.groupby(["st", "year", "category"]).agg(hcup_medicare=("value", lambda v: v.sum(min_count=len(v))),
                                                     any_suppressed=("suppressed", "any")).reset_index()
    panel = hc.merge(cms_state(), on=["st", "year", "category"], how="left").merge(enrollment(), on=["st", "year"], how="left")
    panel = panel[~panel.any_suppressed & panel.hcup_medicare.notna()]
    panel["hcup_per100k_benes"] = panel.hcup_medicare / panel.TOT_BENES * 1e5
    panel["cms_total_per100k_ffs"] = panel.cms_ffs_total / panel.A_ORGNL_MDCR_BENES * 1e5
    panel["cms_pub_per100k_ffs"] = panel.cms_ffs_published / panel.A_ORGNL_MDCR_BENES * 1e5
    panel.to_csv(I / "multistate_panel.csv", index=False)

    rows = []
    for cat, y0, y1 in [("fusion", 2016, 2023), ("fusion", 2019, 2023), ("drg460", 2013, 2023), ("drg460", 2016, 2023)]:
        d = panel[panel.category == cat]
        # 2015 (ICD-9-CM to ICD-10 transition year) is not required: HCUPnet returns far fewer rows for it
        have = d.groupby("st").year.apply(lambda s: set(range(y0, y1 + 1)) - {2015} <= set(s))
        states = sorted(have[have].index)
        b = d[d.st.isin(states) & d.year.isin([y0, y1])].groupby("year")[["hcup_medicare", "cms_ffs_total", "cms_ffs_published", "TOT_BENES", "A_ORGNL_MDCR_BENES"]].sum()
        if len(b) < 2:
            continue
        chg = lambda c: b.loc[y1, c] / b.loc[y0, c] - 1
        rate = lambda num, den: (b.loc[y1, num] / b.loc[y1, den]) / (b.loc[y0, num] / b.loc[y0, den]) - 1
        per_state = d[d.st.isin(states) & d.year.isin([y0, y1])].pivot_table(index="st", columns="year", values=["hcup_medicare", "cms_ffs_published"])
        st_hcup = per_state["hcup_medicare"][y1] / per_state["hcup_medicare"][y0] - 1
        st_pub = per_state["cms_ffs_published"][y1] / per_state["cms_ffs_published"][y0] - 1
        rows.append({"category": cat, "window": f"{y0}-{y1}", "n_states": len(states),
                     "hcup_medicare_change": chg("hcup_medicare"), "hcup_per_bene_change": rate("hcup_medicare", "TOT_BENES"),
                     "cms_total_change": chg("cms_ffs_total"), "cms_total_per_ffs_change": rate("cms_ffs_total", "A_ORGNL_MDCR_BENES"),
                     "cms_published_change": chg("cms_ffs_published"), "cms_pub_per_ffs_change": rate("cms_ffs_published", "A_ORGNL_MDCR_BENES"),
                     "states_where_published_falls_more_than_hcup": int((st_pub < st_hcup).sum()),
                     "median_state_gap_pts": float(((st_hcup - st_pub) * 100).median())})
    summ = pd.DataFrame(rows)
    summ.to_csv(I / "multistate_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
