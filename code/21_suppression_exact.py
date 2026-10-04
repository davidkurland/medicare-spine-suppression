"""Exact volume hidden by provider-level suppression, from CMS 'Medicare Inpatient Hospitals by Geography and Service'.

The geography file reports national and state totals by MS-DRG (state cells <11 are themselves suppressed,
national cells essentially never). Hidden provider-level volume = geography total - sum of published
provider rows, by year x MS-DRG, aggregated to neurosurgical families.
Outputs: data/interim/suppression_exact_national.csv, suppression_exact_state.csv
"""
import re

import numpy as np
import pandas as pd

from common import DRG_TO_CAT, MSDRG_GROUP, ROOT

I = ROOT / "data/interim"
FAMILY = {  # finer spine families
    **{d: "cervical_fusion" for d in (471, 472, 473, 429, 430)},
    **{d: "deformity_fusion" for d in (456, 457, 458)},
    **{d: "thoracolumbar_fusion" for d in (453, 454, 455, 459, 460, 402, 426, 427, 428, 447, 448, 450, 451)},
    **{d: "spine_nonfusion" for d in (28, 29, 30, 490, 491, 518, 519, 520)},
    **{d: "cranial_thrombectomy_group" for d in (23, 24)},
    **{d: "cranial_open_other" for d in (20, 21, 22, 25, 26, 27)},
    **{d: "cranial_shunt" for d in (31, 32, 33)},
}


def load_geo() -> pd.DataFrame:
    frames = []
    for f in sorted((ROOT / "data/raw/cms_mupinp_geo").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        d["year"] = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    g["drg"] = pd.to_numeric(g.DRG_Cd, errors="coerce")
    g["n"] = pd.to_numeric(g.Tot_Dschrgs, errors="coerce")
    g["level"] = g.Rndrng_Prvdr_Geo_Lvl
    g["state"] = np.where(g.level == "National", "US", g.Rndrng_Prvdr_Geo_Desc)
    return g


if __name__ == "__main__":
    geo = load_geo()
    geo = geo[geo.drg.isin(DRG_TO_CAT)]
    geo["category"] = geo.drg.map(DRG_TO_CAT)
    geo["group"] = geo.category.map(MSDRG_GROUP)
    geo["family"] = geo.drg.map(FAMILY)

    rows = pd.read_parquet(I / "cms_neuro_drg_rows.parquet")
    rows["group"] = rows.category.map(MSDRG_GROUP)
    rows["family"] = rows.drg.map(FAMILY)

    # state abbreviations in provider rows vs state names in geography file
    abbr = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
            "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
            "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
            "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
            "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
            "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
            "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
            "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
            "Puerto Rico": "PR"}
    geo["st"] = geo.state.map(lambda s: "US" if s == "US" else abbr.get(s, s))

    out = []
    for key in ["group", "family"]:
        nat_total = geo[geo.st == "US"].groupby(["year", key]).n.sum()
        nat_pub = rows.groupby(["year", key]).n.sum()
        d = pd.concat([nat_total.rename("total_ffs"), nat_pub.rename("published")], axis=1).fillna(0).reset_index()
        d = d.rename(columns={key: "stratum"})
        d["level"] = key
        out.append(d)
    nat = pd.concat(out)
    nat["hidden"] = nat.total_ffs - nat.published
    nat["hidden_share"] = nat.hidden / nat.total_ffs
    nat.to_csv(I / "suppression_exact_national.csv", index=False)

    st_total = geo[geo.st != "US"].groupby(["year", "st", "group"]).n.sum().rename("state_total_published_cells")
    st_pub = rows.groupby(["year", "state", "group"]).n.sum().rename("provider_published")
    st_pub.index = st_pub.index.set_names(["year", "st", "group"])
    st = pd.concat([st_total, st_pub], axis=1).fillna(0).reset_index()
    st["hidden_min"] = st.state_total_published_cells - st.provider_published   # state DRG cells <11 also suppressed
    st.to_csv(I / "suppression_exact_state.csv", index=False)

    pd.set_option("display.width", 220)
    g = nat[nat.level == "group"].pivot(index="year", columns="stratum", values=["total_ffs", "published", "hidden_share"])
    print(g.round(3).to_string())
    f = nat[nat.level == "family"].pivot(index="year", columns="stratum", values="total_ffs")
    print(f.to_string())
    for grp in ["spine", "cranial"]:
        x = nat[(nat.level == "group") & (nat.stratum == grp)].set_index("year")
        for c in ["total_ffs", "published"]:
            print(f"{grp} {c}: 2013 {x.loc[2013, c]:,.0f} -> 2024 {x.loc[2024, c]:,.0f} ({x.loc[2024, c] / x.loc[2013, c] - 1:+.1%}); 2013->2019 {x.loc[2019, c] / x.loc[2013, c] - 1:+.1%}")
