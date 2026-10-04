"""Hospital volume misclassification in Indiana, 2017-2024 (replication of the New York analysis).

All-payer spine discharges by hospital from Indiana Department of Health hospital x MS-DRG x payer tables
(PATS summed over payers, spine MS-DRG group from common.py), linked to CCNs with the crosswalk rows flagged match_ok
(data/interim/state_compare/IN_crosswalk.csv, includes manual overrides). CMS published FFS spine discharges by
CCN from cms_neuro_hospital_year.parquet. IDOH hospitals without a CCN link are excluded; their share of all-payer
spine discharges is reported. CCNs with published CMS rows but no IDOH hospital (e.g., Pinnacle Hospital) are not
all-payer units and are excluded.
Outputs: data/interim/volume_misclassification_in.csv, in_hospital_spine_volume.csv
"""
import importlib

import pandas as pd

from common import DRG_TO_CAT, MSDRG_GROUP, ROOT

mc = importlib.import_module("40_volume_misclassification")
I = ROOT / "data/interim"
RAW = ROOT / "data/raw/states/IN"

if __name__ == "__main__":
    parts = []
    for f in sorted(RAW.glob("*_MSDRG_IN*.csv")):
        y = int(f.name[:4])
        d = pd.read_csv(f)
        d.columns = [c.upper() for c in d.columns]
        d["group"] = pd.to_numeric(d.MSDRG, errors="coerce").map(DRG_TO_CAT).map(MSDRG_GROUP)
        s = d[d.group == "spine"].groupby("HOSPITAL_ID").PATS.sum().rename("all_payer").reset_index()
        parts.append(s.assign(year=y))
    h = pd.concat(parts).rename(columns={"HOSPITAL_ID": "id"})
    xw = pd.read_csv(I / "state_compare/IN_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok == True][["id", "ccn"]]
    m = h.merge(xw, on="id", how="left")
    unlinked = m[m.ccn.isna()].groupby("year").all_payer.sum() / m.groupby("year").all_payer.sum()
    print("unlinked all-payer spine share by year:", unlinked.round(3).to_dict())
    m = m.dropna(subset=["ccn"]).groupby(["ccn", "year"]).all_payer.sum().reset_index()
    cms = pd.read_parquet(I / "cms_neuro_hospital_year.parquet")
    cms = cms[(cms.state == "IN") & (cms.group == "spine")].groupby(["ccn", "year"]).discharges.sum().rename("cms_ffs").reset_index()
    m = m.merge(cms, on=["ccn", "year"], how="left").fillna({"cms_ffs": 0})
    m = m[m.all_payer > 0]
    m.to_csv(I / "in_hospital_spine_volume.csv", index=False)
    out = mc.misclassification(m)
    out["unlinked_allpayer_share"] = out.year.map(unlinked)
    out.to_csv(I / "volume_misclassification_in.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.rule.isin([">=50/yr", ">=100/yr"])].round(3).to_string(index=False))
