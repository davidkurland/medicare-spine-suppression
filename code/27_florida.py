"""Florida AHCA FloridaHealthFinder query tool (free, no terms): Medicare FFS vs Medicare Managed Care for
neurosurgical inpatients (MS-DRG) and outpatient spine CPTs by facility type, 2023-2025.

The public tool posts (src, agg, json filters) to Handler.ashx and receives CSV. downloadtest.floridahealthfinder.gov
is the endpoint that the public FloridaHealthFinder query tool itself calls. Principal payer separates
'Medicare' (A) from 'Medicare Managed Care' (B). MS-DRG filters use AHCA's DRG description strings.
Outputs: data/raw/states/FL/*.csv (raw query results), data/interim/florida/*.csv
"""
import io
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import ROOT

URL = "https://downloadtest.floridahealthfinder.gov/Handler.ashx"
RAW = ROOT / "data/raw/states/FL"
OUT = ROOT / "data/interim/florida"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
CHOICES = RAW / "fl_reportChoices.json"   # JSON saved from the query tool's report-choice list (its Inpatient and Outpatient filter fields and values)

SPINE_KEYS = ["spinal fusion", "back and neck", "spinal procedures", "combined anterior and posterior"]
CRANIAL_KEYS = ["craniotomy", "intracranial vascular procedures", "ventricular shunt"]
OUTPATIENT_CPT = {"22551": "ACDF", "22600": "Posterior cervical fusion", "63030": "Lumbar discectomy",
                  "63047": "Lumbar laminectomy", "22630": "Lumbar interbody fusion", "22633": "Lumbar interbody fusion",
                  "22612": "Lumbar posterolateral fusion", "22856": "Cervical disc arthroplasty"}


def query(src, agg, filters):
    body = f"src={src}&agg={urllib.parse.quote(agg)}&json={urllib.parse.quote(json.dumps(filters))}"
    req = urllib.request.Request(URL, data=body.encode(), headers={"Content-Type": "application/x-www-form-urlencoded",
                                                                   "User-Agent": "Mozilla/5.0 (research; aggregate query)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        txt = r.read().decode("utf-8", errors="replace")
    time.sleep(1.0)
    df = pd.read_csv(io.StringIO(txt))
    return df[~df.iloc[:, 0].astype(str).str.lower().eq("total")]


if __name__ == "__main__":
    rc = json.load(open(CHOICES))
    ip_fields = {f["fieldName"]: f for f in rc["Inpatient"]}
    op_fields = {f["fieldName"]: f for f in rc["Outpatient"]}
    drgs = ip_fields["MS DRG Description"]["values"]
    groups = {"spine": [d for d in drgs if any(k in d.lower() for k in SPINE_KEYS)],
              "cranial": [d for d in drgs if any(k in d.lower() for k in CRANIAL_KEYS)]}
    years = ip_fields["Year"]["values"]
    print({g: len(v) for g, v in groups.items()}, "years", years)
    json.dump(groups, open(OUT / "drg_groups_used.json", "w"), indent=1)

    rows = []
    for y in years:
        for g, vals in groups.items():
            # statewide by payer
            df = query("Inpatient", "Principal Payer", [{"field": "Year", "values": [y]}, {"field": "MS DRG Description", "values": vals}])
            df.to_csv(RAW / f"ip_{g}_{y}_by_payer.csv", index=False)
            for _, r in df.iterrows():
                rows.append({"year": int(y), "group": g, "payer": r["Principal Payer"], "discharges": r["Discharges"]})
            # by hospital, separately for Medicare FFS and Medicare Managed Care
            for payer in ["Medicare", "Medicare Managed Care"]:
                h = query("Inpatient", "Hospital Name", [{"field": "Year", "values": [y]}, {"field": "MS DRG Description", "values": vals},
                                                         {"field": "Principal Payer", "values": [payer]}])
                h.assign(year=int(y), group=g, payer=payer).to_csv(RAW / f"ip_{g}_{y}_{payer.replace(' ', '_')}_by_hospital.csv", index=False)
            print("inpatient", y, g, flush=True)
    state = pd.DataFrame(rows)
    state.to_csv(OUT / "fl_inpatient_by_payer.csv", index=False)
    wide = state.pivot_table(index=["group", "year"], columns="payer", values="discharges", aggfunc="sum").fillna(0)
    wide["medicare_total"] = wide["Medicare"] + wide["Medicare Managed Care"]
    wide["ma_share"] = wide["Medicare Managed Care"] / wide["medicare_total"]
    print(wide[["Medicare", "Medicare Managed Care", "medicare_total", "ma_share"]].round(3).to_string())

    # outpatient CPTs by facility type x payer
    cpt_field = next(k for k in op_fields if "CPT" in k)
    orows = []
    for y in op_fields[next(k for k in op_fields if k.endswith("Year"))]["values"]:
        for cpt, fam in OUTPATIENT_CPT.items():
            if cpt not in op_fields[cpt_field]["values"]:
                continue
            for payer in ["Medicare", "Medicare Managed Care"]:
                yfield = next(k for k in op_fields if k.endswith("Year"))
                df = query("Outpatient", "Facility Type", [{"field": yfield, "values": [y]}, {"field": cpt_field, "values": [cpt]},
                                                           {"field": "Principal Payer", "values": [payer]}])
                for _, r in df.iterrows():
                    orows.append({"year": int(y), "cpt": cpt, "family": fam, "payer": payer, "facility_type": r.iloc[0],
                                  "visits": r.get("Visits", r.get("Discharges", r.iloc[1]))})
        print("outpatient", y, flush=True)
    op = pd.DataFrame(orows)
    op.to_csv(OUT / "fl_outpatient_spine_cpt.csv", index=False)
    print(op.pivot_table(index=["family", "year"], columns=["payer", "facility_type"], values="visits", aggfunc="sum").fillna(0).to_string())
