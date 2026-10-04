"""Case-mix shift in inpatient-only data: as uncomplicated spine cases move outpatient, the inpatient pool gets sicker.

National complete Medicare FFS discharges (CMS by Geography and Service, national rows) by MS-DRG severity tier:
  cervical fusion 471 (MCC) / 472 (CC) / 473 (no CC/MCC)
  spinal fusion except cervical, combined anterior/posterior and curvature 453-458 (MCC / CC / none) and
  spinal fusion except cervical 459 (MCC) / 460 (no MCC)  -> MCC share across 453-460
  back and neck procedures except fusion 518 (MCC or disc device) / 519 (CC) / 520 (no CC/MCC)
Coding-intensity control: craniotomy and endovascular intracranial procedures 025 (MCC) / 026 (CC) / 027 (none),
which did not migrate outpatient. Difference-in-differences in the lowest-tier share (spine family minus cranial control)
separates case-mix migration from secular growth in CC/MCC documentation.
FY2025 (2024 data) fusion DRGs 402, 426-430, 447-448, 450-451 restructured thoracolumbar tiers, so 2024 is excluded
from the 453-460 series.
Outputs: data/interim/casemix_tiers_national.csv, casemix_shift_summary.csv
"""
import glob
import re

import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
FAMILIES = {
    "cervical_fusion": {"mcc": [471], "cc": [472], "none": [473]},
    "thoracolumbar_fusion_453_460": {"mcc": [453, 456, 459], "cc": [454, 457], "none": [455, 458, 460]},
    "back_neck_nonfusion_518_520": {"mcc": [518], "cc": [519], "none": [520]},
    "cranial_control_025_027": {"mcc": [25], "cc": [26], "none": [27]},
    # comparator line; no CC tier exists, so "none" here means without MCC. FY2024 moved hip replacement for fracture
    # out of 469/470 into 521/522, which are folded back in so the series is continuous.
    "joint_replacement_469_470": {"mcc": [469, 521], "none": [470, 522]},
}

if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/cms_mupinp_geo/*.CSV"))):
        y = 2000 + int(re.search(r"DY(\d\d)", f).group(1))
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        d = d[d.Rndrng_Prvdr_Geo_Lvl == "National"]
        d["drg"] = pd.to_numeric(d.DRG_Cd, errors="coerce")
        d["n"] = pd.to_numeric(d.Tot_Dschrgs, errors="coerce")
        pay = pd.to_numeric(d.get("Avg_Mdcr_Pymt_Amt"), errors="coerce") if "Avg_Mdcr_Pymt_Amt" in d else None
        d["pay_total"] = d.n * pay if pay is not None else float("nan")
        for fam, tiers in FAMILIES.items():
            for tier, drgs in tiers.items():
                s = d[d.drg.isin(drgs)]
                rows.append({"year": y, "family": fam, "tier": tier, "discharges": s.n.sum(), "medicare_payment_total": s.pay_total.sum()})
    t = pd.DataFrame(rows)
    t = t[~((t.family == "thoracolumbar_fusion_453_460") & (t.year == 2024))]
    t = t[~((t.family == "back_neck_nonfusion_518_520") & (t.year < 2015))]   # DRGs 518-520 created in FY2015 (2014 partial)
    t["share"] = t.discharges / t.groupby(["year", "family"]).discharges.transform("sum")
    t.to_csv(I / "casemix_tiers_national.csv", index=False)

    w = t.pivot_table(index="year", columns=["family", "tier"], values="share")
    pd.set_option("display.width", 250)
    print((w * 100).round(1).to_string())

    out = []
    ctrl = w[("cranial_control_025_027", "none")]
    for fam in [f for f in FAMILIES if f != "cranial_control_025_027"]:
        s = w[(fam, "none")].dropna()
        y0, y1 = s.index.min(), s.index.max()
        for a, b in [(y0, y1), (y0, 2019), (2019, min(y1, 2023))]:
            out.append({"family": fam, "window": f"{a}-{b}",
                        "none_share_start": s.loc[a], "none_share_end": s.loc[b],
                        "mcc_share_start": w.loc[a, (fam, "mcc")], "mcc_share_end": w.loc[b, (fam, "mcc")],
                        "cranial_none_share_start": ctrl.loc[a], "cranial_none_share_end": ctrl.loc[b],
                        "did_none_share_pts": ((s.loc[b] - s.loc[a]) - (ctrl.loc[b] - ctrl.loc[a])) * 100,
                        "did_mcc_share_pts": ((w.loc[b, (fam, "mcc")] - w.loc[a, (fam, "mcc")]) - (w.loc[b, ("cranial_control_025_027", "mcc")] - w.loc[a, ("cranial_control_025_027", "mcc")])) * 100,
                        "note": "tiers not comparable across FY2018 regrouping; use MCC share" if fam.startswith("thoracolumbar") else ""})
    summ = pd.DataFrame(out)
    summ.to_csv(I / "casemix_shift_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
