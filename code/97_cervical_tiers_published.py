"""Cervical fusion by severity tier in complete and published CMS counts, 2013-2024.

Complete national discharges come from the by Geography and Service totals (national_drg_totals.csv). Published
discharges are the sum of hospital rows in the by Provider and Service files. MS-DRG 473 is cervical fusion without
complication or comorbidity; MS-DRGs 471 and 472 are cervical fusion with a major complication or comorbidity and with a
complication or comorbidity. The table shows which part of the inpatient cervical pool left the public file: the
uncomplicated cells shrank and dropped below the 11-discharge threshold, while the cells with complications stayed
large enough to publish.
Output: data/interim/cervical_tiers_published.csv
"""
import glob
import re

import pandas as pd

from common import ROOT

I = ROOT / "data/interim"

if __name__ == "__main__":
    rows = []
    files = sorted(glob.glob(str(ROOT / "data/raw/cms_mupinp/MUP_I*_PrvSvc.CSV")) + glob.glob(str(ROOT / "data/raw/cms_mupinp/MUP_I*_PRVSVC.CSV")))
    for f in files:
        y = 2000 + int(re.search(r"DY(\d\d)", f).group(1))
        d = pd.read_csv(f, dtype=str, encoding="latin-1", usecols=["DRG_Cd", "Tot_Dschrgs"])
        d["drg"] = pd.to_numeric(d.DRG_Cd, errors="coerce")
        d["n"] = pd.to_numeric(d.Tot_Dschrgs, errors="coerce")
        for k, v in d[d.drg.isin([471, 472, 473])].groupby("drg").n.sum().items():
            rows.append((y, int(k), v))
    pub = pd.DataFrame(rows, columns=["year", "drg", "published"]).pivot(index="year", columns="drg", values="published")
    tot = pd.read_csv(I / "national_drg_totals.csv")
    com = tot[tot.drg.isin([471, 472, 473])].pivot_table(index="year", columns="drg", values="n", aggfunc="sum")
    out = pd.DataFrame({"complete_cc_mcc_471_472": com[471] + com[472], "published_cc_mcc_471_472": pub[471] + pub[472],
                        "complete_none_473": com[473], "published_none_473": pub[473]})
    out["hidden_share_cc_mcc"] = 1 - out.published_cc_mcc_471_472 / out.complete_cc_mcc_471_472
    out["hidden_share_none"] = 1 - out.published_none_473 / out.complete_none_473
    out["complete_share_none"] = out.complete_none_473 / (out.complete_none_473 + out.complete_cc_mcc_471_472)
    out = out.reset_index()
    out.to_csv(I / "cervical_tiers_published.csv", index=False)
    print(out.round(3).to_string(index=False))
