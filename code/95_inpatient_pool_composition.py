"""Composition of the inpatient cervical fusion pool as anterior cervical discectomy and fusion moved outpatient.

Part A (national): primary-surgeon services in the Physician/Supplier Procedure Summary, 2013-2024. Inpatient posterior
cervical fusion (CPT 22600) per 100 inpatient ACDF (CPT 22551), and the same ratio across all settings.

Part B (Texas): Medicare FFS inpatients aged 65 years or older in MS-DRG 473 (the TEAM simulation cohort), by year,
2016-2019: share with a posterior cervical fusion, share discharged to a rehabilitation, skilled nursing, or long-term
care facility, share with 2 or more V28 HCC categories, share aged 75 years or older, and diagnoses coded per record.
The same measures among inpatient ACDF in MS-DRG 473 and, as a comparison with no outpatient migration, inpatient ACDF
in MS-DRG 472. Differences between 2016 and 2019 carry 95% CIs from resampling hospitals. Aggregates only; any group
with 10 or fewer records in a year is left out.
Outputs: data/interim/psps_pcf_per_100_acdf.csv, data/interim/texas/tx_cervical_pool_by_year.csv,
         data/interim/texas/tx_cervical_pool_change.csv
"""
import importlib

import numpy as np
import pandas as pd

from common import ROOT

t2 = importlib.import_module("46_team_cervical_pricing")
I = ROOT / "data/interim"
MEASURES = {"posterior": "posterior cervical fusion, share", "facility": "discharged to a post-acute facility, share",
            "hcc2": "2 or more HCC categories, share", "age75": "aged 75 years or older, share",
            "n_dx": "diagnoses coded per record, mean"}


def national():
    x = pd.read_csv(I / "setting_migration_cpt.csv", dtype={"cpt": str})
    ip = x[x.setting == "inpatient"].pivot_table(index="year", columns="cpt", values="services", aggfunc="sum")
    al = x.pivot_table(index="year", columns="cpt", values="services", aggfunc="sum")
    t = pd.DataFrame({"inpatient_acdf_22551": ip["22551"], "inpatient_pcf_22600": ip["22600"],
                      "all_settings_acdf_22551": al["22551"], "all_settings_pcf_22600": al["22600"]})
    t["pcf_per_100_inpatient_acdf"] = 100 * t.inpatient_pcf_22600 / t.inpatient_acdf_22551
    t["pcf_per_100_acdf_all_settings"] = 100 * t.all_settings_pcf_22600 / t.all_settings_acdf_22551
    return t.reset_index()


def texas():
    d = t2.cohort()
    d["posterior"] = (d.family == "cervical_posterior").astype(float)
    d["acdf"] = (d.family == "cervical_anterior").astype(float)
    d["facility"] = d.dest.isin(["IRF", "SNF", "LTCH"]).astype(float)
    d["hcc2"] = (d.hcc_count >= 2).astype(float)
    d["age75"] = (d.age_grp != "65-74").astype(float)
    groups = {"MS-DRG 473, all inpatients": d[d.msdrg == 473],
              "MS-DRG 473, inpatient ACDF": d[(d.msdrg == 473) & (d.acdf == 1)],
              "MS-DRG 473, inpatient posterior cervical fusion": d[(d.msdrg == 473) & (d.posterior == 1)],
              "MS-DRG 472, inpatient ACDF": d[(d.msdrg == 472) & (d.acdf == 1)]}
    by_year, change = [], []
    rng = np.random.default_rng(7)
    for label, g in groups.items():
        cols = [c for c in MEASURES if not (c == "posterior" and "all inpatients" not in label)]
        for yr, s in g.groupby("year"):
            if len(s) <= 10:
                continue
            by_year.append({"group": label, "year": int(yr), "n": len(s), **{c: s[c].mean() for c in cols}})
        a = g[g.year.isin([2016, 2019])].assign(one=1.0)
        n = a.pivot_table(index="hosp", columns="year", values="one", aggfunc="sum").reindex(columns=[2016, 2019]).fillna(0).values
        idx = rng.integers(0, len(n), size=(4000, len(n)))
        for c in cols:
            s = a.pivot_table(index="hosp", columns="year", values=c, aggfunc="sum").reindex(columns=[2016, 2019]).fillna(0).values
            r = s[idx].sum(1) / n[idx].sum(1)
            lo, hi = np.percentile(r[:, 1] - r[:, 0], [2.5, 97.5])
            v16, v19 = s[:, 0].sum() / n[:, 0].sum(), s[:, 1].sum() / n[:, 1].sum()
            change.append({"group": label, "measure": c, "label": MEASURES[c], "value_2016": v16, "value_2019": v19,
                           "difference": v19 - v16, "lo": lo, "hi": hi, "hospitals": len(n)})
    return pd.DataFrame(by_year), pd.DataFrame(change)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    nat = national()
    nat.to_csv(I / "psps_pcf_per_100_acdf.csv", index=False)
    print(nat.round(1).to_string(index=False))
    by, ch = texas()
    by.to_csv(I / "texas/tx_cervical_pool_by_year.csv", index=False)
    ch.to_csv(I / "texas/tx_cervical_pool_change.csv", index=False)
    print(by.round(4).to_string(index=False))
    print(ch.round(4).to_string(index=False))
