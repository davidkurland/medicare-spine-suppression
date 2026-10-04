"""Sensitivity of the anchored TEAM cervical result to the spending construction, with and without professional fees
and readmissions.

The primary simulation adds professional fees and readmissions to every episode (team_addons.py).
This script documents what that changes, by switching the additions off and on, and tests the anchoring:
  - the capture factor on the 2 cervical lines with no outpatient cases (how much of CMS's benchmark the measured construction reproduces)
  - the outpatient share implied by CMS's published price, against the share counted from Part B claims
  - the capture factor that would make the counted share fit CMS's price, against the factor the 2 lines give
  - how far the result moves when the capture factor is perturbed, or taken from 1 line, or set to 1
Outputs: data/interim/team_rebuilt.csv, team_rebuilt_fragility.csv
"""
import importlib
import warnings

import numpy as np
import pandas as pd

from common import ROOT

warnings.filterwarnings("ignore")
t2 = importlib.import_module("46_team_cervical_pricing")
an = importlib.import_module("51_team_cms_anchor")
r82 = importlib.import_module("82_uncertainty_sensitivity")
ta = t2.ta
I = ROOT / "data/interim"
PCF = (473, "inpatient posterior cervical")


def run(cms, region=an.PRIMARY, k_rule="mean", k_mult=1.0, share=None, seed=7):
    d = t2.cohort()                                   # spending under the team_addons switches in effect
    ip = {t: d[d.msdrg == t].spend.mean() for t in (471, 472, 473)}
    kt = {t: ip[t] / cms.loc[(region, t), "benchmark"] for t in (471, 472)}
    k = {"mean": np.mean(list(kt.values())), "472": kt[472], "one": 1.0}[k_rule] * k_mult
    op_mean = t2.OPPS_ACDF + t2.PAC_OP_HH_RATE * t2.PAC_PAY["HH"] + ta.cervical_outpatient()
    implied = float(np.clip((ip[473] - k * cms.loc[(region, 473), "benchmark"]) / (ip[473] - op_mean), 0.02, 0.95))
    counted = t2.op_share_473()                       # hospital outpatient ACDF services / (those + complete MS-DRG 473 discharges), 2022-2024
    k_fit = (counted * op_mean + (1 - counted) * ip[473]) / cms.loc[(region, 473), "benchmark"]
    used = implied if share is None else (counted if share == "counted" else share)
    pool = t2.risk_predict(r82.pooled(d, used, seed))
    st = t2.static(pool).set_index(["drg", "group"])
    ar, _, _ = t2.archetypes(pool); ar = ar.set_index("archetype")
    s = st.loc[PCF]; ip_mean = st.loc[(473, "inpatient, all"), "mean_spend"]
    return {"capture_471": kt[471], "capture_472": kt[472], "capture_used": k, "capture_that_fits_counted_share": k_fit,
            "implied_outpatient_share": implied, "counted_outpatient_share": counted, "share_used": used,
            "pooled_price": s.benchmark, "outpatient_spend": st.loc[(473, "outpatient ACDF"), "mean_spend"],
            "inpatient_acdf_spend": st.loc[(473, "inpatient ACDF"), "mean_spend"], "pcf_spend": s.mean_spend,
            "pcf_gap_unadjusted": s.gap_unadjusted, "pcf_gap_risk_adjusted": s.gap_risk_adjusted,
            "pcf_gap_vs_cms_dollar_target": (s.mean_spend - s.risk_adjusted_target) / (s.risk_adjusted_target / k),
            "procedure_part": s.mean_spend / ip_mean - 1, "pooling_points": s.gap_unadjusted - (s.mean_spend / ip_mean - 1),
            "pcf472_gap_risk_adjusted": st.loc[(472, "inpatient posterior cervical"), "gap_risk_adjusted"],
            "complex_center_gap": ar.loc["complex spine center (posterior-heavy)", "gap_vs_target"],
            "outpatient_first_gap": ar.loc["outpatient-first ACDF hospital", "gap_vs_target"]}


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    cms = an.cms_prices()
    rows = []

    def add(label, fee="base", readmit="base", enabled=True, **kw):
        ta.set_level(fee=fee, readmit=readmit, enabled=enabled)
        rows.append({"scenario": label, **run(cms, **kw)})
        ta.set_level(fee="base", readmit="base", enabled=True)

    add("without professional fees or readmissions", enabled=False)
    add("fees only", readmit="off")
    add("PRIMARY: fees and readmissions")
    add("fee assumptions low", fee="low")
    add("fee assumptions high", fee="high")
    add("readmission inputs low", readmit="low")
    add("readmission inputs high", readmit="high")
    add("capture factor from MS-DRG 472 only", k_rule="472")
    add("capture factor = 1 (nothing left unmeasured)", k_rule="one")
    add("outpatient share as counted from claims", share="counted")
    add("Middle Atlantic", region="CENS_DIV_2")
    add("without professional fees or readmissions; outpatient share as counted", enabled=False, share="counted")
    out = pd.DataFrame(rows)
    out.to_csv(I / "team_rebuilt.csv", index=False)
    show = ["scenario", "capture_471", "capture_472", "capture_that_fits_counted_share", "implied_outpatient_share", "share_used", "pcf_gap_unadjusted",
            "pcf_gap_risk_adjusted", "pcf_gap_vs_cms_dollar_target", "procedure_part", "pooling_points", "complex_center_gap"]
    print("CONSTRUCTIONS COMPARED, West South Central unless stated\n", out[show].round(3).to_string(index=False))
    print("\nspending, primary:", out.loc[2, ["pooled_price", "outpatient_spend", "inpatient_acdf_spend", "pcf_spend"]].astype(float).round(0).to_dict())

    fr = []
    for label, kw in [("without professional fees or readmissions", dict(enabled=False)), ("primary", {})]:
        for km in (0.9, 1.0, 1.1):
            ta.set_level(enabled=kw.get("enabled", True))
            r = run(cms, k_mult=km)
            fr.append({"construction": label, "capture_multiplier": km, "capture_used": r["capture_used"], "implied_outpatient_share": r["implied_outpatient_share"],
                       "pcf_gap_risk_adjusted": r["pcf_gap_risk_adjusted"], "pcf_gap_vs_cms_dollar_target": r["pcf_gap_vs_cms_dollar_target"]})
    ta.set_level(enabled=True)
    fr = pd.DataFrame(fr); fr.to_csv(I / "team_rebuilt_fragility.csv", index=False)
    print("\nFRAGILITY: capture factor moved 10% either way\n", fr.round(3).to_string(index=False))
