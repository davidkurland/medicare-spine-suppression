"""TEAM cervical simulation: one-way changes with the outpatient share held fixed, and the sources of the procedure gap.

The one-way rows in 82_uncertainty_sensitivity.py refit the outpatient share of the MS-DRG 473 pool after each change,
so the calibration absorbs part of every perturbation. This script repeats the payment rows with the share held at the
primary implied share and at the share counted in Part B claims, and reports how far the pooled mean then sits from the
anchored benchmark. It also splits the spending difference between inpatient posterior cervical fusion and inpatient
ACDF in MS-DRG 473 into its post-acute, professional fee, and readmission parts.
Outputs: data/interim/team_fixed_share.csv, data/interim/pcf_gap_sources.csv
"""
import importlib

import numpy as np
import pandas as pd

from common import ROOT

r82 = importlib.import_module("82_uncertainty_sensitivity")
t2, an = r82.t2, r82.an
I = ROOT / "data/interim"
PCF = (473, "inpatient posterior cervical")


def fixed(d, cms, share, label, family, **kw):
    """Gap with the outpatient share imposed; pooled mean compared with the anchored benchmark for that construction."""
    k, a473, sh_fit, ip = an.anchor(d, cms, an.PRIMARY)
    res, pool, st = r82.cerv_run(d, cms, share=share, **kw)
    return {"family": family, "spec": label, "share_held": share, "share_refit": sh_fit,
            "pcf_gap_risk_adjusted": res["pcf_gap_risk_adjusted"], "pcf_gap_unadjusted": res["pcf_gap_unadjusted"],
            "pooled_mean": st.loc[PCF, "benchmark"], "anchored_benchmark": a473,
            "pooled_mean_vs_anchored_benchmark": st.loc[PCF, "benchmark"] / a473 - 1}


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    cms = an.cms_prices()
    d = t2.cohort()
    base_pac = dict(t2.PAC_PAY)
    base_share = an.anchor(d, cms, an.PRIMARY)[2]
    counted = t2.op_share_473()
    rows = []
    for held, hl in [(base_share, "primary implied share"), (counted, "share counted in Part B claims")]:
        rows.append(fixed(d, cms, held, f"primary construction; {hl}", "Base"))
        for lab, pp in [("IRF payment halved", {**base_pac, "IRF": base_pac["IRF"] / 2}),
                        ("all institutional post-acute payment halved", {**base_pac, "IRF": base_pac["IRF"] / 2, "SNF": base_pac["SNF"] / 2, "LTCH": base_pac["LTCH"] / 2})]:
            rows.append(fixed(t2.cohort(pp), cms, held, f"{lab}; {hl}", "Post-acute payment", pac_pay=pp))
        rng = np.random.default_rng(5)
        dd = d.copy()
        move = dd.dest.isin(["IRF", "SNF", "LTCH"]) & (rng.random(len(dd)) < 0.5)
        dd.loc[move, "spend"] = dd.loc[move, "spend"] - dd.loc[move, "pac_pay"] + base_pac["HH"]
        dd.loc[move, "pac_pay"] = base_pac["HH"]
        rows.append(fixed(dd, cms, held, f"half of institutional discharges sent home with home health; {hl}", "Post-acute payment"))
        for lv in ("low", "high"):
            t2.ta.set_level(fee=lv)
            rows.append(fixed(t2.cohort(), cms, held, f"fee bundle assumptions {lv}; {hl}", "Professional fees"))
            t2.ta.set_level(fee="base")
    out = pd.DataFrame(rows)
    out.to_csv(I / "team_fixed_share.csv", index=False)
    print(out.round(4).to_string(index=False))

    # sources of the difference between inpatient posterior cervical fusion and inpatient ACDF in MS-DRG 473
    g = d[d.msdrg == 473]
    pcf, acdf = g[g.family == "cervical_posterior"], g[g.family == "cervical_anterior"]
    fees = t2.ta.summary().set_index("group").dollars
    src = {"n_pcf": len(pcf), "n_inpatient_acdf": len(acdf),
           "spend_pcf": pcf.spend.mean(), "spend_inpatient_acdf": acdf.spend.mean(),
           "difference": pcf.spend.mean() - acdf.spend.mean(),
           "postacute_part": pcf.pac_pay.mean() - acdf.pac_pay.mean(),
           "fee_part": fees["inpatient posterior cervical fusion"] - fees["inpatient ACDF"]}
    src["other_part"] = src["difference"] - src["postacute_part"] - src["fee_part"]
    # the procedure and pooling parts in dollars and on the common base of the pooled price
    dc = pd.read_csv(I / "pcf_decomposition.csv").set_index(["region", "drg"]).loc[(an.PRIMARY, 473)]
    src.update({"pooled_price": dc.pooled_price, "inpatient_only_price": dc.inpatient_only_price,
                "procedure_dollars": dc.pcf_spend - dc.inpatient_only_price, "pooling_dollars": dc.inpatient_only_price - dc.pooled_price,
                "procedure_points_of_pooled_price": (dc.pcf_spend - dc.inpatient_only_price) / dc.pooled_price,
                "pooling_points_of_pooled_price": (dc.inpatient_only_price - dc.pooled_price) / dc.pooled_price})
    s = pd.Series(src, name="value").rename_axis("quantity").reset_index()
    s.to_csv(I / "pcf_gap_sources.csv", index=False)
    print(s.round(4).to_string(index=False))
