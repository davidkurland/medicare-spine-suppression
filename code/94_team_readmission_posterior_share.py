"""TEAM cervical simulation: a higher readmission rate for posterior cervical fusion, and the hospital result by posterior share.

The primary construction gives inpatient posterior cervical fusion the inpatient ACDF readmission rate, because no
source gives a Medicare rate for the posterior operation. Part A raises the posterior rate to 1.5, 2, and 2.5 times
the inpatient ACDF rate (payment per readmission unchanged) and re-runs the anchored simulation, with the outpatient
share refit and with it held at the primary implied share.

Part B takes the primary pool and computes spending vs target, with the 2% discount, for a hospital whose inpatient
cervical episodes are a stated share posterior fusion and the rest inpatient ACDF: within MS-DRG 473 alone, and across
MS-DRGs 471 to 473 at the cohort's inpatient MS-DRG mix. Group means and risk-adjusted targets are those of the
primary run, so the rows are arithmetic on eTable values.
Outputs: data/interim/team_pcf_readmission.csv, data/interim/team_posterior_share_gradient.csv
"""
import importlib

import pandas as pd

from common import ROOT

r82 = importlib.import_module("82_uncertainty_sensitivity")
t2, an = r82.t2, r82.an
I = ROOT / "data/interim"
PCF = (473, "inpatient posterior cervical")
IP = "inpatient"


def readmit_rows(d, cms):
    base_share = an.anchor(d, cms, an.PRIMARY)[2]
    ip_readmit = t2.ta.readmission(t2.ta.IP)            # dollars per inpatient episode in the primary construction
    rows = []
    for mult in (1.0, 1.5, 2.0, 2.5):
        dd = d.copy()
        post = dd.family == "cervical_posterior"
        dd.loc[post, "spend"] = dd.loc[post, "spend"] + (mult - 1.0) * ip_readmit
        for label, share in (("outpatient share refit", None), ("outpatient share held at primary value", base_share)):
            res, pool, st = r82.cerv_run(dd, cms, share=share)
            rows.append({"posterior_rate_multiple": mult, "share_rule": label,
                         "added_dollars_per_posterior_episode": (mult - 1.0) * ip_readmit,
                         "capture_factor_k": res["capture_factor_k"], "outpatient_share": res["implied_outpatient_share"],
                         "pcf_gap_unadjusted": res["pcf_gap_unadjusted"], "pcf_gap_risk_adjusted": res["pcf_gap_risk_adjusted"],
                         "pcf472_gap_risk_adjusted": res["pcf472_gap_risk_adjusted"],
                         "ip_acdf_gap_risk_adjusted": res["ip_acdf_gap_risk_adjusted"],
                         "complex_center_gap": res["complex_center_gap"]})
    return pd.DataFrame(rows)


def gradient_rows(d, cms):
    res, pool, st = r82.cerv_run(d, cms)
    bench = {drg: pool[pool.msdrg == drg].spend.mean() for drg in (471, 472, 473)}
    predm = {drg: pool[pool.msdrg == drg].pred.mean() for drg in (471, 472, 473)}
    ip = pool[(pool.setting == IP) & pool.family.isin(["cervical_anterior", "cervical_posterior"])]
    drg_mix = ip.msdrg.value_counts(normalize=True).to_dict()      # cohort's inpatient MS-DRG mix (ACDF plus posterior)
    obs = d[(d.msdrg == 473) & d.family.isin(["cervical_anterior", "cervical_posterior"])]
    obs_share = obs.groupby("year").apply(lambda g: (g.family == "cervical_posterior").mean())
    rows = []
    shares = [("no posterior cases", 0.0), (f"Texas MS-DRG 473, 2016 ({obs_share[2016]:.3f})", float(obs_share[2016])),
              (f"Texas MS-DRG 473, 2019 ({obs_share[2019]:.3f})", float(obs_share[2019])),
              ("one quarter", 0.25), ("one half", 0.50), ("three quarters", 0.75), ("all posterior cases", 1.0)]
    for label, p in shares:
        m473 = {(473, "cervical_posterior"): 100 * p, (473, "cervical_anterior"): 100 * (1 - p)}
        mall = {}
        for drg, w in drg_mix.items():
            mall[(drg, "cervical_posterior")] = 100 * w * p
            mall[(drg, "cervical_anterior")] = 100 * w * (1 - p)
        g473, d473 = t2.hospital_result(pool, m473, bench, predm)
        gall, dall = t2.hospital_result(pool, mall, bench, predm)
        rows.append({"posterior_share_label": label, "posterior_share": p,
                     "gap_msdrg473": g473, "dollars_per_100_episodes_msdrg473": d473,
                     "gap_msdrg471_473": gall, "dollars_per_100_episodes_msdrg471_473": dall})
    out = pd.DataFrame(rows)
    out.attrs["drg_mix"] = drg_mix
    out.attrs["obs_n"] = obs.groupby("year").size().to_dict()
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    cms = an.cms_prices()
    d = t2.cohort()
    a = readmit_rows(d, cms)
    a.to_csv(I / "team_pcf_readmission.csv", index=False)
    print(a.round(4).to_string(index=False))
    b = gradient_rows(d, cms)
    b.to_csv(I / "team_posterior_share_gradient.csv", index=False)
    print(b.round(4).to_string(index=False))
    print("inpatient MS-DRG mix used across 471-473:", {k: round(v, 3) for k, v in b.attrs["drg_mix"].items()})
    print("Texas MS-DRG 473 ACDF plus posterior, n by year:", b.attrs["obs_n"])
