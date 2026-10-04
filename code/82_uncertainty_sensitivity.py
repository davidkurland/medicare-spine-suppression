"""Uncertainty and sensitivity analyses. Each block bounds a main estimate, puts an interval on it, or restates it
per beneficiary.

1. All-payer rates. Medicare inpatients per total Medicare beneficiary in the HCUPnet balanced panel (spine fusion,
   joint replacement, cranial), beside CMS FFS discharges per FFS enrollee, with intervals from resampling states.
2. TEAM bootstrap. Hospital-cluster bootstrap of the anchored cervical (51_team_cms_anchor) and lumbar (53_team_lumbar_anchor) simulations.
3. TEAM risk-adjustment bound. CMS reads hierarchical condition categories from 180 days of prior claims; we read
   them from the index stay. We ask how much more comorbidity signal a lookback would have to add, for inpatients
   only, to close the posterior cervical gap.
4. TEAM one-way sensitivity (source for Figure 5D): capture factor, outpatient share, post-acute payment, outpatient
   covariates, region, period, case mix, and the comorbidity bound.
5. Decomposition of the MS-DRG 473 posterior cervical gap into a procedure part (vs an inpatient-only price) and a
   pooling part (the outpatient episodes in the price).

Texas records are used under their data use agreement: aggregates only, no cell of 10 or fewer.
Outputs: data/interim/allpayer_rates.csv, team_bootstrap.csv, hcc_bound.csv, team_tornado.csv,
         pcf_decomposition.csv
"""
import importlib
import warnings

import numpy as np
import pandas as pd

from common import ROOT

warnings.filterwarnings("ignore")
ts = importlib.import_module("45_team_simulation")
t2 = importlib.import_module("46_team_cervical_pricing")
an = importlib.import_module("51_team_cms_anchor")
lu = importlib.import_module("53_team_lumbar_anchor")
rb = importlib.import_module("56_team_anchor_robustness")
hs = importlib.import_module("52_hidden_share_team_episodes")
I = ROOT / "data/interim"
B_TEAM, B_STATES = 2000, 5000
PCF = (473, "inpatient posterior cervical")


# ---------------------------------------------------------------- 1. all-payer rates
def balanced(p, y0=2016, y1=2023):
    have = p.groupby("st").year.apply(lambda s: set(range(y0, y1 + 1)) <= set(s))
    return p[p.st.isin(have[have].index) & p.year.isin([y0, y1])]


def rate_changes(w, y0=2016, y1=2023):
    b = w.groupby("year")[["hcup_medicare", "cms_ffs_total", "cms_ffs_published", "TOT_BENES", "A_ORGNL_MDCR_BENES"]].sum()
    chg = lambda c: b.loc[y1, c] / b.loc[y0, c] - 1
    rate = lambda n, den: (b.loc[y1, n] / b.loc[y1, den]) / (b.loc[y0, n] / b.loc[y0, den]) - 1
    return {"allpayer_count": chg("hcup_medicare"), "allpayer_per_beneficiary": rate("hcup_medicare", "TOT_BENES"),
            "cms_complete_count": chg("cms_ffs_total"), "cms_complete_per_ffs_enrollee": rate("cms_ffs_total", "A_ORGNL_MDCR_BENES"),
            "cms_published_count": chg("cms_ffs_published"), "cms_published_per_ffs_enrollee": rate("cms_ffs_published", "A_ORGNL_MDCR_BENES"),
            "medicare_beneficiaries": chg("TOT_BENES"), "ffs_enrollees": chg("A_ORGNL_MDCR_BENES")}


def allpayer_rates():
    panels = {"spine fusion": pd.read_csv(I / "multistate_panel.csv").query("category == 'fusion'"),
              "joint replacement (469, 470)": pd.read_csv(I / "multistate_joints_panel.csv"),
              "cranial (23, 25-27)": pd.read_csv(I / "multistate_cranial_panel.csv")}
    rng, rows = np.random.default_rng(20261001), []
    for line, p in panels.items():
        if "line" in p.columns and p.line.nunique() > 1:
            p = p[p.line == line]
        w = balanced(p)
        states = sorted(w.st.unique())
        point = rate_changes(w)
        by_state = {s: g for s, g in w.groupby("st")}
        boot = []
        for _ in range(B_STATES):
            pick = rng.choice(states, size=len(states), replace=True)
            boot.append(rate_changes(pd.concat([by_state[s] for s in pick])))
        boot = pd.DataFrame(boot)
        for k, v in point.items():
            rows.append({"line": line, "n_states": len(states), "measure": k, "change": v,
                         "lo": boot[k].quantile(0.025), "hi": boot[k].quantile(0.975)})
    out = pd.DataFrame(rows)
    out.to_csv(I / "allpayer_rates.csv", index=False)
    return out


# ---------------------------------------------------------------- shared TEAM helpers
def pooled(d, share, seed=7, op_covariates="suitable_half", op_hh_rate=None, pac_pay=None):
    """46_team_cervical_pricing.pooled_473 with the seeds exposed, so the bootstrap redraws the outpatient donors each replicate."""
    ip473 = d[d.msdrg == 473]
    acdf = ip473[ip473.family == "cervical_anterior"]
    donors = acdf[acdf.suit <= acdf.suit.median()] if op_covariates == "suitable_half" else acdf
    n_op = int(round(len(ip473) * share / (1 - share)))
    op = donors.sample(n=n_op, replace=True, random_state=seed).copy()
    op["setting"], op["family"] = "outpatient", "cervical_anterior_op"
    rng = np.random.default_rng(seed + 4)
    hh = t2.PAC_OP_HH_RATE if op_hh_rate is None else op_hh_rate
    op["pac_pay"] = np.where(rng.random(n_op) < hh, (pac_pay or t2.PAC_PAY)["HH"], 0.0)
    op["spend"] = t2.OPPS_ACDF + op.pac_pay + t2.ta.cervical_outpatient()
    return pd.concat([d, op], ignore_index=True)


def cerv_run(d, cms, region=an.PRIMARY, seed=7, k_mult=1.0, share=None, k_rule="mean", **kw):
    k, a473, sh, ip = an.anchor(d, cms, region)
    if k_rule != "mean":                      # capture factor from MS-DRG 472 alone, or set to 1
        k = {"472": ip[472] / cms.loc[(region, 472), "benchmark"], "one": 1.0}[k_rule]
    if k_mult != 1.0 or k_rule != "mean":
        op_mean = t2.OPPS_ACDF + t2.PAC_OP_HH_RATE * t2.PAC_PAY["HH"] + t2.ta.cervical_outpatient()
        sh = float(np.clip((ip[473] - k * k_mult * cms.loc[(region, 473), "benchmark"]) / (ip[473] - op_mean), 0.02, 0.95))
    if share is not None:
        sh = share
    pool = t2.risk_predict(pooled(d, sh, seed, **kw))
    st = t2.static(pool).set_index(["drg", "group"])
    ar, _, _ = t2.archetypes(pool)
    ar = ar.set_index("archetype")
    cx, of = ar.loc["complex spine center (posterior-heavy)"], ar.loc["outpatient-first ACDF hospital"]
    res = {"implied_outpatient_share": sh, "capture_factor_k": k * k_mult,
           "pcf_gap_unadjusted": st.loc[PCF, "gap_unadjusted"], "pcf_gap_risk_adjusted": st.loc[PCF, "gap_risk_adjusted"],
           "pcf_share_of_gap_captured": st.loc[PCF, "share_of_gap_captured"],
           "ip_acdf_gap_risk_adjusted": st.loc[(473, "inpatient ACDF"), "gap_risk_adjusted"],
           "op_acdf_gap_unadjusted": st.loc[(473, "outpatient ACDF"), "gap_unadjusted"],
           "pcf472_gap_risk_adjusted": st.loc[(472, "inpatient posterior cervical"), "gap_risk_adjusted"],
           "complex_center_gap": cx.gap_vs_target, "complex_center_dollars_per_100": cx.dollars_per_100_episodes,
           "outpatient_first_gap": of.gap_vs_target}
    return res, pool, st


def cluster_resample(d, rng):
    groups = d.groupby("hosp").indices
    hosp = np.array(list(groups))
    pick = rng.choice(hosp, size=len(hosp), replace=True)
    return d.iloc[np.concatenate([groups[h] for h in pick])].reset_index(drop=True)


LUMBAR_ARCH = {
    "outpatient-first lumbar hospital": {(451, "outpatient posterolateral/PLIF (22612, 22630)"): 60, (451, "inpatient posterior"): 20,
                                         (402, "outpatient TLIF (22633)"): 10, (448, "inpatient posterior"): 10},
    "inpatient single-level hospital": {(451, "inpatient posterior"): 70, (450, "inpatient posterior"): 10, (448, "inpatient posterior"): 20},
    "complex lumbar center (multilevel, combined, ALIF)": {(447, "inpatient posterior"): 15, (448, "inpatient posterior"): 25,
                                                          (426, "inpatient posterior"): 10, (427, "inpatient posterior"): 15, (428, "inpatient posterior"): 10,
                                                          (451, "inpatient anterior (ALIF)"): 10, (451, "inpatient posterior"): 15}}


def lumbar_run(lum, cms, region=an.PRIMARY):
    """The anchoring in 53_team_lumbar_anchor as a function of the cohort."""
    ip = {drg: lum[lum.drg25 == drg].spend.mean() for drg in lu.LUMBAR}
    k_single = np.mean([ip[x] / cms.loc[(region, x), "benchmark"] for x in (447, 448, 450)])
    k_comb = np.mean([ip[x] / cms.loc[(region, x), "benchmark"] for x in (426, 427, 428)])
    op_mean = lu.OPPS_LUMBAR + t2.PAC_OP_HH_RATE * t2.PAC_PAY["HH"]
    s451 = float(np.clip((ip[451] - k_single * cms.loc[(region, 451), "benchmark"]) / (ip[451] - op_mean - t2.ta.lumbar_outpatient(451)), 0.02, 0.95))
    s402 = float(np.clip((ip[402] - k_comb * cms.loc[(region, 402), "benchmark"]) / (ip[402] - op_mean - t2.ta.lumbar_outpatient(402)), 0.02, 0.95))
    pool = lu.add_outpatient(lum, 451, s451, "outpatient posterolateral/PLIF (22612, 22630)")
    pool = lu.add_outpatient(pool, 402, s402, "outpatient TLIF (22633)", donors_family="lumbar_posterior")
    pool = lu.risk_predict(pool, lu.LUMBAR)
    bench = {drg: pool[pool.drg25 == drg].spend.mean() for drg in lu.LUMBAR}
    predm = {drg: pool[pool.drg25 == drg].pred.mean() for drg in lu.LUMBAR}

    def gap(drg, grp):
        g = pool[(pool.drg25 == drg) & (pool.grp == grp)]
        return g.spend.mean() / bench[drg] - 1, g.spend.mean() / (bench[drg] * g.pred.mean() / predm[drg]) - 1

    def arch(mix):
        spend = target = 0.0
        for (drg, grp), n in mix.items():
            g = pool[(pool.drg25 == drg) & (pool.grp == grp)]
            if n == 0 or len(g) == 0 or np.isnan(g.pred.mean()):
                continue
            spend += n * g.spend.mean()
            target += n * bench[drg] * (g.pred.mean() / predm[drg])
        return spend / (target * (1 - t2.DISCOUNT)) - 1

    u, r = gap(402, "inpatient posterior")
    res = {"lumbar402_ip_gap_unadjusted": u, "lumbar402_ip_gap_risk_adjusted": r,
           "lumbar402_op_tlif_gap_unadjusted": gap(402, "outpatient TLIF (22633)")[0],
           "lumbar451_ip_gap_risk_adjusted": gap(451, "inpatient posterior")[1]}
    res.update({f"arch_{k}": arch(v) for k, v in LUMBAR_ARCH.items()})
    return res, pool


def team_bootstrap(cms):
    rng = np.random.default_rng(20261001)
    d = t2.cohort()
    point, _, _ = cerv_run(d, cms)
    boot = []
    for b in range(B_TEAM):
        try:
            boot.append(cerv_run(cluster_resample(d, rng), cms, seed=1000 + b)[0])
        except KeyError:        # a replicate whose posterior cervical cell falls to 10 or fewer is skipped
            continue
    boot = pd.DataFrame(boot)
    rows = [{"model": "cervical", "metric": k, "point": v, "lo": boot[k].quantile(0.025), "hi": boot[k].quantile(0.975),
             "replicates": len(boot), "hospitals": d.hosp.nunique(), "episodes": len(d)} for k, v in point.items()]

    lum = lu.lumbar_cohort()
    lum = lum[lum.drg25.isin(lu.LUMBAR)].reset_index(drop=True)
    lpoint, _ = lumbar_run(lum, cms)
    lboot = []
    for b in range(B_TEAM // 4):
        try:
            lboot.append(lumbar_run(cluster_resample(lum, rng), cms)[0])
        except Exception:
            continue
    lboot = pd.DataFrame(lboot)
    rows += [{"model": "lumbar", "metric": k, "point": v, "lo": lboot[k].quantile(0.025), "hi": lboot[k].quantile(0.975),
              "replicates": len(lboot), "hospitals": lum.hosp.nunique(), "episodes": len(lum)} for k, v in lpoint.items()]
    out = pd.DataFrame(rows)
    out.to_csv(I / "team_bootstrap.csv", index=False)
    return out


# ---------------------------------------------------------------- 3. bound on risk adjustment
def hcc_parts(pool, drg=473):
    """Fit the TEAM-style risk model for one episode type and split each prediction into the part carried by the
    comorbidity terms (HCC flags and HCC count) and the rest (intercept and age)."""
    g = pool[pool.msdrg == drg]
    X = ts.design(g)
    beta, *_ = np.linalg.lstsq(X.values, g.spend.values, rcond=None)
    hcc_cols = [i for i, c in enumerate(X.columns) if c.startswith("hcc")]
    hcc_part = X.values[:, hcc_cols] @ beta[hcc_cols]
    return g, X.values @ beta, hcc_part


def gap_with_scaled_capture(g, pred, hcc_part, lam, who):
    """Residual posterior cervical gap if a lookback multiplied the comorbidity signal by `lam` for the rows in `who`."""
    p2 = pred + (lam - 1.0) * hcc_part * who
    pcf = (g.family == "cervical_posterior").values
    target = g.spend.mean() * p2[pcf].mean() / p2.mean()
    return g.spend.values[pcf].mean() / target - 1


def hcc_bound(cms):
    d = t2.cohort()
    _, pool, st = cerv_run(d, cms)
    g, pred, hp = hcc_parts(pool)
    ip, pcf, op = (g.setting == "inpatient").values, (g.family == "cervical_posterior").values, (g.setting == "outpatient").values
    bench, actual, target = st.loc[PCF, "benchmark"], st.loc[PCF, "mean_spend"], st.loc[PCF, "risk_adjusted_target"]
    rows = [{"quantity": "pooled benchmark, $", "value": bench}, {"quantity": "posterior cervical mean spend, $", "value": actual},
            {"quantity": "posterior cervical risk-adjusted target, $", "value": target},
            {"quantity": "target lift from risk adjustment, %", "value": target / bench - 1},
            {"quantity": "target lift needed to close the gap, %", "value": actual / bench - 1},
            {"quantity": "needed lift / achieved lift", "value": (actual - bench) / (target - bench)},
            {"quantity": "mean HCC count, outpatient ACDF", "value": g.hcc_count.values[op].mean()},
            {"quantity": "mean HCC count, inpatient ACDF", "value": g.hcc_count.values[ip & ~pcf & (g.family == 'cervical_anterior').values].mean()},
            {"quantity": "mean HCC count, inpatient posterior cervical", "value": g.hcc_count.values[pcf].mean()},
            {"quantity": "comorbidity-attributable prediction, outpatient ACDF, $", "value": hp[op].mean()},
            {"quantity": "comorbidity-attributable prediction, inpatient posterior cervical, $", "value": hp[pcf].mean()}]
    for lam in (1.5, 2.0, 3.0, 5.0):
        rows.append({"quantity": f"residual gap, comorbidity signal x{lam:g} for inpatients only", "value": gap_with_scaled_capture(g, pred, hp, lam, ip)})
        rows.append({"quantity": f"residual gap, comorbidity signal x{lam:g} for posterior cervical only", "value": gap_with_scaled_capture(g, pred, hp, lam, pcf)})
    for who, lab in [(ip, "inpatients only"), (pcf, "posterior cervical only")]:
        lo, hi = 1.0, 200.0
        if gap_with_scaled_capture(g, pred, hp, hi, who) > 0:
            rows.append({"quantity": f"multiple of comorbidity signal that closes the gap, {lab}", "value": np.inf})
            continue
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if gap_with_scaled_capture(g, pred, hp, mid, who) > 0 else (lo, mid)
        rows.append({"quantity": f"multiple of comorbidity signal that closes the gap, {lab}", "value": (lo + hi) / 2})
    # every posterior cervical episode scored at the 95th percentile of predicted spending in the pool
    p95 = np.where(pcf, np.percentile(pred, 95), pred)
    rows.append({"quantity": "residual gap, every posterior cervical episode at the pool's 95th percentile prediction",
                 "value": actual / (bench * p95[pcf].mean() / p95.mean()) - 1})
    out = pd.DataFrame(rows)
    out.to_csv(I / "hcc_bound.csv", index=False)
    return out, (g, pred, hp, ip)


# ---------------------------------------------------------------- 4. one-way sensitivity
def tornado(cms, bound):
    d = t2.cohort()
    base_pac = dict(t2.PAC_PAY)
    base = cerv_run(d, cms)[0]
    g, pred, hp, ip = bound
    rows = []

    def add(family, spec, res):
        rows.append({"family": family, "spec": spec, "pcf_gap_risk_adjusted": res["pcf_gap_risk_adjusted"],
                     "complex_center_gap": res.get("complex_center_gap", np.nan),
                     "implied_outpatient_share": res.get("implied_outpatient_share", np.nan)})

    add("Base", "primary specification", base)
    for km in (0.9, 1.1):
        add("Capture factor", f"k x{km:g}", cerv_run(d, cms, k_mult=km)[0])
    for sh in (0.20, 0.40):
        add("Outpatient share of pool", f"{sh:.0%}", cerv_run(d, cms, share=sh)[0])
    add("Outpatient share of pool", "as counted from Part B claims", cerv_run(d, cms, share=t2.op_share_473())[0])
    add("Capture factor rule", "from MS-DRG 472 only", cerv_run(d, cms, k_rule="472")[0])
    add("Capture factor rule", "set to 1 (nothing left unmeasured)", cerv_run(d, cms, k_rule="one")[0])
    for lv in ("low", "high"):                # the measured additions: professional fee bundles and readmission inputs
        t2.ta.set_level(fee=lv)
        add("Professional fees", f"bundle assumptions {lv}", cerv_run(t2.cohort(), cms)[0])
        t2.ta.set_level(fee="base", readmit=lv)
        add("Readmissions", f"inputs {lv}", cerv_run(t2.cohort(), cms)[0])
        t2.ta.set_level(readmit="base")
    t2.ta.set_level(enabled=False)
    add("Construction", "without professional fees or readmissions", cerv_run(t2.cohort(), cms)[0])
    t2.ta.set_level(enabled=True)

    def pac_run(pp):
        dd = t2.cohort(pp)
        return cerv_run(dd, cms, pac_pay=pp)[0]
    add("Post-acute payment", "IRF payment halved", pac_run({**base_pac, "IRF": base_pac["IRF"] / 2}))
    add("Post-acute payment", "SNF payment x0.8", pac_run({**base_pac, "SNF": base_pac["SNF"] * 0.8}))
    add("Post-acute payment", "IRF discharges paid as SNF", pac_run({**base_pac, "IRF": base_pac["SNF"]}))
    add("Post-acute payment", "all institutional post-acute payment halved",
        pac_run({**base_pac, "IRF": base_pac["IRF"] / 2, "SNF": base_pac["SNF"] / 2, "LTCH": base_pac["LTCH"] / 2}))
    # half of institutional post-acute discharges re-paid at the home health rate (the shift from facility to home)
    rng = np.random.default_rng(5)
    dd = d.copy()
    move = dd.dest.isin(["IRF", "SNF", "LTCH"]) & (rng.random(len(dd)) < 0.5)
    dd.loc[move, "spend"] = dd.loc[move, "spend"] - dd.loc[move, "pac_pay"] + base_pac["HH"]
    dd.loc[move, "pac_pay"] = base_pac["HH"]
    add("Post-acute payment", "half of institutional discharges sent home with home health", cerv_run(dd, cms)[0])
    add("Outpatient episodes", "covariates from all inpatient ACDF", cerv_run(d, cms, op_covariates="all")[0])
    add("Outpatient episodes", "home health after 0%", cerv_run(d, cms, op_hh_rate=0.0)[0])
    add("Outpatient episodes", "home health after 15%", cerv_run(d, cms, op_hh_rate=0.15)[0])
    for r, name in rb.DIV.items():
        add("Census division", name, cerv_run(d, cms, region=r)[0])
    for lab, yrs in [("2016-2017", (2016, 2017)), ("2018-2019", (2018, 2019))]:
        add("Period", lab, cerv_run(d[d.year.isin(yrs)].reset_index(drop=True), cms)[0])
    tot = hs.complete()
    for y in (2019, 2024):
        add("National case mix", f"{y} mix", cerv_run(rb.reweight(d, rb.national_mix(tot, y)), cms)[0])
    for lam in (2.0, 3.0):
        add("Comorbidity capture", f"signal x{lam:g} for inpatients only", {"pcf_gap_risk_adjusted": gap_with_scaled_capture(g, pred, hp, lam, ip)})
    out = pd.DataFrame(rows)
    out.to_csv(I / "team_tornado.csv", index=False)
    return out


# ---------------------------------------------------------------- 5. procedure part vs pooling part
def decomposition(cms):
    d = t2.cohort()
    rows = []
    for region in an.REGIONS:
        _, pool, st = cerv_run(d, cms, region=region)
        for drg in (471, 472, 473):
            s = st.loc[(drg, "inpatient posterior cervical")]
            ip_mean = st.loc[(drg, "inpatient, all"), "mean_spend"]
            g = pool[pool.msdrg == drg]
            rows.append({"region": region, "drg": drg, "outpatient_share": (g.setting == "outpatient").mean(),
                         "pooled_price": s.benchmark, "inpatient_only_price": ip_mean, "pcf_spend": s.mean_spend,
                         "gap_vs_pooled": s.gap_unadjusted, "procedure_part_vs_inpatient_only": s.mean_spend / ip_mean - 1,
                         "pooling_part_points": s.gap_unadjusted - (s.mean_spend / ip_mean - 1),
                         "pcf_institutional_postacute_rate": s.irf_snf_rate,
                         "ip_acdf_institutional_postacute_rate": st.loc[(drg, "inpatient ACDF"), "irf_snf_rate"]})
    out = pd.DataFrame(rows)
    out.to_csv(I / "pcf_decomposition.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_colwidth", 80)
    cms = an.cms_prices()
    ar = allpayer_rates()
    print("\n1. ALL-PAYER RATES, 2016-2023\n", ar.assign(**{c: (ar[c] * 100).round(1) for c in ("change", "lo", "hi")}).to_string(index=False))
    hb, bound = hcc_bound(cms)
    print("\n3. RISK-ADJUSTMENT BOUND\n", hb.round(4).to_string(index=False))
    dc = decomposition(cms)
    print("\n5. PROCEDURE VS POOLING\n", dc.round(3).to_string(index=False))
    tn = tornado(cms, bound)
    print("\n4. ONE-WAY SENSITIVITY\n", tn.round(3).to_string(index=False))
    tb = team_bootstrap(cms)
    print("\n2. TEAM BOOTSTRAP\n", tb.round(4).to_string(index=False))
