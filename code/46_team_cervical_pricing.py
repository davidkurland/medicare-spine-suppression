"""TEAM cervical spinal fusion pricing simulation (Medicare payment basis, unanchored to CMS target prices).

Question: under TEAM's pooled MS-DRG/HCPCS episode types, how do inpatient cervical fusion episodes compare with
their target prices, how much does TEAM-style risk adjustment capture, and how does the CY2026 removal of posterior
cervical fusion (CPT 22600) from the Inpatient-Only list change hospital performance?

TEAM rules (CMS TEAM overview; TEAM Clinical Episode Construction Specifications, Table 10; TEAM Risk
Adjustment and Preliminary Target Price Specifications; 42 CFR 512.545):
- Outpatient anchors 22551/22554 are mapped to MS-DRG 473 and pooled with inpatient DRG 473 episodes; 22600 is not
  an anchor (outpatient posterior cervical fusion triggers no TEAM episode). No setting adjuster in the risk model.
- Regional benchmark per MS-DRG/HCPCS episode type; reconciliation risk adjustment uses beneficiary factors of the
  episode (TEAM spinal fusion: 25 V28 HCCs, HCC count 0-4+, age group, prior post-acute care, economic risk; hospital
  bed size, safety net); final normalization = mean benchmark / mean risk-adjusted benchmark (capped +/-5% of
  prospective); retrospective trend = regional PY / baseline spending (capped +/-3% of prospective); 2% discount.
- Reconciliation compares a hospital's total PY spending with total target prices across its episodes in the
  spinal fusion episode category.

Episode spending (Medicare payments, 30 days; professional fees and readmissions from team_addons.py are added to every episode):
  inpatient = national 2023 average Medicare payment for the record's MS-DRG + post-acute payment by discharge
              destination (IRF case, SNF stay, LTCH case, home health period)
  outpatient ACDF (22551/22554) = CY2024 OPPS payment rate (APC 5115) + post-acute payment (assumed home-health rate
              PAC_OP_HH_RATE per episode)
Cohort: Texas PUDF 2016-2019 Medicare FFS inpatients aged >=65 in MS-DRGs 471-473 (TEAM cervical episode types,
unchanged by the FY2025 grouper), with TEAM spinal fusion V28 HCC flags from index-stay diagnoses.
Pool composition: outpatient ACDF share of the DRG 473 episode type from national data (PSPS hospital outpatient
22551+22554 services vs complete FFS DRG 473 discharges, 2022-2024).
Migration: posterior cervical fusion patients move outpatient (exit TEAM) with tier-aligned selection
(THETA, a fixed calibration constant for the suitability score, which includes a DRG-tier term; see cohort()).
Aggregates only are reported (Texas DUA).
Outputs: data/interim/team_unanchored_static.csv, team_unanchored_archetypes.csv, team_unanchored_migration.csv,
         team_unanchored_sensitivity.csv
"""
import importlib

import numpy as np
import pandas as pd

from common import ROOT

ts = importlib.import_module("45_team_simulation")
ta = importlib.import_module("team_addons")     # professional fees and readmissions, added to every episode (primary construction)
I = ROOT / "data/interim"

# ---- parameters ----
# Post-acute payment per IRF case (total payments / cases), SNF stay (total payments / stays), LTCH case and home health period
# (MedPAC Mar 2025 ch6-8, 2023; LTCH July 2024 data book, 2022)
PAC_PAY = {"IRF": 9.6e9 / 404000, "SNF": 30e9 / 1.6e6, "LTCH": 48582.0, "HH": 2022.0}
OPPS_ACDF = 12552.87                                                 # CY2024 OPPS Addendum B, APC 5115
PAC_OP_HH_RATE = 0.05                                                # share of outpatient ACDF episodes with home health
THETA = 12.0                                                         # fixed calibration constant for the suitability score (includes the DRG-tier term)
DISCOUNT = 0.02


def destination(ps):
    return np.select([ps.eq("62"), ps.isin(["03", "61", "64"]), ps.eq("63"), ps.eq("06")], ["IRF", "SNF", "LTCH", "HH"], "home")


def cohort(pac_pay=None):
    pac_pay = pac_pay or PAC_PAY
    d = ts.load()
    d = d[d.msdrg.isin([471, 472, 473])].reset_index(drop=True)
    z = lambda s: (s - s.mean()) / s.std()
    d["tier"] = d.msdrg.map({471: 2, 472: 1, 473: 0})
    d["suit"] = z(d.hcc_count) + z(d.age_grp.map({"65-74": 0, "75-84": 1, "85+": 2})) + z(d.n_poa.fillna(d.n_dx)) + z(d.soi) + z(d.tier)
    pay = pd.read_csv(I / "national_drg_payments.csv").query("year == 2023").set_index("drg").avg_medicare_payment
    d["dest"] = destination(d.pat_status.astype(str))
    d["pac_pay"] = d.dest.map(lambda x: pac_pay.get(x) or 0.0)
    d["spend"] = d.msdrg.map(pay) + d.pac_pay + ta.cervical_inpatient(d)
    d["setting"] = "inpatient"
    return d


def op_share_473():
    c = pd.read_csv(I / "setting_migration_cpt.csv", dtype={"cpt": str})
    op = c[c.year.between(2022, 2024) & (c.setting == "hospital outpatient") & c.cpt.isin(["22551", "22554"])].services.sum()
    g = pd.read_csv(I / "national_drg_totals.csv")
    ip = g[(g.drg == 473) & g.year.between(2022, 2024)].n.sum()
    return op / (op + ip)


def pooled_473(d, share_op, op_covariates="suitable_half", op_hh_rate=None, pac_pay=None):
    """Append synthetic outpatient ACDF episodes to the DRG 473 type at the national outpatient share.
    Covariates for outpatient episodes: the most suitable half of inpatient ACDF in DRG 473 (default) or all."""
    ip473 = d[d.msdrg == 473]
    acdf = ip473[ip473.family == "cervical_anterior"]
    donors = acdf[acdf.suit <= acdf.suit.median()] if op_covariates == "suitable_half" else acdf
    n_op = int(round(len(ip473) * share_op / (1 - share_op)))
    op = donors.sample(n=n_op, replace=True, random_state=7).copy()
    op["setting"], op["family"] = "outpatient", "cervical_anterior_op"
    rng = np.random.default_rng(11)
    op_hh_rate = PAC_OP_HH_RATE if op_hh_rate is None else op_hh_rate
    op["pac_pay"] = np.where(rng.random(n_op) < op_hh_rate, (pac_pay or PAC_PAY)["HH"], 0.0)
    op["spend"] = OPPS_ACDF + op.pac_pay + ta.cervical_outpatient()
    return pd.concat([d, op], ignore_index=True)


def risk_predict(pool):
    pool["pred"] = np.nan
    for drg in (471, 472, 473):
        ix = pool.msdrg == drg
        pool.loc[ix, "pred"] = ts.fit_predict(pool[ix], "spend")
    return pool


def static(pool):
    rows = []
    for drg in (471, 472, 473):
        g = pool[pool.msdrg == drg]
        bench = g.spend.mean()
        for lab, s in [("all episodes", g), ("inpatient, all", g[g.setting == "inpatient"]), ("outpatient ACDF", g[g.setting == "outpatient"]),
                       ("inpatient ACDF", g[(g.setting == "inpatient") & (g.family == "cervical_anterior")]),
                       ("inpatient posterior cervical", g[g.family == "cervical_posterior"])]:
            if len(s) <= 10:
                continue
            actual, risk_adj_target = s.spend.mean(), bench * s.pred.mean() / g.pred.mean()
            rows.append({"drg": drg, "group": lab, "n": len(s), "benchmark": bench, "mean_spend": actual,
                         "gap_unadjusted": actual / bench - 1, "risk_adjusted_target": risk_adj_target,
                         "gap_risk_adjusted": actual / risk_adj_target - 1,
                         "share_of_gap_captured": (risk_adj_target - bench) / (actual - bench) if abs(actual - bench) > 1 else np.nan,
                         "irf_snf_rate": s.dest.isin(["IRF", "SNF", "LTCH"]).mean()})
    return pd.DataFrame(rows)


def hospital_result(pool, mix, bench_by_drg, pred_mean_by_drg, trend=1.0, norm=None):
    """mix: dict of (drg, setting_family) -> episode count for a hospital; returns total spend / total target - 1."""
    spend = target = 0.0
    for (drg, fam), n in mix.items():
        g = pool[(pool.msdrg == drg) & (pool.family == fam)]
        if n == 0 or len(g) == 0:
            continue
        q = g.pred.mean() / pred_mean_by_drg[drg]
        nf = 1.0 if norm is None else norm[drg]
        spend += n * g.spend.mean()
        target += n * bench_by_drg[drg] * trend * q * nf
    return spend / (target * (1 - DISCOUNT)) - 1, spend - target * (1 - DISCOUNT)


ARCHETYPES = {
    "outpatient-first ACDF hospital": {(473, "cervical_anterior_op"): 75, (473, "cervical_anterior"): 5, (472, "cervical_anterior"): 15, (471, "cervical_anterior"): 5},
    "regional-average mix": None,
    "inpatient ACDF hospital": {(473, "cervical_anterior"): 55, (472, "cervical_anterior"): 35, (471, "cervical_anterior"): 10},
    "complex spine center (posterior-heavy)": {(473, "cervical_posterior"): 25, (472, "cervical_posterior"): 30, (471, "cervical_posterior"): 15,
                                               (473, "cervical_anterior"): 15, (472, "cervical_anterior"): 10, (471, "cervical_anterior"): 5},
}


def archetypes(pool):
    bench = {drg: pool[pool.msdrg == drg].spend.mean() for drg in (471, 472, 473)}
    predm = {drg: pool[pool.msdrg == drg].pred.mean() for drg in (471, 472, 473)}
    counts = pool.groupby(["msdrg", "family"]).size()
    arche = dict(ARCHETYPES)
    arche["regional-average mix"] = {k: v * 100 / counts.sum() for k, v in counts.items()}
    rows = []
    for k, v in arche.items():
        gap, dollars = hospital_result(pool, v, bench, predm)
        rows.append({"archetype": k, "gap_vs_target": gap, "dollars_per_100_episodes": dollars})
    return pd.DataFrame(rows), bench, predm


def run(pac_pay=None, op_covariates="suitable_half", op_hh_rate=None):
    d = cohort(pac_pay)
    s = op_share_473()
    pool = risk_predict(pooled_473(d, s, op_covariates, op_hh_rate, pac_pay))
    return pool, s


if __name__ == "__main__":
    pool, s = run()
    st = static(pool)
    st.insert(0, "op_share_473_pool", s)
    st.to_csv(I / "team_unanchored_static.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"outpatient share of DRG 473 episode type: {s:.3f}")
    print(st.round(3).to_string(index=False))

    ar, bench, predm = archetypes(pool)
    ar.to_csv(I / "team_unanchored_archetypes.csv", index=False)
    print(ar.round(3).to_string(index=False))

    # 2026 dynamic: regional migration of posterior cervical fusion out of TEAM
    rows = []
    for m in (0.10, 0.22, 0.35):
        w = ts.expected_weights(pool, ["cervical_posterior"], m, THETA)
        reg = pool.assign(w=w)
        reg_bench = {drg: (reg[reg.msdrg == drg].w * reg[reg.msdrg == drg].spend).sum() / reg[reg.msdrg == drg].w.sum() for drg in (471, 472, 473)}
        trend = {drg: min(max(reg_bench[drg] / bench[drg], 0.97), 1.03) for drg in (471, 472, 473)}
        reg_pred = {drg: (reg[reg.msdrg == drg].w * reg[reg.msdrg == drg].pred).sum() / reg[reg.msdrg == drg].w.sum() for drg in (471, 472, 473)}
        norm = {drg: min(max(predm[drg] / reg_pred[drg], 0.95), 1.05) for drg in (471, 472, 473)}
        for label, mh in [("keeps posterior cervical inpatient", 0.0), ("migrates at regional rate", m), ("migrates at twice regional rate", min(2 * m, 0.9))]:
            wh = ts.expected_weights(pool, ["cervical_posterior"], mh, THETA)
            ph = pool.assign(w=wh)
            spend = target = 0.0
            for drg in (471, 472, 473):
                g = ph[ph.msdrg == drg]
                n = g.w.sum()
                spend += (g.w * g.spend).sum()
                target += n * bench[drg] * trend[drg] * ((g.w * g.pred).sum() / n / predm[drg]) * norm[drg]
            rows.append({"regional_migration": m, "hospital": label, "hospital_migration": mh, "trend_473": trend[473], "norm_473": norm[473],
                         "gap_vs_target": spend / (target * (1 - DISCOUNT)) - 1})
    mg = pd.DataFrame(rows)
    mg.to_csv(I / "team_unanchored_migration.csv", index=False)
    print(mg.round(4).to_string(index=False))

    # sensitivity grid
    base_pac = dict(PAC_PAY)
    variants = [("base", {}), ("outpatient covariates = all inpatient ACDF", {"op_covariates": "all"}),
                ("outpatient home health rate 0%", {"op_hh_rate": 0.0}), ("outpatient home health rate 15%", {"op_hh_rate": 0.15}),
                ("IRF payment halved", {"pac_pay": {**base_pac, "IRF": base_pac["IRF"] / 2}}),
                ("SNF payment x0.8 (30-day window)", {"pac_pay": {**base_pac, "SNF": base_pac["SNF"] * 0.8}}),
                ("IRF discharges paid as SNF", {"pac_pay": {**base_pac, "IRF": base_pac["SNF"]}})]
    sens = []
    for name, kw in variants:
        pl, sh = run(**kw)
        stv = static(pl).set_index(["drg", "group"])
        arv, _, _ = archetypes(pl)
        arv = arv.set_index("archetype")
        sens.append({"variant": name,
                     "drg473_inpatient_posterior_gap": stv.loc[(473, "inpatient posterior cervical"), "gap_unadjusted"],
                     "drg473_inpatient_posterior_gap_risk_adj": stv.loc[(473, "inpatient posterior cervical"), "gap_risk_adjusted"],
                     "drg473_inpatient_acdf_gap_risk_adj": stv.loc[(473, "inpatient ACDF"), "gap_risk_adjusted"],
                     "complex_center_gap": arv.loc["complex spine center (posterior-heavy)", "gap_vs_target"],
                     "outpatient_first_gap": arv.loc["outpatient-first ACDF hospital", "gap_vs_target"]})
    for th in (6.0, 20.0):
        w = ts.expected_weights(pool, ["cervical_posterior"], 0.22, th)
        sens.append({"variant": f"migration selection theta={th:g} (22% regional)",
                     "drg473_inpatient_posterior_gap": np.nan, "drg473_inpatient_posterior_gap_risk_adj": np.nan,
                     "drg473_inpatient_acdf_gap_risk_adj": np.nan, "complex_center_gap": np.nan, "outpatient_first_gap": np.nan,
                     "pool_spending_ratio_473": (w * pool.spend)[pool.msdrg == 473].sum() / w[pool.msdrg == 473].sum() / pool[pool.msdrg == 473].spend.mean()})
    sens = pd.DataFrame(sens)
    sens.to_csv(I / "team_unanchored_sensitivity.csv", index=False)
    print(sens.round(3).to_string(index=False))
