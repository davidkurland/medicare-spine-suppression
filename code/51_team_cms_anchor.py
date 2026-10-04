"""TEAM cervical fusion simulation re-anchored to CMS's published PY1 preliminary target prices.

CMS published DRG-region (census division) preliminary target prices for PY1 on 2025-12-10
(data/raw/team/team-py1-prlim-target-prices.xlsx; benchmark x trend x (1 - 2%)). The benchmark for the MS-DRG 473
episode type pools outpatient ACDF (HCPCS 22551/22554 mapped to DRG 473) with inpatient DRG 473 stays.

Our construct (46_team_cervical_pricing: national DRG payment + post-acute payment + professional fees and readmissions
from team_addons.py; no other Part B) captures a stable fraction of CMS standardized episode spending on the two cervical tiers that have
no outpatient component (471, 472). We use that fraction, k, to express CMS's 473 benchmark in our construct, then
solve for the outpatient share of the pool that reproduces it:
    share_op = (mean inpatient 473 spend - k * CMS 473 benchmark) / (mean inpatient 473 spend - mean outpatient spend)
Everything downstream (risk adjustment, archetypes, 2026 migration) reruns with that share. Two regions: the West
South Central division (our Texas records) and the Middle Atlantic division (New York).
Outputs: data/interim/team_anchor_variants_summary.csv, team_anchor_variants_static.csv, team_anchor_variants_archetypes.csv,
         team_anchor_variants_migration.csv
"""
import importlib

import numpy as np
import pandas as pd

from common import ROOT

t2 = importlib.import_module("46_team_cervical_pricing")
ts = importlib.import_module("45_team_simulation")
I = ROOT / "data/interim"
XLSX = ROOT / "data/raw/team/team-py1-prlim-target-prices.xlsx"
REGIONS = {"CENS_DIV_7": "West South Central (AR, LA, OK, TX)", "CENS_DIV_2": "Middle Atlantic (NJ, NY, PA)"}


def cms_prices():
    df = pd.read_excel(XLSX, sheet_name="Prelim_TP_DRG_Reg", header=7)
    df.columns = ["sort", "episode", "drg", "region", "region_desc", "benchmark", "trend", "discount", "target"]
    df = df[pd.to_numeric(df.drg, errors="coerce").notna()].copy()
    df["drg"] = df.drg.astype(int)
    for c in ["benchmark", "trend", "discount", "target"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    sf = df[df.episode.str.contains("Spinal", na=False)]
    return sf.groupby(["region", "drg"])[["benchmark", "trend", "discount", "target"]].first()


def anchor(d, cms, region):
    ip = {drg: d[d.msdrg == drg].spend.mean() for drg in (471, 472, 473)}
    k = float(np.mean([ip[471] / cms.loc[(region, 471), "benchmark"], ip[472] / cms.loc[(region, 472), "benchmark"]]))
    anchored_473 = k * cms.loc[(region, 473), "benchmark"]
    op_mean = t2.OPPS_ACDF + t2.PAC_OP_HH_RATE * t2.PAC_PAY["HH"] + t2.ta.cervical_outpatient()
    share = float(np.clip((ip[473] - anchored_473) / (ip[473] - op_mean), 0.02, 0.95))
    return k, anchored_473, share, ip


def migration(pool, bench, predm):
    rows = []
    for m in (0.10, 0.22, 0.35):
        w = ts.expected_weights(pool, ["cervical_posterior"], m, t2.THETA)
        reg = pool.assign(w=w)
        reg_bench = {drg: (reg[reg.msdrg == drg].w * reg[reg.msdrg == drg].spend).sum() / reg[reg.msdrg == drg].w.sum() for drg in (471, 472, 473)}
        trend = {drg: min(max(reg_bench[drg] / bench[drg], 0.97), 1.03) for drg in (471, 472, 473)}
        reg_pred = {drg: (reg[reg.msdrg == drg].w * reg[reg.msdrg == drg].pred).sum() / reg[reg.msdrg == drg].w.sum() for drg in (471, 472, 473)}
        norm = {drg: min(max(predm[drg] / reg_pred[drg], 0.95), 1.05) for drg in (471, 472, 473)}
        for label, mh in [("keeps posterior cervical inpatient", 0.0), ("migrates at regional rate", m), ("migrates at twice regional rate", min(2 * m, 0.9))]:
            wh = ts.expected_weights(pool, ["cervical_posterior"], mh, t2.THETA)
            ph = pool.assign(w=wh)
            spend = target = 0.0
            for drg in (471, 472, 473):
                g = ph[ph.msdrg == drg]
                n = g.w.sum()
                spend += (g.w * g.spend).sum()
                target += n * bench[drg] * trend[drg] * ((g.w * g.pred).sum() / n / predm[drg]) * norm[drg]
            rows.append({"regional_migration": m, "hospital": label, "hospital_migration": mh, "trend_473": trend[473], "norm_473": norm[473],
                         "gap_vs_target": spend / (target * (1 - t2.DISCOUNT)) - 1})
    return pd.DataFrame(rows)


def run_anchored(region, cms, pac_pay=None, op_covariates="suitable_half", op_hh_rate=None):
    d = t2.cohort(pac_pay)
    k, a473, share, ip = anchor(d, cms, region)
    pool = t2.risk_predict(t2.pooled_473(d, share, op_covariates, op_hh_rate, pac_pay))
    return pool, k, a473, share, ip


PRIMARY = "CENS_DIV_7"     # our records are Texas, so the West South Central division is the primary anchoring


if __name__ == "__main__":
    cms = cms_prices()
    summary, statics, arches, migs = [], [], [], []
    for region, desc in REGIONS.items():
        pool, k, a473, share, ip = run_anchored(region, cms)
        st = t2.static(pool)
        ar, bench, predm = t2.archetypes(pool)
        mg = migration(pool, bench, predm)
        for df in (st, ar, mg):
            df.insert(0, "region", region)
        statics.append(st); arches.append(ar); migs.append(mg)
        s = st.set_index(["drg", "group"]); a = ar.set_index("archetype")
        summary.append({"region": region, "region_desc": desc,
                        "cms_benchmark_471": cms.loc[(region, 471), "benchmark"], "cms_benchmark_472": cms.loc[(region, 472), "benchmark"],
                        "cms_benchmark_473": cms.loc[(region, 473), "benchmark"], "cms_target_473": cms.loc[(region, 473), "target"],
                        "cms_trend_473": cms.loc[(region, 473), "trend"],
                        "our_inpatient_471": ip[471], "our_inpatient_472": ip[472], "our_inpatient_473": ip[473],
                        "capture_factor_k": k, "anchored_473_benchmark_our_construct": a473, "implied_outpatient_share": share,
                        "pool_mean_473": s.loc[(473, "all episodes"), "benchmark"],
                        "pcf_gap_unadjusted": s.loc[(473, "inpatient posterior cervical"), "gap_unadjusted"],
                        "pcf_gap_risk_adjusted": s.loc[(473, "inpatient posterior cervical"), "gap_risk_adjusted"],
                        "acdf_ip_gap_unadjusted": s.loc[(473, "inpatient ACDF"), "gap_unadjusted"],
                        "acdf_ip_gap_risk_adjusted": s.loc[(473, "inpatient ACDF"), "gap_risk_adjusted"],
                        "acdf_op_gap_unadjusted": s.loc[(473, "outpatient ACDF"), "gap_unadjusted"],
                        "complex_center_gap": a.loc["complex spine center (posterior-heavy)", "gap_vs_target"],
                        "complex_center_dollars_per_100": a.loc["complex spine center (posterior-heavy)", "dollars_per_100_episodes"],
                        "outpatient_first_gap": a.loc["outpatient-first ACDF hospital", "gap_vs_target"],
                        "keep_inpatient_gap_at_35pct_migration": mg[(mg.regional_migration == 0.35) & (mg.hospital == "keeps posterior cervical inpatient")].gap_vs_target.iloc[0],
                        "regional_rate_gap_at_35pct_migration": mg[(mg.regional_migration == 0.35) & (mg.hospital == "migrates at regional rate")].gap_vs_target.iloc[0]})
    summary = pd.DataFrame(summary)
    summary.to_csv(I / "team_anchor_variants_summary.csv", index=False)
    pd.concat(statics).to_csv(I / "team_anchor_variants_static.csv", index=False)
    pd.concat(arches).to_csv(I / "team_anchor_variants_archetypes.csv", index=False)
    pd.concat(migs).to_csv(I / "team_anchor_variants_migration.csv", index=False)

    # primary-region outputs (same layout as team_unanchored_*; read by Figure 5 and the Supplement tables)
    st = pd.concat(statics); st = st[st.region == PRIMARY].drop(columns="region")
    st.insert(0, "op_share_473_pool", summary.set_index("region").loc[PRIMARY, "implied_outpatient_share"])
    st.to_csv(I / "team_anchored_static.csv", index=False)
    pd.concat(arches).query("region == @PRIMARY").drop(columns="region").to_csv(I / "team_anchored_archetypes.csv", index=False)
    pd.concat(migs).query("region == @PRIMARY").drop(columns="region").to_csv(I / "team_anchored_migration.csv", index=False)

    # sensitivity grid under the primary anchoring, plus the alternatives to the anchoring itself
    base_pac = dict(t2.PAC_PAY)
    variants = [("base (West South Central anchoring)", PRIMARY, {}),
                ("Middle Atlantic anchoring", "CENS_DIV_2", {}),
                ("outpatient covariates = all inpatient ACDF", PRIMARY, {"op_covariates": "all"}),
                ("outpatient home health rate 0%", PRIMARY, {"op_hh_rate": 0.0}),
                ("outpatient home health rate 15%", PRIMARY, {"op_hh_rate": 0.15}),
                ("IRF payment halved", PRIMARY, {"pac_pay": {**base_pac, "IRF": base_pac["IRF"] / 2}}),
                ("SNF payment x0.8 (30-day window)", PRIMARY, {"pac_pay": {**base_pac, "SNF": base_pac["SNF"] * 0.8}}),
                ("IRF discharges paid as SNF", PRIMARY, {"pac_pay": {**base_pac, "IRF": base_pac["SNF"]}})]
    sens = []
    for name, region, kw in variants:
        pl, k, a473, share, _ = run_anchored(region, cms, **kw)
        stv = t2.static(pl).set_index(["drg", "group"])
        arv, _, _ = t2.archetypes(pl); arv = arv.set_index("archetype")
        sens.append({"variant": name, "implied_outpatient_share": share, "capture_factor_k": k,
                     "drg473_inpatient_posterior_gap": stv.loc[(473, "inpatient posterior cervical"), "gap_unadjusted"],
                     "drg473_inpatient_posterior_gap_risk_adj": stv.loc[(473, "inpatient posterior cervical"), "gap_risk_adjusted"],
                     "drg473_inpatient_acdf_gap_risk_adj": stv.loc[(473, "inpatient ACDF"), "gap_risk_adjusted"],
                     "complex_center_gap": arv.loc["complex spine center (posterior-heavy)", "gap_vs_target"],
                     "outpatient_first_gap": arv.loc["outpatient-first ACDF hospital", "gap_vs_target"]})
    # the unanchored alternative: outpatient share taken from PSPS service counts
    pl, s_psps = t2.run()
    stv = t2.static(pl).set_index(["drg", "group"]); arv, _, _ = t2.archetypes(pl); arv = arv.set_index("archetype")
    sens.append({"variant": "outpatient share from PSPS service counts (unanchored)", "implied_outpatient_share": s_psps, "capture_factor_k": np.nan,
                 "drg473_inpatient_posterior_gap": stv.loc[(473, "inpatient posterior cervical"), "gap_unadjusted"],
                 "drg473_inpatient_posterior_gap_risk_adj": stv.loc[(473, "inpatient posterior cervical"), "gap_risk_adjusted"],
                 "drg473_inpatient_acdf_gap_risk_adj": stv.loc[(473, "inpatient ACDF"), "gap_risk_adjusted"],
                 "complex_center_gap": arv.loc["complex spine center (posterior-heavy)", "gap_vs_target"],
                 "outpatient_first_gap": arv.loc["outpatient-first ACDF hospital", "gap_vs_target"]})
    pool, *_ = run_anchored(PRIMARY, cms)
    for th in (6.0, 20.0):
        w = ts.expected_weights(pool, ["cervical_posterior"], 0.22, th)
        sens.append({"variant": f"migration selection theta={th:g} (22% regional)",
                     "pool_spending_ratio_473": (w * pool.spend)[pool.msdrg == 473].sum() / w[pool.msdrg == 473].sum() / pool[pool.msdrg == 473].spend.mean()})
    sens = pd.DataFrame(sens)
    sens.to_csv(I / "team_anchored_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.set_index("region").T.round(3).to_string())
    print(sens.round(3).to_string(index=False))
