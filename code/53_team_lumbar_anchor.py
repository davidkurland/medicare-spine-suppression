"""TEAM noncervical (lumbar/thoracic) spinal fusion pricing, anchored to CMS's published PY1 prices.

Regroups Texas 2016-2019 Medicare FFS fusion records (pre-FY2025 MS-DRGs 453-455, 459-460) into the FY2025 episode
types from ICD-10-PCS fusion codes:
  multiple level  = any fusion code with a "2 or more" body part (thoracic 0RG7/0RG8, lumbar 0SG1), or single-joint
                    fusion codes in 2 or more distinct noncervical body parts (rule B, primary; rule A = "2 or more"
                    body parts only, sensitivity)
  combined A/P    = the pre-FY2025 grouper's own assignment (MS-DRGs 453-455), which encodes anterior plus posterior
                    column fusion
  402 single combined | 426/427/428 multiple combined (MCC/CC/none from 453/454/455) | 447/448 multiple (MCC/none
  from 459/460) | 450/451 single (MCC/none from 459/460). Pre-FY2025 456-458 (curvature/malignancy/infection) are outside TEAM.
  Cervical 471-473 records with both columns fused move to 429 (MCC) / 430; they are reported for information.
Outpatient anchors (TEAM Table 10): 22612 and 22630 -> DRG 451; 22633 -> DRG 402; CY2024 OPPS APC 5116 $17,774.76.
Inpatient spending = 2024 national average Medicare payment for the FY2025 DRG (geography file) + post-acute payment.
Anchoring as in 51_team_cms_anchor.py: capture factor k from lines with no outpatient component, then the outpatient share of each
anchored line that reproduces CMS's benchmark (West South Central primary; Middle Atlantic reported).
Outputs: data/interim/team_anchored_lumbar_summary.csv, team_anchored_lumbar_static.csv, team_anchored_lumbar_archetypes.csv,
         team_anchored_lumbar_regroup_check.csv
"""
import importlib
import re

import numpy as np
import pandas as pd

from common import ROOT

ts = importlib.import_module("45_team_simulation")
t2 = importlib.import_module("46_team_cervical_pricing")
a51 = importlib.import_module("51_team_cms_anchor")
I = ROOT / "data/interim"
OPPS_LUMBAR = 17774.76      # CY2024 OPPS Addendum B, APC 5116 (22612, 22630, 22633)
LUMBAR = [402, 426, 427, 428, 447, 448, 450, 451]
MULTI = re.compile(r"^(0RG7|0RG8|0SG1)")
NONCERV = re.compile(r"^(0RG[678A]|0SG[013])")
CERV = re.compile(r"^0RG[0124]")


def regroup(d, rule="B"):
    codes = d.pcs.fillna("").str.split("|")
    def feats(cs):
        f = [c for c in cs if re.match(r"^0[RS]G", c)]
        nc = [c for c in f if NONCERV.match(c)]
        cv = [c for c in f if CERV.match(c)]
        multi = any(MULTI.match(c) for c in nc) or (rule == "B" and len({c[:4] for c in nc}) >= 2)
        return pd.Series({
            "multi": multi,
            "cerv_combined": any(c[6] in "0J" for c in cv) and any(c[6] == "1" for c in cv),
            "n_noncerv": len(nc)})
    x = codes.apply(feats)
    d = pd.concat([d, x], axis=1)
    old = d.msdrg
    new = pd.Series(np.nan, index=d.index)
    tier3 = old.map({453: "MCC", 454: "CC", 455: "none"})
    tier2 = old.map({459: "MCC", 460: "none"})
    m = old.isin([453, 454, 455])
    new[m & d.multi] = tier3[m & d.multi].map({"MCC": 426, "CC": 427, "none": 428})
    new[m & ~d.multi] = 402
    m = old.isin([459, 460])
    new[m & d.multi] = tier2[m & d.multi].map({"MCC": 447, "none": 448})
    new[m & ~d.multi] = tier2[m & ~d.multi].map({"MCC": 450, "none": 451})
    m = old.isin([471, 472, 473])
    new[m & d.cerv_combined] = np.where(old[m & d.cerv_combined] == 471, 429, 430)
    new[m & ~d.cerv_combined] = old[m & ~d.cerv_combined]
    d["drg25"] = new
    return d


def spend_2024():
    g = pd.read_csv(I / "national_drg_totals.csv")
    return g[g.year == 2024].set_index("drg").Avg_Mdcr_Pymt_Amt


def lumbar_cohort(rule="B"):
    d = ts.load()
    d = d[d.msdrg.isin([453, 454, 455, 459, 460, 471, 472, 473])].reset_index(drop=True)
    d = regroup(d, rule)
    z = lambda s: (s - s.mean()) / s.std()
    d["tier"] = d.msdrg.map({453: 2, 454: 1, 455: 0, 459: 2, 460: 0, 471: 2, 472: 1, 473: 0})
    d["suit"] = z(d.hcc_count) + z(d.age_grp.map({"65-74": 0, "75-84": 1, "85+": 2})) + z(d.n_poa.fillna(d.n_dx)) + z(d.soi) + z(d.tier)
    pay = spend_2024()
    d["dest"] = t2.destination(d.pat_status.astype(str))
    d["pac_pay"] = d.dest.map(lambda x: t2.PAC_PAY.get(x) or 0.0)
    d["spend"] = d.drg25.map(pay) + d.pac_pay
    d["spend"] = d.spend + np.where(d.drg25.isin(LUMBAR), t2.ta.lumbar_inpatient(d), 0.0)
    d["setting"] = "inpatient"
    # analysis groups inside each lumbar line
    d["grp"] = np.select(
        [d.family == "lumbar_posterior", d.family == "lumbar_anterior", d.family == "thoracic_other"],
        ["inpatient posterior", "inpatient anterior (ALIF)", "inpatient thoracic"], "inpatient other")
    return d


def add_outpatient(d, drg, share, label, donors_family="lumbar_posterior"):
    ip = d[d.drg25 == drg]
    donors = ip[ip.family == donors_family]
    donors = donors[donors.suit <= donors.suit.median()]
    n_op = int(round(len(ip) * share / (1 - share)))
    op = donors.sample(n=n_op, replace=True, random_state=7).copy()
    rng = np.random.default_rng(11)
    op["pac_pay"] = np.where(rng.random(n_op) < t2.PAC_OP_HH_RATE, t2.PAC_PAY["HH"], 0.0)
    op["spend"] = OPPS_LUMBAR + op.pac_pay + t2.ta.lumbar_outpatient(drg)
    op["setting"], op["grp"] = "outpatient", label
    return pd.concat([d, op], ignore_index=True)


def risk_predict(pool, drgs):
    pool["pred"] = np.nan
    for drg in drgs:
        ix = pool.drg25 == drg
        if ix.sum() > 30:
            pool.loc[ix, "pred"] = ts.fit_predict(pool[ix], "spend")
    return pool


if __name__ == "__main__":
    cms = a51.cms_prices()
    d = lumbar_cohort()
    # regroup check against the national 2024 mix of the FY2025 noncervical DRGs
    g = pd.read_csv(I / "national_drg_totals.csv"); nat = g[(g.year == 2024) & g.drg.isin(LUMBAR)].set_index("drg").n
    tx = d[d.drg25.isin(LUMBAR)].drg25.value_counts()
    chk = pd.DataFrame({"texas_2016_2019": tx, "national_2024": nat}).fillna(0)
    chk["texas_share"] = chk.texas_2016_2019 / chk.texas_2016_2019.sum(); chk["national_share"] = chk.national_2024 / chk.national_2024.sum()
    chk.to_csv(I / "team_anchored_lumbar_regroup_check.csv")
    print("regroup check (share of noncervical TEAM lines):"); print((chk[["texas_share", "national_share"]] * 100).round(1).to_string())
    cerv_moved = d[d.msdrg.isin([471, 472, 473])].cerv_combined.mean()
    print(f"cervical records that would move to 429/430 (combined A/P): {cerv_moved:.1%}")

    lum = d[d.drg25.isin(LUMBAR)].copy()
    summary, statics, arches = [], [], []
    for region, desc in a51.REGIONS.items():
        ip = {drg: lum[lum.drg25 == drg].spend.mean() for drg in LUMBAR}
        k_single = np.mean([ip[x] / cms.loc[(region, x), "benchmark"] for x in (447, 448, 450)])
        k_comb = np.mean([ip[x] / cms.loc[(region, x), "benchmark"] for x in (426, 427, 428)])
        op_mean = OPPS_LUMBAR + t2.PAC_OP_HH_RATE * t2.PAC_PAY["HH"]
        anch451 = k_single * cms.loc[(region, 451), "benchmark"]; anch402 = k_comb * cms.loc[(region, 402), "benchmark"]
        s451 = float(np.clip((ip[451] - anch451) / (ip[451] - op_mean - t2.ta.lumbar_outpatient(451)), 0.02, 0.95))
        s402 = float(np.clip((ip[402] - anch402) / (ip[402] - op_mean - t2.ta.lumbar_outpatient(402)), 0.02, 0.95))
        pool = add_outpatient(lum, 451, s451, "outpatient posterolateral/PLIF (22612, 22630)")
        pool = add_outpatient(pool, 402, s402, "outpatient TLIF (22633)", donors_family="lumbar_posterior")
        pool = risk_predict(pool, LUMBAR)
        rows = []
        for drg in LUMBAR:
            gg = pool[pool.drg25 == drg]; bench = gg.spend.mean()
            for lab, s in [("all episodes", gg)] + [(lab, gg[gg.grp == lab]) for lab in gg.grp.unique()]:
                if len(s) <= 10: continue
                actual = s.spend.mean(); rat = bench * s.pred.mean() / gg.pred.mean()
                rows.append({"region": region, "drg": drg, "group": lab, "n": len(s), "benchmark": bench, "mean_spend": actual,
                             "gap_unadjusted": actual / bench - 1, "risk_adjusted_target": rat, "gap_risk_adjusted": actual / rat - 1})
        st = pd.DataFrame(rows); statics.append(st)
        bench = {drg: pool[pool.drg25 == drg].spend.mean() for drg in LUMBAR}
        predm = {drg: pool[pool.drg25 == drg].pred.mean() for drg in LUMBAR}
        # hospital archetypes per 100 noncervical episodes; mix keyed by (drg25, grp)
        counts = pool.groupby(["drg25", "grp"]).size()
        A = {"outpatient-first lumbar hospital": {(451, "outpatient posterolateral/PLIF (22612, 22630)"): 60, (451, "inpatient posterior"): 20,
                                                 (402, "outpatient TLIF (22633)"): 10, (402, "inpatient other"): 0, (448, "inpatient posterior"): 10},
             "regional-average mix": {k: v * 100 / counts.sum() for k, v in counts.items()},
             "inpatient single-level hospital": {(451, "inpatient posterior"): 70, (450, "inpatient posterior"): 10, (448, "inpatient posterior"): 20},
             "complex lumbar center (multilevel, combined, ALIF)": {(447, "inpatient posterior"): 15, (448, "inpatient posterior"): 25,
                                                                   (426, "inpatient posterior"): 10, (427, "inpatient posterior"): 15, (428, "inpatient posterior"): 10,
                                                                   (451, "inpatient anterior (ALIF)"): 10, (451, "inpatient posterior"): 15}}
        def result(mix):
            spend = target = 0.0
            for (drg, grp), n in mix.items():
                gg = pool[(pool.drg25 == drg) & (pool.grp == grp)]
                if n == 0 or len(gg) == 0 or np.isnan(gg.pred.mean()): continue
                spend += n * gg.spend.mean(); target += n * bench[drg] * (gg.pred.mean() / predm[drg])
            return spend / (target * (1 - t2.DISCOUNT)) - 1, spend - target * (1 - t2.DISCOUNT)
        ar = pd.DataFrame([{"region": region, "archetype": k, "gap_vs_target": result(v)[0], "dollars_per_100_episodes": result(v)[1]} for k, v in A.items()])
        arches.append(ar)
        S = st.set_index(["drg", "group"])
        def gap(drg, grp, c="gap_unadjusted"): return S.loc[(drg, grp), c] if (drg, grp) in S.index else np.nan
        summary.append({"region": region, "region_desc": desc, "cms_benchmark_451": cms.loc[(region, 451), "benchmark"], "cms_target_451": cms.loc[(region, 451), "target"],
                        "cms_benchmark_402": cms.loc[(region, 402), "benchmark"], "cms_target_402": cms.loc[(region, 402), "target"],
                        "capture_factor_single": k_single, "capture_factor_combined": k_comb,
                        "implied_op_share_451": s451, "implied_op_share_402": s402,
                        "op_plf_gap_451": gap(451, "outpatient posterolateral/PLIF (22612, 22630)"),
                        "ip_posterior_gap_451": gap(451, "inpatient posterior"), "ip_posterior_gap_451_risk_adj": gap(451, "inpatient posterior", "gap_risk_adjusted"),
                        "ip_alif_gap_451": gap(451, "inpatient anterior (ALIF)"), "ip_alif_gap_451_risk_adj": gap(451, "inpatient anterior (ALIF)", "gap_risk_adjusted"),
                        "op_tlif_gap_402": gap(402, "outpatient TLIF (22633)"), "ip_combined_gap_402": gap(402, "inpatient posterior"),
                        "ip_combined_gap_402_risk_adj": gap(402, "inpatient posterior", "gap_risk_adjusted"),
                        **{f"arch_{k}": v for k, v in ar.set_index("archetype").gap_vs_target.items()}})
    # sensitivity: level rule A ("2 or more" body parts only), primary region
    dA = lumbar_cohort("A"); lumA = dA[dA.drg25.isin(LUMBAR)].copy(); region = "CENS_DIV_7"
    ipA = {drg: lumA[lumA.drg25 == drg].spend.mean() for drg in LUMBAR}
    kA = np.mean([ipA[x] / cms.loc[(region, x), "benchmark"] for x in (447, 448, 450)]); kC = np.mean([ipA[x] / cms.loc[(region, x), "benchmark"] for x in (426, 427, 428)])
    op_mean = OPPS_LUMBAR + t2.PAC_OP_HH_RATE * t2.PAC_PAY["HH"]
    sA451 = float(np.clip((ipA[451] - kA * cms.loc[(region, 451), "benchmark"]) / (ipA[451] - op_mean), 0.02, 0.95))
    sA402 = float(np.clip((ipA[402] - kC * cms.loc[(region, 402), "benchmark"]) / (ipA[402] - op_mean), 0.02, 0.95))
    poolA = risk_predict(add_outpatient(add_outpatient(lumA, 451, sA451, "outpatient posterolateral/PLIF (22612, 22630)"), 402, sA402, "outpatient TLIF (22633)"), LUMBAR)
    def gapA(drg, grp, c="gap_unadjusted"):
        gg = poolA[poolA.drg25 == drg]; s = gg[gg.grp == grp]; b = gg.spend.mean()
        return s.spend.mean() / b - 1 if c == "gap_unadjusted" else s.spend.mean() / (b * s.pred.mean() / gg.pred.mean()) - 1
    summary.append({"region": "CENS_DIV_7 (level rule A)", "region_desc": "sensitivity: multiple level = 2-or-more body parts only",
                    "capture_factor_single": kA, "capture_factor_combined": kC, "implied_op_share_451": sA451, "implied_op_share_402": sA402,
                    "op_plf_gap_451": gapA(451, "outpatient posterolateral/PLIF (22612, 22630)"), "ip_posterior_gap_451": gapA(451, "inpatient posterior"),
                    "ip_posterior_gap_451_risk_adj": gapA(451, "inpatient posterior", "risk"), "op_tlif_gap_402": gapA(402, "outpatient TLIF (22633)"),
                    "ip_combined_gap_402": gapA(402, "inpatient posterior"), "ip_combined_gap_402_risk_adj": gapA(402, "inpatient posterior", "risk"),
                    "texas_single_level_share": (~dA.multi[dA.msdrg.isin([453, 454, 455, 459, 460])]).mean()})
    pd.DataFrame(summary).to_csv(I / "team_anchored_lumbar_summary.csv", index=False)
    pd.concat(statics).to_csv(I / "team_anchored_lumbar_static.csv", index=False)
    pd.concat(arches).to_csv(I / "team_anchored_lumbar_archetypes.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).set_index("region").T.round(3).to_string())
    print(pd.concat(statics).query("region == 'CENS_DIV_7' and drg in [451, 402]").round(3).to_string(index=False))
