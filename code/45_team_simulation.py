"""TEAM selection simulation: how outpatient migration after CY2026 Inpatient-Only (IPO) list removals could push
inpatient spinal fusion episodes over their target prices, and how much TEAM's pricing mechanics absorb.

Facts used (primary CMS documents: TEAM final rule, 42 CFR part 512; TEAM performance year 1 specifications):
- TEAM spinal fusion anchors: inpatient MS-DRGs 402, 426-430, 447-448, 450-451, 471-473; outpatient HCPCS 22551, 22554,
  22612, 22630, 22633 only. Posterior cervical (22600), anterior lumbar interbody (22558) and C1-C2 (22595) fusion,
  removed from the IPO list for CY2026, are NOT outpatient anchors: patients who move outpatient leave TEAM.
- Target prices: MS-DRG/HCPCS episode type x region benchmark from 3 baseline years (PY1: 2022-2024), prospective
  trend, 2% discount, risk adjustment by one-stage weighted linear regression per episode type (spinal fusion: 25
  V28 HCC flags, HCC count 0/1/2/3/4+, age group, prior post-acute care, beneficiary economic risk; hospital bed
  size, safety net), final normalization factor (capped 5%) and retrospective trend adjustment (capped 3%).

Data: Texas PUDF 2016-2019 Medicare FFS inpatients aged >=65 in MS-DRGs 471-473 (TEAM cervical episode types) and
453-460 (pre-FY2025 noncervical fusion DRGs, a proxy for TEAM's restructured noncervical DRGs).
Resource measures (relative, mean 1 within DRG):
  index   = total charges / mean charges in the same hospital-year-DRG cell (n>=5; else DRG-year), removing hospital
            markup differences
  episode = 0.75 x index + 0.25 x post-acute index (institutional post-acute discharge weight 1, home health 0.25)
TEAM risk model emulated with available adjusters: 25 TEAM spinal fusion V28 HCC flags from index-stay diagnoses
(CMS 2024 midyear/final mappings), HCC count (all V28 HCCs, no hierarchies) 0/1/2/3/4+, age group 65-74/75-84/85+.
Not available: prior post-acute care, economic risk, bed size/safety net (hospital constants).

Migration model: within an eligible procedure family, probability of moving outpatient p_i = logistic(a - theta*r_i),
where r_i is the percentile rank of a suitability score (HCC count, age group, present-on-admission diagnosis count,
APR severity; higher = less suitable) and a is solved so mean p = m. Expected post-migration means use weights 1-p.
theta is calibrated so that removing the national migration fraction of ACDF (cervical_anterior family) reproduces the
migration-attributable fall in the MS-DRG 473 share of cervical fusion discharges, net of coding-intensity drift in
the craniotomy control (MS-DRG 027).

Pricing scenarios per episode type:
  region migrates m (all hospitals): spending ratio R = C1/C0; measured-risk ratio Q = P1/P0
    target ratio = min(R, 1.03) [retrospective trend] x Q / min(Q, 1.05) [normalization cancels measured risk]
  hospital migrates m_h while region migrates m:  target_h = min(R_m, 1.03) x Q_h / min(Q_m, 1.05)
  over-target gap = R_h / target_h - 1 (before the 2% discount, which applies equally to all)
Outputs: data/interim/team_sim_calibration.csv, team_sim_results.csv, team_sim_summary.csv (aggregates only)
"""
import numpy as np
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
TEAM_SF_HCC = {17, 18, 37, 48, 93, 125, 126, 127, 155, 180, 181, 182, 192, 193, 199, 224, 226, 238, 249, 253, 254, 267, 326, 383, 401}
EPISODE_COST_PARTICIPANT = 42880  # Martin 2026 Spine J, adjusted mean 30-day inpatient fusion episode, TEAM participants
AGE = {16: "65-74", 17: "65-74", 25: "65-74", 18: "75-84", 19: "75-84", 26: "75-84", 20: "85+", 21: "85+"}
INST_PAC = {"03", "61", "62", "63", "64"}
HOME_HEALTH = {"06"}


def load():
    d = pd.read_parquet(I / "texas/tx_fusion_records.parquet")
    # fixed row order: parallel reads return rows in varying order, and the seeded draws downstream pick rows by position
    d = d.sort_values(list(d.columns), kind="mergesort", na_position="last").reset_index(drop=True)
    d = d[(d.payer == "ffs") & d.age_code.isin(AGE) & (d.charges > 0) & d.soi.between(1, 4)].copy()
    d["age_grp"] = d.age_code.map(AGE)
    m = pd.read_csv(ROOT / "data/raw/cms_hcc/mappings/2024 Midyear_Final ICD-10-CM Mappings.csv", skiprows=3, dtype=str, encoding="latin-1")
    m.columns = [c.replace("\n", " ") for c in m.columns]
    m = m.dropna(subset=["CMS-HCC Model Category V28"])
    code2hcc = m.groupby("Diagnosis Code")["CMS-HCC Model Category V28"].apply(lambda s: {int(float(x)) for x in s}).to_dict()
    hccs = d.dx.fillna("").map(lambda s: set().union(*[code2hcc.get(c.replace(".", ""), set()) for c in s.split("|") if c]) if s else set())
    d["hcc_count"] = hccs.map(len)
    d["hcc_cnt_cat"] = d.hcc_count.clip(upper=4)
    for h in sorted(TEAM_SF_HCC):
        d[f"hcc{h}"] = hccs.map(lambda s, h=h: int(h in s))
    # relative index cost within hospital-year-DRG cells
    cell = d.groupby(["hosp", "year", "msdrg"]).charges.transform("mean")
    ncell = d.groupby(["hosp", "year", "msdrg"]).charges.transform("size")
    drgyr = d.groupby(["year", "msdrg"]).charges.transform("mean")
    d["index_rel"] = np.where(ncell >= 5, d.charges / cell, d.charges / drgyr)
    d["pac"] = np.where(d.pat_status.isin(INST_PAC), 1.0, np.where(d.pat_status.isin(HOME_HEALTH), 0.25, 0.0))
    for c in ("index_rel", "pac"):
        d[c] = d[c] / d.groupby("msdrg")[c].transform("mean")
    d["episode_rel"] = 0.75 * d.index_rel + 0.25 * d.pac
    # suitability score (higher = less suitable for outpatient surgery)
    z = lambda s: (s - s.mean()) / s.std()
    d["suit"] = z(d.hcc_count) + z(d.age_grp.map({"65-74": 0, "75-84": 1, "85+": 2})) + z(d.n_poa.fillna(d.n_dx)) + z(d.soi)
    return d


def design(g):
    X = pd.get_dummies(g[["age_grp"]].astype(str), drop_first=True).astype(float)
    X = pd.concat([X, pd.get_dummies(g.hcc_cnt_cat.astype(int).astype(str), prefix="hcc_n", drop_first=True).astype(float)], axis=1)
    X = pd.concat([X, g[[f"hcc{h}" for h in sorted(TEAM_SF_HCC)]].astype(float)], axis=1)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "const", 1.0)
    return X


def fit_predict(g, y):
    X = design(g)
    beta, *_ = np.linalg.lstsq(X.values, g[y].values, rcond=None)
    return X.values @ beta


def migration_probs(rank, m, theta):
    if m <= 0:
        return np.zeros_like(rank)
    lo, hi = -50.0, 50.0
    for _ in range(80):
        a = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(a - theta * rank)))
        lo, hi = (a, hi) if p.mean() < m else (lo, a)
    return p


def expected_weights(g, families, m, theta):
    w = np.ones(len(g))
    for fam in families:
        idx = np.where(g.family.values == fam)[0]
        if len(idx) == 0:
            continue
        r = pd.Series(g.suit.values[idx]).rank(pct=True).values
        w[idx] = 1 - migration_probs(r, m, theta)
    return w


def calibrate(d):
    """theta such that removing m_acdf of ACDF inpatients lowers the MS-DRG 473 share by the migration-attributable amount."""
    tiers = pd.read_csv(I / "casemix_tiers_national.csv")
    sh = tiers[tiers.tier == "none"].pivot_table(index="year", columns="family", values="share")
    cerv, ctrl = sh["cervical_fusion"], sh["cranial_control_025_027"]
    counterfactual_2024 = cerv.loc[2016] * ctrl.loc[2024] / ctrl.loc[2016]
    target_drop = counterfactual_2024 - cerv.loc[2024]            # migration-attributable fall (share points / 100)
    mig = pd.read_csv(I / "setting_migration_cpt.csv", dtype={"cpt": str})
    a = mig[mig.cpt == "22551"].pivot_table(index="year", columns="setting", values="services", aggfunc="sum").fillna(0)
    tot = a.sum(axis=1)
    m_acdf = 1 - a.loc[2024, "inpatient"] / (a.loc[2016, "inpatient"] * tot.loc[2024] / tot.loc[2016])
    c = d[d.msdrg.isin([471, 472, 473])].reset_index(drop=True)
    base = (c.msdrg == 473).mean()
    rows = []
    for theta in np.round(np.arange(0, 20.01, 0.5), 2):
        w = expected_weights(c, ["cervical_anterior"], m_acdf, theta)
        share = (w * (c.msdrg == 473)).sum() / w.sum()
        rows.append({"theta": theta, "drg473_share_after": share, "drop": base - share})
    cal = pd.DataFrame(rows)
    feasible = cal[cal["drop"] >= target_drop]
    theta_star = float(feasible.theta.min()) if len(feasible) else float(cal.theta.max())
    cal["target_drop"], cal["m_acdf"], cal["baseline_share"], cal["theta_star"] = target_drop, m_acdf, base, theta_star
    return cal, theta_star, m_acdf, target_drop


def simulate(d, theta, outcome):
    rows = []
    groups = {"471": [471], "472": [472], "473": [473], "cervical 471-473": [471, 472, 473],
              "noncervical 459-460": [459, 460], "noncervical 453-455": [453, 454, 455]}
    for gname, drgs in groups.items():
        g = d[d.msdrg.isin(drgs)].reset_index(drop=True)
        if len(g) < 200:
            continue
        # risk model per episode type (DRG), then pooled predictions
        g["pred"] = np.nan
        for drg in drgs:
            ix = g.msdrg == drg
            g.loc[ix, "pred"] = fit_predict(g[ix], outcome)
        eligible = ["cervical_posterior"] if drgs[0] >= 471 else ["lumbar_anterior"]
        n_elig = int((g.family.isin(eligible)).sum())
        C0, P0 = g[outcome].mean(), g.pred.mean()
        for m in (0.10, 0.22, 0.35):
            wm = expected_weights(g, eligible, m, theta)
            Rm = (wm * g[outcome]).sum() / wm.sum() / C0
            Qm = (wm * g.pred).sum() / wm.sum() / P0
            for label, mh in [("no migration", 0.0), ("regional rate", m), ("twice regional", min(2 * m, 0.9))]:
                wh = expected_weights(g, eligible, mh, theta)
                Rh = (wh * g[outcome]).sum() / wh.sum() / C0
                Qh = (wh * g.pred).sum() / wh.sum() / P0
                target = min(Rm, 1.03) * Qh / min(Qm, 1.05)
                gap = Rh / target - 1
                rows.append({"outcome": outcome, "theta": theta, "episode_group": gname, "n_baseline": len(g), "n_eligible_family": n_elig,
                             "eligible_share": n_elig / len(g), "regional_migration": m, "hospital": label, "hospital_migration": mh,
                             "spending_ratio_hospital": Rh, "measured_risk_ratio_hospital": Qh, "spending_ratio_region": Rm,
                             "measured_risk_ratio_region": Qm, "target_ratio": target, "over_target_gap": gap,
                             "gap_dollars_per_episode": gap * EPISODE_COST_PARTICIPANT,
                             "risk_adjustment_capture": (Qh - 1) / (Rh - 1) if abs(Rh - 1) > 1e-9 else np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = load()
    print("records", len(d), "mean HCC count", round(d.hcc_count.mean(), 2), "TEAM HCC any", round(d[[f'hcc{h}' for h in TEAM_SF_HCC]].max(axis=1).mean(), 3))
    cal, theta_star, m_acdf, target_drop = calibrate(d)
    cal.to_csv(I / "team_sim_calibration.csv", index=False)
    print(f"ACDF migration fraction {m_acdf:.3f}; target DRG 473 share drop {target_drop * 100:.1f} pts; theta* = {theta_star}")
    res = pd.concat([simulate(d, th, oc) for oc in ("episode_rel", "index_rel") for th in sorted({0.0, theta_star, min(theta_star * 2, 20.0)})])
    res.to_csv(I / "team_sim_results.csv", index=False)
    pd.set_option("display.width", 260)
    key = res[(res.outcome == "episode_rel") & (res.theta == theta_star)]
    print(key[["episode_group", "n_baseline", "eligible_share", "regional_migration", "hospital", "spending_ratio_hospital", "measured_risk_ratio_hospital",
               "target_ratio", "over_target_gap", "gap_dollars_per_episode", "risk_adjustment_capture"]].round(4).to_string(index=False))
    summ = res.pivot_table(index=["outcome", "theta", "episode_group", "regional_migration"], columns="hospital", values="over_target_gap").reset_index()
    summ.to_csv(I / "team_sim_summary.csv", index=False)
