"""Robustness of the anchored TEAM cervical simulation to the substrate: region, period, and national case mix.

The simulation rests on Texas Medicare FFS records from 2016-2019 anchored to CMS's published PY1 price for one
census division. Three things could make that substrate drive the result rather than the pricing structure:
  1. Region. We re-anchor to each of the 9 census divisions CMS published, not just West South Central.
  2. Period. We re-run on 2016-2017 and on 2018-2019 separately.
  3. National case mix. Cervical fusion severity has shifted nationally since 2016 as uncomplicated cases moved
     outpatient (DRG 473 fell from 47% of cervical fusion discharges in 2017 to 25% in 2024). We reweight the
     Texas cohort to the national MS-DRG 471/472/473 mix for 2019 and for 2024 and re-run.
The reported quantity is the risk-adjusted gap for inpatient posterior cervical fusion, the paper's headline
TEAM number, plus the implied outpatient share and the complex-center archetype.

Outputs: data/interim/team_anchored_robustness.csv, data/interim/team_anchored_casemix_representativeness.csv,
         manuscript/figures/eFigure_team_robustness.png/.pdf
"""
import importlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT

t2 = importlib.import_module("46_team_cervical_pricing")
an = importlib.import_module("51_team_cms_anchor")
hs = importlib.import_module("52_hidden_share_team_episodes")
I = ROOT / "data/interim"
FIG = ROOT / "manuscript/figures"
DRGS = (471, 472, 473)
DIV = {"CENS_DIV_1": "New England", "CENS_DIV_2": "Middle Atlantic", "CENS_DIV_3": "East North Central",
       "CENS_DIV_4": "West North Central", "CENS_DIV_5": "South Atlantic", "CENS_DIV_6": "East South Central",
       "CENS_DIV_7": "West South Central", "CENS_DIV_8": "Mountain", "CENS_DIV_9": "Pacific"}
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight"})


def national_mix(tot, year):
    v = {d: float(tot.get((year, d), 0.0)) for d in DRGS}
    s = sum(v.values())
    return {d: v[d] / s for d in DRGS}


def reweight(d, target, seed=3):
    """Resample within MS-DRG so the between-DRG mix matches `target`, keeping the cohort size."""
    n, rng, out = len(d), np.random.default_rng(seed), []
    for drg, share in target.items():
        g = d[d.msdrg == drg]
        out.append(g.sample(n=max(int(round(n * share)), 1), replace=True, random_state=int(rng.integers(1e6))))
    return pd.concat(out, ignore_index=True)


def run_on(d, region, cms):
    """The anchoring pipeline on a supplied cohort (51_team_cms_anchor.run_anchored builds its own)."""
    k, a473, share, ip = an.anchor(d, cms, region)
    pool = t2.risk_predict(t2.pooled_473(d, share))
    st = t2.static(pool).set_index(["drg", "group"])
    ar, _, _ = t2.archetypes(pool)
    ar = ar.set_index("archetype")
    return dict(capture_factor_k=k, implied_outpatient_share=share,
                pcf_gap_unadjusted=st.loc[(473, "inpatient posterior cervical"), "gap_unadjusted"],
                pcf_gap_risk_adjusted=st.loc[(473, "inpatient posterior cervical"), "gap_risk_adjusted"],
                op_gap=st.loc[(473, "outpatient ACDF"), "gap_unadjusted"],
                complex_center_gap=ar.loc["complex spine center (posterior-heavy)", "gap_vs_target"],
                outpatient_first_gap=ar.loc["outpatient-first ACDF hospital", "gap_vs_target"], n_episodes=len(pool))


if __name__ == "__main__":
    cms = an.cms_prices()
    base = t2.cohort()
    tot = hs.complete()
    rows = []

    for r, name in DIV.items():                                        # 1. every census division CMS published
        rows.append(dict(family="Census division", spec=name, primary=(r == an.PRIMARY),
                         cms_benchmark_473=cms.loc[(r, 473), "benchmark"], **run_on(base, r, cms)))
    for lab, yrs in [("2016-2017", (2016, 2017)), ("2018-2019", (2018, 2019))]:   # 2. period
        rows.append(dict(family="Period", spec=lab, primary=False, cms_benchmark_473=cms.loc[(an.PRIMARY, 473), "benchmark"],
                         **run_on(base[base.year.isin(yrs)].reset_index(drop=True), an.PRIMARY, cms)))
    for y in (2019, 2024):                                             # 3. national case mix
        rows.append(dict(family="National case mix", spec=f"reweighted to {y} national MS-DRG mix", primary=False,
                         cms_benchmark_473=cms.loc[(an.PRIMARY, 473), "benchmark"],
                         **run_on(reweight(base, national_mix(tot, y)), an.PRIMARY, cms)))
    rob = pd.DataFrame(rows)
    rob.to_csv(I / "team_anchored_robustness.csv", index=False)

    tx = base.groupby("msdrg").size()
    tx = {d: float(tx.get(d, 0)) / float(tx.sum()) for d in DRGS}
    rep = pd.DataFrame([dict(source="Texas records, 2016-2019 (our cohort)", **{f"drg_{d}": tx[d] for d in DRGS}),
                        dict(source="National FFS, 2017", **{f"drg_{d}": national_mix(tot, 2017)[d] for d in DRGS}),
                        dict(source="National FFS, 2019", **{f"drg_{d}": national_mix(tot, 2019)[d] for d in DRGS}),
                        dict(source="National FFS, 2024", **{f"drg_{d}": national_mix(tot, 2024)[d] for d in DRGS})])
    rep.to_csv(I / "team_anchored_casemix_representativeness.csv", index=False)

    # eFigure: the headline TEAM number under every substrate choice
    SPINE, GRAY, DARK = "#D55E00", "#8C8C8C", "#222222"
    rob["label"] = rob.spec.str.replace("reweighted to ", "", regex=False)
    rob["fam_short"] = rob.family.map({"Census division": "Census division", "Period": "Period", "National case mix": "Case mix"})
    fig, ax = plt.subplots(figsize=(6.9, 4.4))
    fig.subplots_adjust(left=0.40)
    rob = rob.iloc[::-1].reset_index(drop=True)
    base_val = rob[rob.primary].pcf_gap_risk_adjusted.iloc[0] * 100
    lo, hi = rob.pcf_gap_risk_adjusted.min() * 100, rob.pcf_gap_risk_adjusted.max() * 100
    ax.axvspan(lo, hi, color=SPINE, alpha=0.10, lw=0, zorder=0)
    ax.axvline(base_val, color=SPINE, lw=1.2, ls="--", zorder=1)
    for i, r in rob.iterrows():
        ax.plot(r.pcf_gap_risk_adjusted * 100, i, "o", ms=7.5 if r.primary else 5.5,
                color=SPINE if r.primary else DARK, zorder=3)
        left = r.pcf_gap_risk_adjusted * 100 < base_val - 0.5        # keep the label clear of the dashed primary line
        ax.annotate(f"{r.pcf_gap_risk_adjusted * 100:.0f}%", (r.pcf_gap_risk_adjusted * 100, i),
                    xytext=(-8 if left else 8, 0), ha="right" if left else "left", textcoords="offset points", va="center", fontsize=6.8,
                    color=SPINE if r.primary else DARK, fontweight="bold" if r.primary else "normal")
    ax.axvline(0, color=DARK, lw=0.8)
    ax.set_yticks(range(len(rob)))
    ax.set_yticklabels([f"{s}  (primary)" if p else s for s, p in zip(rob.label, rob.primary)])
    for t, p in zip(ax.get_yticklabels(), rob.primary):
        if p:
            t.set_fontweight("bold"); t.set_color(SPINE)
    prev = None
    for i, f in enumerate(rob.family):
        if f != prev and i:
            ax.axhline(i - 0.5, color="#DCDCDC", lw=0.7)
        prev = f
    for f in rob.fam_short.unique():
        ix = rob.index[rob.fam_short == f]
        ax.text(-0.52, (ix.min() + ix.max()) / 2, f, transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=7.2, fontweight="bold", color=GRAY)
    ax.set_xlim(-1.5, 36)
    ax.set_ylim(-0.7, len(rob) - 0.3)
    ax.set_xlabel("Inpatient posterior cervical fusion vs its risk-adjusted TEAM target, %")
    ax.text(hi + 0.6, len(rob) - 0.45, f"full range {lo:.0f}% to {hi:.0f}%", ha="left", va="bottom",
            fontsize=6.8, color=SPINE, fontweight="bold")
    fig.savefig(FIG / "eFigure_team_robustness.png")
    fig.savefig(FIG / "eFigure_team_robustness.pdf")
    plt.close(fig)

    print(rob.iloc[::-1][["family", "spec", "capture_factor_k", "implied_outpatient_share",
                          "pcf_gap_risk_adjusted", "complex_center_gap"]].round(3).to_string(index=False))
    print("\nCASE MIX\n", rep.round(3).to_string(index=False))
