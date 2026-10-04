"""Figure 5. TEAM pricing of cervical spinal fusion: pooled episode types penalize inpatient and posterior surgery.

A  Every price line with an outpatient anchor (cervical DRG 473, lumbar DRGs 451 and 402): mean 30-day Medicare spending
   by setting and procedure vs the pooled price.
B  Gap vs target before -> after TEAM risk adjustment (dumbbells) with bootstrap interval, inpatient posterior and
   anterior cervical fusion (DRG 473) and inpatient posterior lumbar fusion (DRGs 451, 402).
C  Hospital archetypes, cervical and lumbar: performance vs target including the 2% discount. Stop-loss limits apply to a
   hospital's aggregate target across episode categories, so they are not drawn.
D  One-way sensitivity of the risk-adjusted posterior cervical gap (82_uncertainty_sensitivity.py): range within each
   family of assumptions, with the hospital-cluster bootstrap interval.
The CY2026 migration projection is a Supplement figure (eFigure_team_migration).
Output: manuscript/figures/Figure5_team.png/.pdf, eFigure_team_migration.png/.pdf
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

ETAB_SPEND = "eTables 35 and 36"   # Supplement tables of cervical spending by setting and procedure
I = ROOT / "data/interim"
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)
SPINE, BLUE, GRAY, DARK, GREEN = "#D55E00", "#0072B2", "#8C8C8C", "#222222", "#009E73"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 9, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight"})


def panel_letter(fig, ax, L, fallback=-0.10):
    """Bold panel letter on the title line. Left-aligned to the y-axis label when the panel has one; otherwise
    centered over the y tick labels; otherwise a fixed offset left of the plot."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pos = ax.get_position()
    lab = ax.yaxis.label
    if lab.get_text():
        x = inv.transform((lab.get_window_extent(r).x0, 0))[0]; ha = "left"
    else:
        bb = ax.yaxis.get_tightbbox(r)
        if bb is not None and bb.width > 12:
            x = inv.transform(((bb.x0 + bb.x1) / 2, 0))[0]; ha = "center"
        else:
            x = pos.x0 + fallback * pos.width; ha = "left"
    tb = ax._left_title.get_window_extent(r)
    y = inv.transform((0, tb.y0))[1] + 1.5 / 72 / fig.get_figheight()
    fig.text(x, y, L, fontsize=10, fontweight="bold", va="bottom", ha=ha)


PANELS = []


def head(ax, L, title, sub):
    n = sub.count("\n") + 1
    ax.set_title(title, loc="left", fontweight="bold", pad=6 + 9 * n)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=6.8, color="#555555", va="bottom", linespacing=1.25)
    PANELS.append((ax, L))


def pct(v):
    return f"{v:+.0f}%".replace("-", "−")


if __name__ == "__main__":
    st = pd.read_csv(I / "team_anchored_static.csv")
    ar = pd.read_csv(I / "team_anchored_archetypes.csv")
    mg = pd.read_csv(I / "team_anchored_migration.csv")
    ls = pd.read_csv(I / "team_anchored_lumbar_static.csv"); ls = ls[ls.region == "CENS_DIV_7"]
    la = pd.read_csv(I / "team_anchored_lumbar_archetypes.csv"); la = la[la.region == "CENS_DIV_7"].set_index("archetype")
    fig = plt.figure(figsize=(7.4, 7.4))
    gs_top = fig.add_gridspec(1, 1, top=0.945, bottom=0.63, left=0.085, right=0.975)
    gs = fig.add_gridspec(1, 3, top=0.485, bottom=0.07, left=0.175, right=0.975, wspace=0.78, width_ratios=[0.8, 1.35, 0.95])
    bt = pd.read_csv(I / "team_bootstrap.csv").set_index("metric")
    tn = pd.read_csv(I / "team_tornado.csv")

    # A: every price line with an outpatient anchor
    axA = ax = fig.add_subplot(gs_top[0, 0])
    c473 = st[st.drg == 473].set_index("group"); l451 = ls[ls.drg == 451].set_index("group"); l402 = ls[ls.drg == 402].set_index("group")
    groups = [("Cervical fusion\nMS-DRG 473", c473, [("outpatient ACDF", "Outpatient\nACDF", GREEN), ("inpatient ACDF", "Inpatient\nACDF", BLUE),
                                            ("inpatient posterior cervical", "Inpatient\nposterior", SPINE)]),
              ("Lumbar single-level fusion\nMS-DRG 451", l451, [("outpatient posterolateral/PLIF (22612, 22630)", "Outpatient\nPLF/PLIF", GREEN),
                                                      ("inpatient posterior", "Inpatient\nposterior", SPINE), ("inpatient anterior (ALIF)", "Inpatient\nALIF", BLUE)]),
              ("Lumbar single-level combined fusion\nMS-DRG 402", l402, [("outpatient TLIF (22633)", "Outpatient\nTLIF", GREEN), ("inpatient posterior", "Inpatient\nposterior", SPINE)])]
    x = 0; ticks, labels = [], []
    for title, g, bars in groups:
        bench = g.loc["all episodes", "benchmark"] / 1000; x0 = x
        for key, lab, c in bars:
            v = g.loc[key, "mean_spend"] / 1000
            ax.bar(x, v, color=c, width=0.7)
            ylab = bench + 0.7 if -4.2 < v - bench < 0 else v + 0.7   # keep a label clear of the dashed price line
            ax.text(x, ylab, pct(g.loc[key, "gap_unadjusted"] * 100), ha="center", va="bottom", fontsize=7, color=c, fontweight="bold")
            ax.text(x, 0.8, f"${v:.1f}k", ha="center", va="bottom", fontsize=6.3, color="white", fontweight="bold")
            ticks.append(x); labels.append(lab); x += 1
        ax.plot([x0 - 0.45, x - 0.55], [bench, bench], color=DARK, ls="--", lw=1)
        ax.text((x0 + x - 1) / 2, 49.5, f"{title}\npooled price ${bench:.1f}k", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=DARK)
        x += 1.2
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_xlim(-0.6, x - 1.2 - 0.4)
    ax.set_ylim(0, 58); ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("Mean 30-day spending, $ thousands")
    head(ax, "A", "Episode spending by setting and procedure on each line with an outpatient anchor", "Spending vs the pooled price (dashed)")

    # B: before -> after risk adjustment for the inpatient groups on each anchored line
    axB = ax = fig.add_subplot(gs[0, 0])
    rowsB = [("Cervical 473\nposterior fusion", c473.loc["inpatient posterior cervical"], SPINE, "pcf_gap_risk_adjusted"),
             ("Cervical 473\nACDF", c473.loc["inpatient ACDF"], BLUE, "ip_acdf_gap_risk_adjusted"),
             ("Lumbar 451\nposterior fusion", l451.loc["inpatient posterior"], SPINE, "lumbar451_ip_gap_risk_adjusted"),
             ("Lumbar 402\nposterior fusion", l402.loc["inpatient posterior"], SPINE, "lumbar402_ip_gap_risk_adjusted")]
    for i, (lab, r, c, key) in enumerate(rowsB):
        u, a = r.gap_unadjusted * 100, r.gap_risk_adjusted * 100
        ax.plot([bt.loc[key, "lo"] * 100, bt.loc[key, "hi"] * 100], [i + 0.26, i + 0.26], color=c, lw=0.9, solid_capstyle="butt")
        for e in ("lo", "hi"):
            ax.plot([bt.loc[key, e] * 100] * 2, [i + 0.20, i + 0.32], color=c, lw=0.9)
        if abs(u - a) > 0.8:
            ax.annotate("", xy=(a, i), xytext=(u, i), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, shrinkA=0, shrinkB=0, mutation_scale=8))
        else:
            ax.plot([a], [i], marker=">", ms=4, color=c)
        ax.plot(u, i, "o", ms=4.5, mfc="white", mec=c, mew=1.2)
        ax.text(max(u, a) + 4.5, i, f"{pct(u)} to {pct(a)}", va="center", ha="left", fontsize=6.4, color=c, fontweight="bold")
    ax.set_yticks(range(len(rowsB))); ax.set_yticklabels([r[0] for r in rowsB], fontsize=6.5)
    ax.set_ylim(len(rowsB) - 0.5, -0.6); ax.set_xlim(0, 78); ax.set_xticks([0, 20, 40, 60]); ax.spines["bottom"].set_bounds(0, 60)
    ax.set_xlabel("Inpatient spending vs target, %")
    head(ax, "B", "Gap before and after\nrisk adjustment", "○ before, arrowhead after; bar, 95% CI\nafter; all groups in " + ETAB_SPEND)

    # C: hospital archetypes, cervical and lumbar
    ax = fig.add_subplot(gs[0, 1])
    a = ar.set_index("archetype")
    rows = [("outpatient-first ACDF hospital", "Outpatient-first\nACDF", a), ("outpatient-first lumbar hospital", "Outpatient-first\nlumbar", la),
            ("regional-average mix", "Regional\naverage mix", a), ("inpatient ACDF hospital", "Inpatient\nACDF", a),
            ("inpatient single-level hospital", "Inpatient\nsingle-level lumbar", la),
            ("complex lumbar center (multilevel, combined, ALIF)", "Complex lumbar\ncenter (75% multilevel)", la),
            ("complex spine center (posterior-heavy)", "Complex cervical\ncenter (posterior)", a)]
    vals = [src.loc[k, "gap_vs_target"] * 100 for k, _, src in rows]
    ax.barh(range(len(rows)), vals, color=[GREEN if v < 0 else SPINE for v in vals], height=0.62)
    for i, (k, _, src) in enumerate(rows):
        v = vals[i]; dol = src.loc[k, "dollars_per_100_episodes"]
        ax.text(max(v, 0) + 0.6, i, f"{v:+.1f}% ({'+' if dol >= 0 else '−'}${abs(dol) / 1000:,.0f}k)".replace("-", "−"), va="center", ha="left", fontsize=6.3)
    ax.axvline(0, color=DARK, lw=0.6)   # stop-loss limits apply to a hospital's aggregate target across all episode categories, so none are drawn
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=6.2)
    ax.set_ylim(len(rows) - 0.4, -0.6); ax.set_xlim(-18, 30)
    ax.set_xlabel("Spending vs target, % (per 100 episodes)")
    head(ax, "C", "Simulated hospital\nperformance by case mix", "By procedure mix and setting,\nwith the 2% discount")

    # D: one-way sensitivity of the risk-adjusted posterior cervical gap
    ax = fig.add_subplot(gs[0, 2])
    base = tn[tn.family == "Base"].pcf_gap_risk_adjusted.iloc[0] * 100
    # post-acute rows repeated with the outpatient share held at the primary implied share (92_team_fixed_share_sensitivity.py)
    fixed_share = pd.read_csv(I / "team_fixed_share.csv")
    fx = fixed_share[(fixed_share.family == "Post-acute payment") & fixed_share.spec.str.endswith("primary implied share")].pcf_gap_risk_adjusted * 100
    fam = lambda f, drop=(): tn[(tn.family == f) & ~tn.spec.isin(drop)].pcf_gap_risk_adjusted * 100
    bars = [("Capture factor\n±10%", fam("Capture factor")), ("Capture factor\nrule", fam("Capture factor rule")),
            ("Outpatient share\n20% to 68%", fam("Outpatient share of pool")), ("Professional\nfees", fam("Professional fees")),
            ("Readmissions", fam("Readmissions")), ("Post-acute\npayment", pd.concat([fam("Post-acute payment"), fx])),
            ("Outpatient case\nassumptions", fam("Outpatient episodes")), ("Census\ndivision", fam("Census division")),
            ("Study period", fam("Period")), ("Comorbidity\nsignal ×2", fam("Comorbidity capture")[:1])]
    bars = [(lab, min(v.min(), base), max(v.max(), base)) for lab, v in bars]
    bars.sort(key=lambda t: t[1] - t[2])
    bars.append(("Bootstrap\n95% CI", bt.loc["pcf_gap_risk_adjusted", "lo"] * 100, bt.loc["pcf_gap_risk_adjusted", "hi"] * 100))
    for i, (lab, lo, hi) in enumerate(bars):
        c = GRAY if lab.startswith("Bootstrap") else SPINE
        ax.barh(i, hi - lo, left=lo, color=c, height=0.5, alpha=0.85)
        ax.text(lo - 0.8, i, f"{lo:.0f}", va="center", ha="right", fontsize=5.8, color="#555555")
        ax.text(hi + 0.8, i, f"{hi:.0f}", va="center", ha="left", fontsize=5.8, color="#555555")
    ax.axvline(base, color=DARK, lw=0.9, ls="--")
    ax.text(base + 0.6, len(bars) - 0.45, f"Primary {base:.0f}%", fontsize=5.8, ha="left", va="bottom", color=DARK)
    ax.set_yticks(range(len(bars))); ax.set_yticklabels([b_[0] for b_ in bars], fontsize=5.6)
    ax.set_ylim(len(bars) + 0.25, -0.6); ax.set_xlim(12, 50); ax.set_xticks([20, 30, 40])
    ax.set_xlabel("Posterior cervical fusion\nvs risk-adjusted target, %")
    head(ax, "D", "One-way sensitivity,\ncervical 473 posterior", "Range within each family\nof assumptions")

    # align the bottom grid's leftmost ink (B's tick labels) with A's leftmost ink (its y-axis label), measured after layout
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    a_left = inv.transform((axA.yaxis.label.get_window_extent(r).x0, 0))[0]
    b_left = inv.transform((axB.yaxis.get_tightbbox(r).x0, 0))[0]
    gs.update(left=gs.left + (a_left - b_left))
    for a_, L in PANELS:
        panel_letter(fig, a_, L)
    fig.savefig(OUT / "Figure5_team.png"); fig.savefig(OUT / "Figure5_team.pdf")
    plt.close(fig)

    # Supplement: projected performance as posterior cervical fusion moves outpatient
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for h, lab, c, ls_ in [("keeps posterior cervical inpatient", "Keeps them inpatient", SPINE, "-"),
                           ("migrates at regional rate", "Migrates at the regional rate", GRAY, "--"),
                           ("migrates at twice regional rate", "Migrates at twice the regional rate", GREEN, "-")]:
        sub = mg[mg.hospital == h].sort_values("regional_migration")
        ax.plot(sub.regional_migration * 100, sub.gap_vs_target * 100, color=c, ls=ls_, marker="o", ms=3)
        ax.text(sub.regional_migration.iloc[-1] * 100 + 1.2, sub.gap_vs_target.iloc[-1] * 100, lab, color=c, fontsize=6.5, va="center")
    ax.set_xlabel("Posterior cervical fusions moving outpatient in the region, %")
    ax.set_ylabel("Projected spending vs target, %")
    ax.set_xticks([10, 22, 35]); ax.set_xlim(7, 62); ax.set_ylim(0, 4); ax.spines["bottom"].set_bounds(10, 35)
    fig.savefig(OUT / "eFigure_team_migration.png"); fig.savefig(OUT / "eFigure_team_migration.pdf")
    print("saved")
