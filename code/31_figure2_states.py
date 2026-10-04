"""Figure 2: what state all-payer data see that the CMS public FFS file does not.

A  Medicare Advantage share of Medicare neurosurgical inpatients in state data (Indiana 2021-2024,
   Florida 2023-2025, Texas 2016-2019) vs national MA share of Part A enrollees.
B  CMS visibility: CMS-published FFS discharges as a share of all Medicare inpatients recorded in state data
   (New York 2013-2024, where SPARCS "Medicare" includes MA; Indiana; Florida; Texas).
C  Multistate (HCUPnet): all-payer Medicare spine fusion (CCSR MST013) vs CMS FFS fusion DRGs, complete and
   published, summed over states with continuous data, indexed to 2016.
D  Per-state change 2016-2023, HCUPnet all-Medicare count vs CMS published FFS count, for spine fusion, joint
   replacement (MS-DRGs 469-470) and cranial (MS-DRGs 23 and 25-27).
Output: manuscript/figures/Figure2_states.png/.pdf
"""
import importlib
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)
SPINE, GRAY, DARK, LIGHT = "#D55E00", "#8C8C8C", "#222222", "#C8C8C8"
STATE_COL = {"New York": "#000000", "Indiana": "#009E73", "Florida": "#CC79A7", "Texas": "#E69F00"}
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 9, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight"})


def letter(ax, s):
    ax.text(-0.2, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def state_series():
    """Returns long frame: state, group, year, ma_share, visible_share."""
    rows = []
    ind = pd.read_csv(I / "IN_medicare_decomposition.csv")
    for r in ind.itertuples():
        rows.append(("Indiana", r.grp, r.year, r.advantage_share, r.cms_reported_share))
    fl = pd.read_csv(I / "florida/fl_inpatient_by_payer.csv")
    flw = fl.pivot_table(index=["group", "year"], columns="payer", values="discharges", aggfunc="sum")
    st = pd.read_csv(I / "suppression_exact_state.csv")
    for (g, y), r in flw.iterrows():
        tot = r["Medicare"] + r["Medicare Managed Care"]
        pub = st[(st.st == "FL") & (st.group == g) & (st.year == y)].provider_published
        rows.append(("Florida", g, y, r["Medicare Managed Care"] / tot, pub.iloc[0] / tot if len(pub) else np.nan))
    tx = pd.read_csv(I / "texas/tx_drg_groups_by_payer.csv")
    for r in tx[tx.year >= 2016].itertuples():
        rows.append(("Texas", r.group, r.year, r.ma_share, r.visible_share_of_medicare))
    ny = importlib.import_module("12_ny_payer_series").ny_payer_series()
    cms = pd.read_parquet(I / "cms_neuro_hospital_year.parquet")
    cms = cms[(cms.state == "NY") & cms.group.isin(["spine", "cranial"])].groupby(["group", "year"]).discharges.sum()
    for r in ny[ny.year.between(2013, 2024)].itertuples():
        rows.append(("New York", r.grp, r.year, np.nan, cms.get((r.grp, r.year), np.nan) / r.medicare))
    return pd.DataFrame(rows, columns=["state", "group", "year", "ma_share", "visible_share"])


if __name__ == "__main__":
    s = state_series()
    s.to_csv(I / "state_visibility_series.csv", index=False)
    ns = pd.read_csv(I / "national_sensitivity.csv")
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ns.year, ns.ma_share_us * 100, color=LIGHT, lw=3, label="US MA share of Part A enrollees")
    for stt in ["Texas", "Indiana", "Florida"]:
        for g, ls in [("spine", "-"), ("cranial", "--")]:
            d = s[(s.state == stt) & (s.group == g)].sort_values("year")
            ax.plot(d.year, d.ma_share * 100, color=STATE_COL[stt], ls=ls, marker="o", ms=2.5,
                    label=f"{stt}, {g}" if g == "spine" else None)
        d = s[(s.state == stt) & (s.group == "spine")].sort_values("year")
        ax.text(d.year.iloc[-1] + 0.3, d.ma_share.iloc[-1] * 100, stt, fontsize=6.5, color=STATE_COL[stt], va="center")
    ax.set_ylim(0, 65)
    ax.set_xlim(2012.6, 2026.5)
    ax.set_xticks([2013, 2016, 2019, 2022, 2025])
    ax.set_ylabel("Medicare Advantage share, %")
    ax.set_title("Medicare inpatients in Advantage plans", loc="left", fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(frameon=False, loc="upper left", handles=[ax.get_lines()[0], Line2D([], [], color=DARK, ls="-", label="Spine"),
                                                        Line2D([], [], color=DARK, ls="--", label="Cranial")])
    letter(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ends = []
    for stt in ["New York", "Texas", "Indiana", "Florida"]:
        # New York cranial omitted: SPARCS APR-DRGs 020-022 exclude endovascular procedures that MS-DRGs 023-027 include
        for g, ls in ([("spine", "-")] if stt == "New York" else [("spine", "-"), ("cranial", "--")]):
            d = s[(s.state == stt) & (s.group == g)].sort_values("year")
            ax.plot(d.year, (1 - d.visible_share) * 100, color=STATE_COL[stt], ls=ls, marker="o", ms=2.5)
        d = s[(s.state == stt) & (s.group == "spine")].sort_values("year").dropna(subset=["visible_share"])
        ends.append([d.year.iloc[-1] + 0.3, (1 - d.visible_share.iloc[-1]) * 100, stt, (1 - d.visible_share.iloc[-1]) * 100])
    ends.sort(key=lambda e: e[1])       # pull the 2024 endpoints apart so the 4 state names stay legible
    for j in range(1, len(ends)):
        ends[j][1] = max(ends[j][1], ends[j - 1][1] + 4.5)
    for xe, ye, stt, y_true in ends:
        ax.text(xe + 0.9, ye, stt, fontsize=6.5, color=STATE_COL[stt], va="center")
        ax.plot([xe - 0.1, xe + 0.75], [y_true, ye], color=STATE_COL[stt], lw=0.45, clip_on=False)   # leader to the spine end point
    ax.set_ylim(0, 100)
    ax.set_xlim(2012.6, 2026.5)
    ax.set_xticks([2013, 2016, 2019, 2022, 2025])
    ax.set_ylabel("Medicare inpatients in state data\nwith no published CMS row, %")
    ax.set_title("Share of invisible Medicare inpatients", loc="left", fontweight="bold")
    letter(ax, "B")

    p = pd.read_csv(I / "multistate_panel.csv")
    f = p[(p.category == "fusion") & p.year.between(2016, 2023)]
    have = f.groupby("st").year.nunique()
    states = have[have == 8].index
    agg = f[f.st.isin(states)].groupby("year")[["hcup_medicare", "cms_ffs_total", "cms_ffs_published", "TOT_BENES"]].sum()
    ax = fig.add_subplot(gs[1, 0])
    for col, ls, lab, c in [("hcup_medicare", "-", "All Medicare (FFS + MA), state data", DARK),
                            ("cms_ffs_total", "--", "CMS FFS, all discharges", SPINE),
                            ("cms_ffs_published", "-", "CMS FFS, published rows", SPINE)]:
        idx = agg[col] / agg[col].loc[2016] * 100
        ax.plot(idx.index, idx, color=c, ls=ls, lw=2 if col == "hcup_medicare" else 1.6, label=lab)
        ax.text(2023.2, idx.loc[2023], f"{idx.loc[2023] - 100:+.0f}%".replace("-", "−"), fontsize=6.5, color=c, va="center")
    # the same all-Medicare count per Medicare beneficiary (adjusts the count for enrollment change)
    rate = agg.hcup_medicare / agg.TOT_BENES
    ridx = rate / rate.loc[2016] * 100
    ax.plot(ridx.index, ridx, color=DARK, ls=":", lw=1.4, label="All Medicare, per beneficiary")
    ax.text(2023.2, ridx.loc[2023], f"{ridx.loc[2023] - 100:+.0f}%".replace("-", "−"), fontsize=6.5, color=DARK, va="center")
    ax.axhline(100, color=GRAY, lw=0.6)
    ax.set_ylim(40, 138)   # headroom for the legend above the 2019 peak
    ax.set_xlim(2015.7, 2024.2)
    ax.set_ylabel("Index (2016 = 100)")
    ax.set_title(f"Spine fusion inpatients, {len(states)} states", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    JOINT = "#0072B2"
    w = f[f.st.isin(states) & f.year.isin([2016, 2023])].pivot_table(index="st", columns="year", values=["hcup_medicare", "cms_ffs_published"])
    x = (w["hcup_medicare"][2023] / w["hcup_medicare"][2016] - 1) * 100
    yv = (w["cms_ffs_published"][2023] / w["cms_ffs_published"][2016] - 1) * 100
    # joint replacement (MS-DRGs 469-470 in both sources) as a comparator service line
    jp = pd.read_csv(I / "multistate_joints_panel.csv")
    jh = jp.groupby("st").year.apply(lambda s_: set(range(2016, 2024)) <= set(s_))
    jw = jp[jp.st.isin(jh[jh].index) & jp.year.isin([2016, 2023])].pivot_table(index="st", columns="year", values=["hcup_medicare", "cms_ffs_published"])
    jx = (jw["hcup_medicare"][2023] / jw["hcup_medicare"][2016] - 1) * 100
    jy = (jw["cms_ffs_published"][2023] / jw["cms_ffs_published"][2016] - 1) * 100
    lim = [-100, 140]
    ax.fill_between(lim, lim, [lim[0], lim[0]], color="#B0B0B0", alpha=0.12, lw=0)
    ax.plot(lim, lim, color=GRAY, lw=0.8, ls=":")
    cp = pd.read_csv(I / "multistate_cranial_panel.csv")   # cranial, MS-DRGs 23 and 25-27 in both sources
    chv = cp.groupby("st").year.apply(lambda s_: set(range(2016, 2024)) <= set(s_))
    cw = cp[cp.st.isin(chv[chv].index) & cp.year.isin([2016, 2023])].pivot_table(index="st", columns="year", values=["hcup_medicare", "cms_ffs_published"])
    cx = (cw["hcup_medicare"][2023] / cw["hcup_medicare"][2016] - 1) * 100
    cy = (cw["cms_ffs_published"][2023] / cw["cms_ffs_published"][2016] - 1) * 100
    ax.scatter(cx, cy, s=14, color="#6E6E6E", edgecolor="white", lw=0.4, zorder=3)
    ax.scatter(x, yv, s=14, color=SPINE, edgecolor="white", lw=0.4, zorder=4)
    ax.scatter(jx, jy, s=14, color=JOINT, edgecolor="white", lw=0.4, zorder=4)
    tot = lambda wdf, c: (wdf[c][2023].sum() / wdf[c][2016].sum() - 1) * 100
    for lab, wdf, c, xy in [("Spine fusion", w, SPINE, (40, -60)),
                            ("Joint replacement", jw, JOINT, (-97, 78)),
                            ("Cranial", cw, "#6E6E6E", (-97, 132))]:
        txt = f"{lab}, {len(wdf)} states\nall-Medicare {tot(wdf, 'hcup_medicare'):+.0f}%, published {tot(wdf, 'cms_ffs_published'):+.0f}%"
        ax.text(*xy, re.sub(r"-(?=\d)", "\u2212", txt), color=c, fontsize=6.1, ha="left", va="top", fontweight="bold")
    ax.annotate("", xy=(-82, -66), xytext=(-82, 50), arrowprops=dict(arrowstyle="-|>", color=JOINT, lw=0.8, mutation_scale=7))
    ax.axhline(0, color=LIGHT, lw=0.6)
    ax.axvline(0, color=LIGHT, lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("All-Medicare inpatient count (state data), % change")
    ax.set_ylabel("CMS published FFS count, % change")
    ax.set_title("Change 2016–2023 by state", loc="left", fontweight="bold")
    letter(ax, "D")
    fig.savefig(OUT / "Figure2_states.png")
    fig.savefig(OUT / "Figure2_states.pdf")
    print(s.pivot_table(index=["state", "group"], columns="year", values="visible_share").round(2).to_string())
