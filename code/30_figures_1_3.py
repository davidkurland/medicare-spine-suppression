"""Main-text figures 1 and 3 (Figure 2 is made by 31_figure2_states.py).

Figure 1  National Medicare FFS: published vs complete FFS vs per-enrollee (spine, joint replacement, cranial),
          hidden share, and decomposition of the published change by spine family.
Figure 3  Setting migration: inpatient share of Medicare FFS procedures, cervical / lumbar / joint / cranial.
Outputs: manuscript/figures/Figure1_national and Figure3_setting_migration, as .png (600 dpi) and .pdf
"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)
SPINE, BLUE, GRAY, DARK, LIGHT = "#D55E00", "#0072B2", "#8C8C8C", "#222222", "#C8C8C8"
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442", "#000000"]
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def panel_letter(fig, ax, L, fallback=-0.10):
    """Bold panel letter on the title line, aligned with the left edge of the y-axis label and tick labels
    (or set a fixed distance left of the plot when the panel has no y axis)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pos = ax.get_position()
    bb = ax.yaxis.get_tightbbox(r)
    if bb is not None and bb.width > 12:
        x = inv.transform((bb.x0, 0))[0]
    else:
        x = pos.x0 + fallback * pos.width
    tb = ax._left_title.get_window_extent(r)
    y = inv.transform((0, tb.y0))[1] + 1.5 / 72 / fig.get_figheight()
    fig.text(x, y, L, fontsize=10, fontweight="bold", va="bottom", ha="left")


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", fontsize=6.5, color=color, va="center")


def fig1():
    """Panels A-C show spine, joint replacement and cranial volume indexed to 2013: published hospital rows,
    all FFS discharges and all FFS discharges per FFS enrollee. The shaded wedge between the published and
    complete lines is the volume missing from the public file. Panel D shows the hidden share for the three
    lines and panel E decomposes the published spine change."""
    from matplotlib.lines import Line2D
    dec = pd.read_csv(I / "national_decomposition.csv")
    cmp = pd.read_csv(I / "service_line_comparators.csv")
    sup = pd.read_csv(I / "suppression_share_by_line.csv").set_index("line")
    sup23 = pd.read_csv(I / "suppression_share_by_line_2013_2023.csv").set_index("line")

    fig = plt.figure(figsize=(7.2, 7.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.0], hspace=0.52)
    top = outer[0].subgridspec(1, 3, wspace=0.30)
    bot = outer[1].subgridspec(1, 2, wspace=0.58)
    panels = []

    def head(ax, L, title, sub=None):
        ax.set_title(title + ("\n" + sub if sub else ""), loc="left", fontweight="bold", linespacing=1.5)
        panels.append((ax, L))

    LINE_COL = {"Spine": SPINE, "Joint replacement": BLUE, "Cranial": "#6E6E6E"}
    SUBS = {"Spine": "Partial migration", "Joint replacement": "Migration mostly complete", "Cranial": "No migration"}

    # A-C: published vs complete for each service line; the shaded wedge is what suppression removes
    for k, (name, L) in enumerate(zip(["Spine", "Joint replacement", "Cranial"], "ABC")):
        ax = fig.add_subplot(top[0, k])
        d = cmp[cmp.line == name].set_index("year")
        col = LINE_COL[name]
        pub = d.published / d.published.loc[2013] * 100
        tot = d.complete / d.complete.loc[2013] * 100
        rate = d.per_1k_ffs / d.per_1k_ffs.loc[2013] * 100
        ax.fill_between(d.index, pub, tot, color=col, alpha=0.20, lw=0, zorder=1)
        ax.plot(d.index, tot, color=col, ls="--", lw=1.5, zorder=3)
        ax.plot(d.index, pub, color=col, ls="-", lw=1.9, zorder=4)
        ax.plot(d.index, rate, color=DARK, ls=":", lw=1.3, zorder=2)
        ax.axhline(100, color=GRAY, lw=0.6, zorder=0)
        # keep the two end labels legible when the published and complete lines finish close together
        ys = sorted([(pub.loc[2024], "pub"), (tot.loc[2024], "tot")])
        if ys[1][0] - ys[0][0] < 16:
            mid = (ys[0][0] + ys[1][0]) / 2
            ys = [(mid - 8, ys[0][1]), (mid + 8, ys[1][1])]
        for yv, which in ys:
            sr = pub if which == "pub" else tot
            ax.annotate(f"{sr.loc[2024] - 100:+.0f}%".replace("-", "\u2212"), (2024.15, yv),
                        xytext=(3, 0), textcoords="offset points", fontsize=6.8, color=col,
                        fontweight="bold", va="center")
        if name == "Spine":                # CY2024 holds 9 months under the outgoing fusion DRGs and 3 under the FY2025 DRGs
            ax.axvline(2023.5, color=GRAY, lw=0.7, ls=(0, (1, 2)), zorder=0)
            ax.text(2023.3, 172, "2024: year split\nbetween 2 DRG sets", fontsize=5.6, color="#555555", ha="right", va="top", linespacing=1.2)
        ax.set_ylim(-14, 178)
        ax.set_xlim(2012.6, 2026.4)
        ax.set_xticks([2013, 2018, 2024])
        ax.set_yticks([0, 50, 100, 150])
        if k == 0:
            ax.set_ylabel("Volume, index (2013 = 100)")
        else:
            ax.set_yticklabels([])
        s_pct = sup.loc[name, "suppression_pct_of_published_change"]
        s23 = sup23.loc[name, "suppression_pct_of_published_change"]
        ax.text(0.5, 0.012, f"Suppression share of published log\nchange: {s23:.0f}% to 2023, {s_pct:.0f}% to 2024", transform=ax.transAxes,
                fontsize=6.4, ha="center", va="bottom", color=DARK, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LIGHT, lw=0.5))
        head(ax, L, name, SUBS[name])

    h = [Line2D([], [], color=DARK, ls="-", lw=1.9, label="Published hospital rows"),
         Line2D([], [], color=DARK, ls="--", lw=1.5, label="All FFS discharges"),
         Line2D([], [], color=DARK, ls=":", lw=1.3, label="All FFS per FFS enrollee"),
         mpl.patches.Patch(fc=GRAY, alpha=0.30, lw=0, label="Gap, published vs all FFS")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.505), ncol=4, frameon=False,
               fontsize=6.8, handlelength=2.2, columnspacing=1.6)

    # D: share hidden, all three lines
    ax = fig.add_subplot(bot[0, 0])
    for name in ["Spine", "Joint replacement", "Cranial"]:
        d = cmp[cmp.line == name].set_index("year")
        ax.plot(d.index, d.hidden_share * 100, color=LINE_COL[name], marker="o", ms=2.4, label=name)
        end_label(ax, 2024, d.hidden_share.loc[2024] * 100, f"{d.hidden_share.loc[2024] * 100:.0f}%", LINE_COL[name])
    ax.axvline(2023.5, color=GRAY, lw=0.7, ls=(0, (1, 2)), zorder=0)
    ax.text(2023.3, 54, "2024: year split\nbetween 2 DRG sets", fontsize=5.6, color="#555555", ha="right", va="top", linespacing=1.2)
    ax.set_ylim(0, 55)
    ax.set_xlim(2012.6, 2025.8)
    ax.set_xticks([2013, 2016, 2019, 2022, 2024])
    ax.set_ylabel("FFS discharges hidden, %")
    ax.legend(frameon=False, loc="upper left", fontsize=6.8)
    head(ax, "D", "Share of volume missing from the file")

    # E: decomposition of the published spine decline
    ax = fig.add_subplot(bot[0, 1])
    fams = [("thoracolumbar_fusion", "Thoracolumbar fusion"), ("deformity_fusion", "Deformity fusion"),
            ("cervical_fusion", "Cervical fusion"), ("spine_nonfusion", "Spine nonfusion"), ("spine", "All spine, 2013\u20132024"), ("spine@2013-2023", "All spine, 2013\u20132023")]
    comp = [("suppression_log", "Suppression", "#9E9E9E"), ("enrollment_log", "FFS enrollment", "#E69F00"),
            ("rate_log", "Per-enrollee change", SPINE)]
    d = dec[(dec.window == "2013-2024")].set_index("stratum")
    d23 = dec[(dec.window == "2013-2023")].set_index("stratum").rename(index=lambda k: k + "@2013-2023")
    d = pd.concat([d, d23])
    for i, (key, lab) in enumerate(fams):
        pos, neg = 0.0, 0.0
        for c, clab, cc in comp:
            v = d.loc[key, c] * 100
            left = pos if v >= 0 else neg
            ax.barh(i, v, left=left, color=cc, height=0.6, edgecolor="white", lw=0.5, label=clab if i == 0 else None)
            if v >= 0:
                pos += v
            else:
                neg += v
        tot = d.loc[key, "log_published"] * 100
        ax.plot([tot], [i], marker="|", ms=10, color=DARK, mew=1.5)
        ax.text(min(neg, tot) - 3, i, f"{np.expm1(d.loc[key, 'log_published']) * 100:+.0f}%".replace("-", "\u2212"),
                va="center", ha="right", fontsize=6.5)
    ax.axvline(0, color=GRAY, lw=0.6)
    ax.set_yticks(np.arange(len(fams)))
    ax.set_yticklabels([l for _, l in fams])
    ax.invert_yaxis()
    ax.set_xlim(-230, 40)
    ax.set_xlabel("Contribution to published change, log points")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3)
    head(ax, "E", "Why published spine volume fell")

    for a, L in panels:
        panel_letter(fig, a, L)
    save(fig, "Figure1_national")


def fig3():
    m = pd.read_csv(I / "setting_migration.csv")
    w = m.pivot_table(index=["family", "year"], columns="setting", values="services", aggfunc="sum").fillna(0)
    w["share"] = w["inpatient"] / w.sum(axis=1)
    w = w.reset_index()
    panels = [
        ("Cervical spine", [("ACDF (22551)", "ACDF"), ("Cervical disc arthroplasty (22856)", "Arthroplasty"),
                            ("Posterior cervical fusion (22600)", "Posterior fusion"), ("Cervical laminectomy/laminoplasty (63045, 63050, 63051)", "Laminectomy")]),
        ("Lumbar spine", [("Lumbar discectomy (63030)", "Discectomy"), ("Lumbar laminectomy (63047)", "Laminectomy"),
                          ("Lumbar interbody fusion, posterior/TLIF (22630, 22633)", "Posterior/TLIF"),
                          ("Lumbar posterolateral fusion (22612)", "Posterolateral"), ("Anterior lumbar interbody fusion (22558)", "ALIF")]),
        ("Joint replacement", [("Total knee arthroplasty (27447)", "Knee"), ("Total hip arthroplasty (27130)", "Hip")]),
        ("Cranial", [("Craniotomy for tumor (61510, 61512, 61518)", "Craniotomy, tumor"), ("Burr hole/craniotomy for hematoma (61154, 61312)", "Hematoma evacuation"),
                     ("Ventriculoperitoneal shunt (62223)", "VP shunt"), ("Mechanical thrombectomy (61645)", "Thrombectomy")]),
    ]
    fig = plt.figure(figsize=(7.4, 3.5))
    # 4 panels in the order of Figure 1 (spine, joints, cranial); gutter sized to each panel's longest end label
    axes = [fig.add_axes([x, 0.125, 0.140, 0.76]) for x in (0.07, 0.308, 0.546, 0.762)]
    for ax, (title, fams), L in zip(axes, panels, "ABCD"):
        ends = []
        for i, (fam, lab) in enumerate(fams):
            d = w[(w.family == fam) & (w.year >= 2013)].sort_values("year")
            d = d[d[["inpatient", "hospital outpatient", "ASC", "other"]].sum(axis=1) >= 400]
            if d.empty:
                continue
            col = OKABE[i]
            ax.plot(d.year, d.share * 100, color=col, marker="o", ms=2, lw=1.4)
            ends.append([d.year.iloc[-1], d.share.iloc[-1] * 100, lab, col, d.share.iloc[-1] * 100])
        ends.sort(key=lambda e: e[1])
        for j in range(1, len(ends)):
            if ends[j][1] - ends[j - 1][1] < 7:
                ends[j][1] = ends[j - 1][1] + 7
        over = ends[-1][1] - 101 if ends else 0
        if over > 0:                      # keep the label stack inside the axis
            for e in ends:
                e[1] -= over
        for x, yv, lab, col, y_true in ends:
            ax.text(2026.2, yv, lab, fontsize=6, color=col, va="center")
            # leader from the line's last point to its label, so a label moved to avoid overlap stays tied to its line
            ax.plot([2024.35, 2025.9], [y_true, yv], color=col, lw=0.45, clip_on=False, solid_capstyle="round")
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        for gy in (20, 40, 60, 80, 100):
            ax.axhline(gy, color="#E8E8E8", lw=0.5, zorder=0)
        ax.set_ylim(0, 108)
        ax.set_xlim(2012.5, 2024.5)
        ax.set_xticks([2013, 2018, 2024])
        ax.set_title(title, loc="left", fontweight="bold")
        if ax is axes[0]:
            ax.spines["left"].set_bounds(0, 100)   # end the axis at the last labeled tick
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(left=False)
    axes[0].set_ylabel("Performed as inpatient,\n% of Medicare FFS services")
    for a, L in zip(axes, "ABCD"):
        panel_letter(fig, a, L, fallback=-0.22)
    save(fig, "Figure3_setting_migration")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["fig1", "fig3"]):
        globals()[name]()
        print("saved", name)
