"""Figure 4. Consequences: invisible high-volume hospitals and a sicker inpatient pool.

A  New York 2024: every hospital with >=100 all-payer spine discharges, ranked by volume; hospitals with no published
   CMS spine row in orange.
B  New York: among hospitals with >=100 all-payer cases in the CMS file, share with every cell suppressed, for spine,
   joint replacement and cranial.
C  National complete Medicare FFS discharges in the lowest severity tier: cervical fusion without CC/MCC (MS-DRG 473),
   joint replacement without MCC (MS-DRGs 470 and 522), and craniotomy without CC/MCC (MS-DRG 027, coding-intensity
   control).
D  New York all payers: share of APR severity level 1 among cervical fusion/other back-neck (APR-DRG 321) vs craniotomy
   (APR-DRG 021, control).
Output: manuscript/figures/Figure4_consequences.png/.pdf
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)
SPINE, CRAN, LIGHT = "#D55E00", "#6E6E6E", "#D0D0D0"   # cranial = gray control, as in Figure 1
JOINT = "#0072B2"
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
    """Bold one-line title with a lighter subtitle naming the measure; the letter is placed later by panel_letter."""
    ax.set_title(title, loc="left", fontweight="bold", pad=15)
    ax.text(0, 1.025, sub, transform=ax.transAxes, fontsize=6.8, color="#555555", va="bottom")
    PANELS.append((ax, L))


def label_line(ax, d, ycol, color, dy=0):
    v = d[ycol].iloc[-1] * 100
    ax.annotate(f"{v:.0f}%", (d.year.iloc[-1], v), xytext=(4, dy), textcoords="offset points", fontsize=6.5, color=color, va="center")
    ax.annotate(f"{d[ycol].iloc[0] * 100:.0f}%", (d.year.iloc[0], d[ycol].iloc[0] * 100), xytext=(-5, 0), textcoords="offset points",
                fontsize=6.5, color=color, va="center", ha="right")


if __name__ == "__main__":
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.62, top=0.92, bottom=0.07)

    # A: New York 2024, every hospital with >=100 all-payer spine discharges, ranked; absent hospitals in orange
    r = pd.read_csv(I / "ny_rates_rankings/rank_table.csv")
    g = r[(r.grp == "spine") & (r.year == 2024) & (r.all_payer >= 100)].sort_values("all_payer", ascending=False).reset_index(drop=True)
    absent = g.cms_ffs <= 0
    n_inv, n_hv = int(absent.sum()), len(g)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(g.index[~absent], g.all_payer[~absent], color=LIGHT, width=0.8, label="Has a published CMS spine row")
    ax.bar(g.index[absent], g.all_payer[absent], color=SPINE, width=0.8, label=f"No published row ({n_inv} of {n_hv})")
    ax.set_xlim(-1, n_hv)
    ax.set_xticks([])
    ax.set_ylim(0, 2400)
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.set_ylabel("All-payer spine discharges, 2024")
    ax.set_xlabel(f"{n_hv} New York hospitals, ranked by volume")
    ax.spines["bottom"].set_visible(False)
    med = int(g.all_payer[absent].median()); mx = int(g.all_payer[absent].max())
    ax.text(0.98, 0.62, f"Absent hospitals: median {med},\nlargest {mx} spine discharges", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=SPINE)
    ax.legend(frameon=False, loc="upper right", fontsize=6.3)
    head(ax, "A", f"{n_inv} of {n_hv} New York hospitals are absent", "2024, hospitals with ≥100 all-payer spine discharges, ranked")

    # B: mechanism test in New York (Indiana is in the Supplement).
    # Among high-volume hospitals that appear in the CMS file, the share whose every cell for the line is suppressed;
    # exempt and unlinked hospitals are excluded from all lines (57_joints_missing_hospitals.py). End labels carry the 2024 counts.
    sup = pd.read_csv(I / "suppressed_high_volume_hospitals.csv")
    ax = fig.add_subplot(gs[0, 1])
    ends = []
    for line, c in [("Spine", SPINE), ("Joint replacement", JOINT), ("Cranial", CRAN)]:
        d = sup[(sup.line == line) & (sup.state == "New York")].sort_values("year").dropna(subset=["suppressed_share"])
        ax.plot(d.year, d.suppressed_share * 100, color=c, marker="o", ms=2.5, lw=1.5, clip_on=False, zorder=5)
        r = d.iloc[-1]
        ends.append([r.suppressed_share * 100, f"{line} {r.suppressed_share * 100:.0f}%", f"{int(r.suppressed)} of {int(r.in_file)}", c])
    ends.sort(key=lambda e: e[0])
    ends[0][0] = max(ends[0][0], 1.5)
    for q in range(1, len(ends)):
        ends[q][0] = max(ends[q][0], ends[q - 1][0] + 6.5)
    for yv, lab, n, c in ends:
        ax.text(2024.6, yv + 1.2, lab, fontsize=6.4, color=c, va="center")
        ax.text(2024.6, yv - 1.6, n, fontsize=5.8, color=c, va="center", alpha=0.85)
    ax.set_ylim(0, 42)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlim(2012.6, 2024.4)
    ax.spines["bottom"].set_bounds(2012.6, 2024.4)
    ax.set_xticks([2013, 2016, 2019, 2022, 2024])
    ax.set_ylabel("Hospitals with every cell suppressed, %")
    head(ax, "B", "Suppression hides spine hospitals most", "New York, ≥100 cases a year and in the CMS file")

    # C: national FFS, share without CC/MCC
    t = pd.read_csv(I / "casemix_tiers_national.csv")
    ax = fig.add_subplot(gs[1, 0])
    # each line in its own lowest severity tier; joint replacement has no CC tier, so its lowest tier is "without MCC"
    for fam, lab, c, xy, va in [("joint_replacement_469_470", "Joint replacement (MS-DRGs 470, 522)", JOINT, (2019.5, 99), "bottom"),
                                ("cervical_fusion", "Cervical fusion (MS-DRG 473)", SPINE, (2021.7, 48), "bottom"),
                                ("cranial_control_025_027", "Craniotomy control (MS-DRG 027)", CRAN, (2018.5, 15), "top")]:
        d = t[(t.family == fam) & (t.tier == "none") & (t.year >= 2016)].sort_values("year")   # from 2016, the first full year of ICD-10-CM coding
        ax.plot(d.year, d.share * 100, color=c, marker="o", ms=2.5)
        label_line(ax, d, "share", c)
        ax.text(*xy, lab, color=c, fontsize=6.5, ha="center", va=va)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(2014.0, 2025.4)
    ax.spines["bottom"].set_bounds(2016, 2024)
    ax.set_xticks([2016, 2018, 2020, 2022, 2024])
    ax.set_ylabel("% of inpatient discharges")
    head(ax, "C", "Uncomplicated cases left the pool", "US Medicare FFS, share in the lowest severity tier, 2016-2024")

    # D: New York all payers, minor severity share
    sv = pd.read_csv(I / "ny_severity_shift.csv")
    sv = sv[sv.scope == "all payers"]
    ax = fig.add_subplot(gs[1, 1])
    for lab, name, c in [("cervical fusion & other back/neck (321)", "Cervical fusion, back/neck (APR-DRG 321)", SPINE),
                         ("craniotomy except trauma (021, control)", "Craniotomy control (APR-DRG 021)", CRAN)]:
        d = sv[(sv.apr == lab) & (sv.year >= 2016)].sort_values("year")
        ax.plot(d.year, d.soi_minor_share * 100, color=c, marker="o", ms=2.5)
        label_line(ax, d, "soi_minor_share", c)
        if c == SPINE:
            ax.text(2024.6, 62, "Cervical fusion, back/neck (APR-DRG 321)", color=c, fontsize=6.5, ha="right", va="bottom")
        else:
            ax.text(2020, 8, name, color=c, fontsize=6.5, ha="center", va="top")
    ax.set_ylim(0, 78)
    ax.set_xlim(2014, 2025.4)
    ax.spines["bottom"].set_bounds(2016, 2024)
    ax.set_xticks([2016, 2018, 2020, 2022, 2024])
    ax.set_ylabel("% with minor severity of illness")
    head(ax, "D", "Minor severity fell in both groups", "New York all-payer inpatients, share with minor APR severity, 2016-2024")

    for a, L in PANELS:
        panel_letter(fig, a, L)
    fig.savefig(OUT / "Figure4_consequences.png")
    fig.savefig(OUT / "Figure4_consequences.pdf")
    print("saved")
