"""Was the 2021 dip-and-rebound in inpatient share COVID, clinical over-reach, or Medicare's Inpatient-Only (IPO) list?

Natural experiment from CMS OPPS rules:
  CY2020  22633/22634 (posterior lumbar interbody + posterolateral fusion) removed from IPO (84 FR 61142), stayed off.
  CY2021  IPO phase-out began: 298 services removed, mostly musculoskeletal (85 FR 85866).
  CY2022  phase-out halted; removed services returned to IPO except 22630 (and 23472, 27702) (86 FR 63458).
  Jul 2021 prior authorization required for 22551/22552 in hospital outpatient departments.
Codes billable in ASCs throughout (22551, 22612, 63030, 63045, 63047) were not IPO and serve as comparators;
codes on the IPO list in every year (62223, 63051) serve as negative controls.
If COVID capacity drove the 2021 shift, all classes should move together in 2020-2021 and rebound together;
if the IPO list drove it, only codes removed in 2021 and returned in 2022 should show a V.
Classes verified against CMS OPPS Addendum E, CY2019-CY2024 (38_addendum_e.py -> data/interim/ipo_status_by_year.csv).
22632 (removed 2021, returned 2022, removed again 2023) is an add-on code and is not plotted.
Knee (27447) and hip (27130) replacement, removed from the IPO list in CY2018 and CY2020 and never returned, are the joint
replacement comparator in panel A; panel C adds complete joint replacement DRG discharges per FFS enrollee.
Outputs: data/interim/ipo_policy_by_cpt.csv, ipo_policy_summary.csv; manuscript/figures/eFigure_ipo_policy.png/pdf
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8.5, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight"})
GRAY, LIGHT = "#8C8C8C", "#D9D9D9"

CLASSES = {
    "Removed 2021, returned 2022": {"22600": "Posterior cervical fusion", "22558": "ALIF", "22595": "C1-C2 fusion"},
    "Removed 2021, stayed off": {"22630": "PLIF, interbody only"},
    "Removed 2020, stayed off": {"22633": "PLIF/TLIF + posterolateral", "22634": "PLIF/TLIF add-on level"},
    "Never IPO (ASC-payable)": {"22551": "ACDF", "22612": "Posterolateral fusion", "63045": "Cervical laminectomy",
                                "63047": "Lumbar laminectomy", "63030": "Lumbar discectomy"},
    "IPO throughout (control)": {"62223": "VP shunt", "63051": "Cervical laminoplasty"},
    "Joint replacement (removed 2018 and 2020, stayed off)": {"27447": "Knee replacement", "27130": "Hip replacement"},
}
# blue is reserved for joint replacement in every figure, so "removed 2021, stayed off" takes amber
COL = {"Removed 2021, returned 2022": "#D55E00", "Removed 2021, stayed off": "#E69F00",
       "Removed 2020, stayed off": "#009E73", "Never IPO (ASC-payable)": "#6F6F6F", "IPO throughout (control)": "#CC79A7",
       "Joint replacement (removed 2018 and 2020, stayed off)": "#0072B2"}


def load():
    d = pd.read_csv(I / "setting_migration_cpt.csv", dtype={"cpt": str})
    d = d[d.year.between(2015, 2024)]
    w = d.pivot_table(index=["cpt", "year"], columns="setting", values="services", aggfunc="sum").fillna(0)
    rep = d.pivot_table(index=["cpt", "year"], columns="setting", values="svc_reported", aggfunc="sum").fillna(0)
    w["total"] = w.sum(axis=1)
    w["outpatient_share"] = (w["hospital outpatient"] + w["ASC"]) / w["total"]
    w["outpatient_reported"] = rep["hospital outpatient"] + rep["ASC"]
    w = w.reset_index()
    cls = {c: (k, lab) for k, v in CLASSES.items() for c, lab in v.items()}
    w = w[w.cpt.isin(cls)]
    w["ipo_class"] = w.cpt.map(lambda c: cls[c][0])
    w["label"] = w.cpt.map(lambda c: cls[c][1])
    return w


def summary(w):
    """Per code: outpatient share 2019/2020/2021/2022/2024 and the 2021 'V' (rise into 2021 minus fall out of it)."""
    p = w.pivot_table(index=["ipo_class", "cpt", "label"], columns="year", values="outpatient_share")
    s = pd.DataFrame({y: p[y] * 100 for y in (2019, 2020, 2021, 2022, 2024)})
    s.columns = [f"op_share_{y}" for y in s.columns]
    s["change_2020_2021_pts"] = (p[2021] - p[2020]) * 100
    s["change_2021_2022_pts"] = (p[2022] - p[2021]) * 100
    s["reversal_pct_of_2021_rise"] = -s.change_2021_2022_pts / s.change_2020_2021_pts * 100
    return s.reset_index()


def figure(w, nat):
    fig = plt.figure(figsize=(7.4, 6.8))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.95)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    # A: outpatient share, codes with a policy change
    for ax, keys, title, L in [(axes[0], ["Removed 2021, returned 2022", "Removed 2021, stayed off", "Removed 2020, stayed off", "IPO throughout (control)", "Joint replacement (removed 2018 and 2020, stayed off)"], "Codes whose IPO status changed, and controls", "A"),
                               (axes[1], ["Never IPO (ASC-payable)"], "Codes never on IPO list", "B")]:
        ends = []
        for k in keys:
            for cpt, g in w[w.ipo_class == k].groupby("cpt"):
                g = g[g.year >= 2016].sort_values("year")
                ax.plot(g.year, g.outpatient_share * 100, color=COL[k], marker="o", ms=2, lw=1.2,
                        alpha=1 if k != "Never IPO (ASC-payable)" else 0.85)
                ends.append([g.outpatient_share.iloc[-1] * 100, f"{g.label.iloc[0]} ({cpt})", COL[k], g.outpatient_share.iloc[-1] * 100])
        ends.sort(key=lambda e: e[0])
        if ends:
            ends[0][0] = max(ends[0][0], 3.0)     # keep the lowest label clear of the axis line
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + 5.0)
        for y, lab, c, y_true in ends:
            ax.text(2025.5, y, lab, fontsize=6, color=c, va="center")
            ax.plot([2024.2, 2025.35], [y_true, y], color=c, lw=0.45, clip_on=False)      # leader from the 2024 point to its label
        for x in (2020, 2021, 2022):
            ax.axvline(x, color=LIGHT, lw=0.8, zorder=0)
        ax.set_xlim(2015.6, 2024.4)
        ax.set_xticks([2016, 2018, 2020, 2022, 2024])
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", fontweight="bold", pad=24)
        ax.text(-0.25, 1.17, L, transform=ax.transAxes, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Performed outpatient (HOPD or ASC),\n% of Medicare FFS services")
    for x, t, yy in [(2020, "COVID", 101), (2021, "IPO cut", 107), (2022, "IPO back", 101)]:
        for a in axes[:2]:
            a.text(x, yy, t, fontsize=5.5, color=GRAY, ha="center", va="bottom", clip_on=False)
    for k, c in COL.items():
        axes[0].plot([], [], color=c, label=k)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", fontsize=6.2, ncol=3, bbox_to_anchor=(0.5, 0.535))
    for x in (2018, 2020):   # knee and hip replacement left the IPO list
        axes[0].annotate("", xy=(x, 96), xytext=(x, 100), arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=0.8))
    # C: complete national inpatient DRG discharges per FFS enrollee
    ax = axes[2]
    ends_c = []
    # cranial is the gray control in every figure; blue is reserved for joint replacement
    for fam, lab, c in [("cervical_fusion", "Cervical fusion DRGs", "#D55E00"), ("thoracolumbar_fusion", "Thoracolumbar fusion DRGs", "#009E73"),
                        ("spine_nonfusion", "Spine nonfusion DRGs", "#CC79A7"), ("cranial_open_other", "Craniotomy DRGs (020-022, 025-027)", "#6E6E6E")]:
        s = nat[fam]
        idx = s / s.loc[2019] * 100
        ax.plot(idx.index, idx, color=c, marker="o", ms=2, lw=1.3)
        ends_c.append([idx.loc[2024], lab, c, idx.loc[2024]])
    jr = pd.read_csv(I / "service_line_comparators.csv").query("line == 'Joint replacement'").set_index("year").per_1k_ffs.loc[2017:2024]
    jr = jr / jr.loc[2019] * 100
    ax.plot(jr.index, jr, color="#0072B2", marker="o", ms=2, lw=1.3)
    ends_c.append([jr.loc[2024], "Joint replacement DRGs", "#0072B2", jr.loc[2024]])
    ends_c.sort(key=lambda e: e[0])
    for j in range(1, len(ends_c)):
        ends_c[j][0] = max(ends_c[j][0], ends_c[j - 1][0] + 5.5)
    for y, lab, c, y_true in ends_c:
        ax.text(2024.75, y, lab, fontsize=6, color=c, va="center")
        ax.plot([2024.1, 2024.7], [y_true, y], color=c, lw=0.45, clip_on=False)
    ax.axhline(100, color=LIGHT, lw=0.6)
    ax.set_xlim(2016.6, 2024.3)
    ax.set_xticks([2017, 2019, 2021, 2023])
    ax.set_ylim(20, 130)
    ax.set_yticks([25, 50, 75, 100, 125])
    ax.set_ylabel("Inpatient discharges per FFS enrollee\n(complete data, 2019 = 100)")
    ax.set_title("National inpatient volume", loc="left", fontweight="bold")
    ax.text(-0.09, 1.05, "C", transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.savefig(OUT / "eFigure_ipo_policy.png")
    fig.savefig(OUT / "eFigure_ipo_policy.pdf")


if __name__ == "__main__":
    w = load()
    w.to_csv(I / "ipo_policy_by_cpt.csv", index=False)
    s = summary(w)
    s.to_csv(I / "ipo_policy_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(1).to_string(index=False))
    print(s.groupby("ipo_class")[["change_2020_2021_pts", "change_2021_2022_pts"]].median().round(1))
    nat = pd.read_csv(I / "suppression_exact_national.csv")
    ns = pd.read_csv(I / "national_sensitivity.csv").set_index("year")
    nat = nat.pivot_table(index="year", columns="stratum", values="total_ffs").div(ns.ffs_part_a_benes, axis=0)
    nat = nat.loc[2017:2024]
    print((nat / nat.loc[2019] * 100).round(1)[["cervical_fusion", "thoracolumbar_fusion", "spine_nonfusion", "cranial_open_other"]].to_string())
    figure(w, nat)
