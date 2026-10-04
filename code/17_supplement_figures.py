"""Supplement figures built from the state and national panels: the specification curve for the change in inpatient
volume, agreement between case definitions, completeness of state discharge data against cost reports, and Indiana
hospital rank concordance under automated and verified linkage.

Outputs: manuscript/figures/eFigure_{specification_trend,case_definitions,completeness,linkage_sensitivity}.png and .pdf
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
SC = I / "state_compare"
OUT = ROOT / "manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)

SPINE, CRAN, GRAY, DARK, LIGHT = "#D55E00", "#6E6E6E", "#8C8C8C", "#222222", "#BDBDBD"   # cranial = gray control
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def specification_trend():
    t = pd.read_csv(I / "multiverse_trend.csv").dropna(subset=["change"])
    sources = ["SPARCS all-payer", "SPARCS Medicare payer", "CMS Medicare FFS, NY", "CMS Medicare FFS, US"]
    labels = ["NY all-payer\n(state records)", "NY Medicare payer\n(state records)", "NY Medicare FFS\n(CMS)", "US Medicare FFS\n(CMS)"]
    # spine only: the New York APR-DRG and CMS MS-DRG cranial definitions differ, so a cranial panel would compare unlike cohorts
    fig, ax0 = plt.subplots(figsize=(4.6, 3.2))
    axes = [ax0]
    marks = {"2013-2024": "o", "2016-2024": "s", "2019-2024": "^"}
    for ax, g, col in [(axes[0], "spine", SPINE)]:
        for i, s in enumerate(sources):
            d = t[(t.grp == g) & (t.source == s)]
            for w, mk in marks.items():
                dd = d[d.window == w]
                jit = np.linspace(-0.18, 0.18, len(dd)) if len(dd) > 1 else np.zeros(len(dd))
                ax.scatter(dd.change * 100, np.full(len(dd), i) + jit, marker=mk, s=16, facecolor=col if s.startswith("CMS") else "white", edgecolor=col, lw=0.8, label=w if i == 0 else None)
        ax.axvline(0, color=GRAY, lw=0.6)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel("Change in inpatient spine discharges, %")
    axes[0].legend(title="Window", frameon=False, loc="lower right", fontsize=6.5, title_fontsize=6.5)
    fig.tight_layout()
    save(fig, "eFigure_specification_trend")


def case_definitions():
    nyd = pd.read_csv(I / "ny_neuro_definitions.csv").query("year >= 2013")     # study period
    ind = pd.read_csv(SC / "IN_neuro_definitions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for g, col in [("spine", SPINE), ("cranial", CRAN)]:
        d = nyd[nyd.grp == g]
        axes[0].plot(d.year, d.jaccard, color=col, marker="o", ms=2.5, label=g.title())
        d = ind[ind.grp == g]
        axes[1].plot(d.year, d.pcs_over_msdrg, color=col, marker="o", ms=2.5, label=g.title())
    axes[0].axvline(2015.75, color=LIGHT, ls=":"); axes[0].text(2015.85, 0.43, "ICD-10", fontsize=6, color=GRAY)
    axes[0].axvline(2018, color=LIGHT, ls=":"); axes[0].text(2018.1, 0.43, "CCSR", fontsize=6, color=GRAY)
    axes[0].set_ylim(0.4, 1); axes[0].set_ylabel("Jaccard overlap"); axes[0].set_title("NY: APR-DRG vs principal procedure cohorts", loc="left", fontweight="bold")
    axes[1].axhline(1, color=GRAY, lw=0.6); axes[1].set_ylabel("PCS-defined ÷ MS-DRG-defined discharges")
    axes[1].set_title("IN: ICD-10-PCS vs MS-DRG cohorts", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    panel(axes[0], "A"); panel(axes[1], "B")
    fig.tight_layout(w_pad=3)
    save(fig, "eFigure_case_definitions")


def completeness():
    ny = pd.read_csv(I / "ny_external_validity_by_year.csv").query("year >= 2013")     # study period
    inn = pd.read_csv(SC / "IN_by_year.csv")
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.axhspan(0.95, 1.05, color="#EEEEEE", lw=0)
    ax.plot(ny.year, ny.median_ratio_no_infant, color=DARK, marker="o", ms=2.5, label="New York (infants excluded)")
    ax.plot(inn.year, inn.median_ratio, color=GRAY, marker="s", ms=2.5, ls="--", label="Indiana")
    ax.axhline(1, color=GRAY, lw=0.6)
    ax.set_ylim(0.93, 1.07)
    ax.set_yticks([0.95, 1.0, 1.05])
    ax.set_ylabel("Median hospital ratio,\nstate ÷ cost-report discharges")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "eFigure_completeness")


def linkage_sensitivity():
    auto = pd.read_csv(ROOT / "data/manual/IN_rank_agreement_automated_crosswalk.csv")
    cells = pd.read_csv(SC / "IN_neuro_vs_cms_cells.csv", dtype={"ccn": str})
    from scipy.stats import spearmanr
    rows = []
    for (g, y), d in cells[cells.year >= 2017].groupby(["grp", "year"]):
        hp = d.dropna(subset=["cms_ffs"]).fillna({"state_medicare": 0}).groupby("ccn")[["cms_ffs", "state_medicare"]].sum()
        rows.append({"grp": g, "year": y, "rho": spearmanr(hp.cms_ffs, hp.state_medicare).statistic})
    ver = pd.DataFrame(rows).merge(auto, on=["grp", "year"])
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for g, col in [("spine", SPINE), ("cranial", CRAN)]:
        d = ver[ver.grp == g]
        ax.plot(d.year, d.rho, color=col, marker="o", ms=2.5, label=f"{g.title()}, verified linkage")
        ax.plot(d.year, d.rho_hospital_automated, color=col, ls="--", marker="o", ms=2.5, mfc="white", label=f"{g.title()}, automated linkage")
    jr = pd.read_csv(SC / "IN_joint_rank_concordance.csv")   # from 61_indiana_joint_linkage.py, same construction
    ax.plot(jr.year, jr.rho, color="#0072B2", marker="o", ms=2.5, label="Joint replacement, verified linkage")
    ax.plot(jr.year, jr.rho_hospital_automated, color="#0072B2", ls="--", marker="o", ms=2.5, mfc="white", label="Joint replacement, automated linkage")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spearman ρ,\nstate vs CMS hospital volume")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.set_size_inches(7.0, 3.4)
    fig.tight_layout()
    save(fig, "eFigure_linkage_sensitivity")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    which = sys.argv[1:] or ["specification_trend", "case_definitions", "completeness", "linkage_sensitivity"]
    for w in which:
        globals()[w]()
        print("saved", w)
