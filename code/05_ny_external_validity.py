"""External validity of SPARCS hospital-year discharge counts against CMS HCRIS cost reports.

Unit: CCN x calendar year, 2012-2022 (full HCRIS coverage). SPARCS counts come from DOH's
published facility aggregate (all PFIs mapped to the CCN summed). HCRIS counts exclude newborn
nursery stays, so SPARCS is expected to run modestly higher at hospitals with obstetrics.
Inputs: data/raw/ny_ivw2-k53g.csv (health.data.ny.gov facility aggregate, Socrata dataset
ivw2-k53g), data/interim/ny_pfi_ccn_crosswalk.csv (03_ny_crosswalk.py),
data/interim/hcris_hospital_cy.parquet (02_hcris_national.py).
Outputs: data/interim/ny_ccn_year_compare.parquet, ny_hospital_discordance.csv,
ny_external_validity_by_year.csv, and two diagnostic plots (figures/ny_sparcs_vs_hcris_2019.png,
figures/ny_ratio_by_year.png).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})


def build() -> pd.DataFrame:
    fa = pd.read_csv(ROOT / "data/raw/ny_ivw2-k53g.csv")
    fa["infant"] = fa["Patient Age Group"].astype(str) == "0"
    sp = (fa.assign(inf_n=fa["Number of Discharges"] * fa.infant)
            .groupby(["Facility ID", "Discharge Year"])
            .agg(sparcs=("Number of Discharges", "sum"), sparcs_infant=("inf_n", "sum")).reset_index())
    sp = sp.rename(columns={"Facility ID": "pfi", "Discharge Year": "year"})
    xw = pd.read_csv(ROOT / "data/interim/ny_pfi_ccn_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok][["pfi", "ccn"]].drop_duplicates()
    sp = sp.merge(xw, on="pfi")
    sp = sp.groupby(["ccn", "year"]).agg(sparcs=("sparcs", "sum"), sparcs_infant=("sparcs_infant", "sum"),
                                         n_pfi=("pfi", "nunique")).reset_index()
    # HCRIS S-3 totals exclude the newborn nursery; age-0 stays approximate that exclusion
    sp["sparcs_no_infant"] = sp.sparcs - sp.sparcs_infant
    h = pd.read_parquet(ROOT / "data/interim/hcris_hospital_cy.parquet")
    h = h[(h.state == "NY") & h.year.between(2012, 2022) & (h.coverage_frac > 0.95)]
    m = h.merge(sp, on=["ccn", "year"], how="outer", indicator=True)
    m = m[m.year.between(2012, 2022)]
    m["log_ratio"] = np.log(m.sparcs / m.discharges)
    return m


if __name__ == "__main__":
    m = build()
    m.to_parquet(ROOT / "data/interim/ny_ccn_year_compare.parquet", index=False)
    both = m[(m._merge == "both") & (m.discharges > 0) & (m.sparcs > 0)].copy()
    acute = both[both.fac_type.isin(["STH", "CAH", "CH"])]

    by_year = acute.groupby("year").apply(lambda d: pd.Series({
        "hospitals": len(d), "sparcs": d.sparcs.sum(), "hcris": d.discharges.sum(),
        "agg_ratio": d.sparcs.sum() / d.discharges.sum(),
        "agg_ratio_no_infant": d.sparcs_no_infant.sum() / d.discharges.sum(),
        "median_ratio": np.exp(d.log_ratio.median()),
        "median_ratio_no_infant": np.exp(np.log(d.sparcs_no_infant / d.discharges).median()),
        "within_10pct_no_infant": (np.log(d.sparcs_no_infant / d.discharges).abs() < np.log(1.1)).mean(),
        "within_10pct": (d.log_ratio.abs() < np.log(1.1)).mean(),
        "within_25pct": (d.log_ratio.abs() < np.log(1.25)).mean(),
        "off_2x": (d.log_ratio.abs() > np.log(2)).mean(),
        "corr_log": np.corrcoef(np.log(d.sparcs), np.log(d.discharges))[0, 1],
    }), include_groups=False)
    print(by_year.round(3).to_string())

    only_h = m[(m._merge == "left_only") & m.fac_type.isin(["STH", "CAH", "CH"])]
    print("\nHCRIS acute hospital-years with no SPARCS match:", len(only_h), "hospitals:", only_h.ccn.nunique())
    print(only_h.groupby(["ccn", "name"]).discharges.mean().sort_values(ascending=False).head(15).round(0))

    per_h = (acute.groupby(["ccn", "name"])
                  .agg(years=("year", "nunique"), med_log=("log_ratio", "median"), sd_log=("log_ratio", "std"),
                       hcris_mean=("discharges", "mean"), sparcs_mean=("sparcs", "mean"))
                  .reset_index())
    per_h["median_ratio"] = np.exp(per_h.med_log)
    per_h["unstable"] = per_h.sd_log > np.log(1.25)
    print("\nMost discordant hospitals (median ratio SPARCS/HCRIS):")
    print(per_h.assign(d=per_h.med_log.abs()).sort_values("d", ascending=False)
               .head(15)[["name", "years", "median_ratio", "sd_log", "hcris_mean", "sparcs_mean"]].round(2).to_string())
    print("\nHospitals with year-to-year unstable ratio (sd log > log 1.25):", per_h.unstable.sum(), "of", len(per_h))

    # Diagnostic plot 1: hospital-level agreement, 2019
    d = acute[acute.year == 2019]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = [50, 2e5]
    ax.fill_between(lim, [lim[0] / 1.25, lim[1] / 1.25], [lim[0] * 1.25, lim[1] * 1.25], color="#e6e5e0", lw=0, label="±25%")
    ax.plot(lim, lim, color=GRAY, lw=1)
    ax.scatter(d.discharges, d.sparcs, s=22, color=BLUE, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="CMS cost report discharges (HCRIS)",
           ylabel="SPARCS discharges")
    ax.set_title("NY hospitals, 2019: SPARCS vs CMS cost reports", loc="left", color=INK, fontsize=11)
    for r in d.assign(a=d.log_ratio.abs()).nlargest(5, "a").itertuples():
        ax.annotate(r.name.title()[:28], (r.discharges, r.sparcs), fontsize=7, color=INK2,
                    xytext=(4, 4), textcoords="offset points")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "ny_sparcs_vs_hcris_2019.png"); plt.close(fig)

    # Diagnostic plot 2: distribution of the SPARCS/HCRIS ratio by year
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    yrs = sorted(acute.year.unique())
    ax.boxplot([np.exp(acute[acute.year == y].log_ratio) for y in yrs], positions=yrs, widths=0.55,
               showfliers=True, patch_artist=True,
               boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE), medianprops=dict(color=INK, lw=1.5),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
               flierprops=dict(marker="o", markersize=3, markerfacecolor=GRAY, markeredgecolor="none"))
    ax.axhline(1, color=GRAY, lw=1)
    ax.plot(yrs, by_year.agg_ratio, color=ORANGE, lw=2, marker="o", ms=5, label="Statewide ratio (sum/sum)")
    ax.set(yscale="log", ylim=(0.3, 3), ylabel="SPARCS ÷ HCRIS discharges")
    ax.set_yticks([0.5, 0.8, 1, 1.25, 2]); ax.set_yticklabels(["0.5", "0.8", "1", "1.25", "2"])
    ax.set_title("Hospital-level agreement by year (acute NY hospitals)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "ny_ratio_by_year.png"); plt.close(fig)

    per_h.to_csv(ROOT / "data/interim/ny_hospital_discordance.csv", index=False)
    by_year.to_csv(ROOT / "data/interim/ny_external_validity_by_year.csv")
