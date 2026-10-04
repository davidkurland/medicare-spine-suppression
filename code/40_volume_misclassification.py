"""Hospital volume misclassification: what a researcher using CMS's public FFS file would conclude about which
hospitals are high volume, vs all-payer state records. New York SPARCS, spine (APR-DRG spine definition), 2013-2024.

Inputs: data/interim/ny_rates_rankings/rank_table.csv (hospital x year: all_payer, medicare [incl. MA], cms_ffs published,
cms_status). Hospitals are the CCN-consolidated units used in the NY analysis.
Definitions per year:
  high volume (truth)  = all-payer spine discharges >= threshold (50, 100, 200 per year) or top quartile
  high volume (CMS)    = hospital among the same number of top hospitals ranked by published CMS FFS discharges
  invisible            = hospital with >= threshold all-payer cases but no published CMS spine row
Reports: sensitivity of CMS-based classification, share of all-payer high-volume hospitals invisible in CMS,
share of all-payer spine patients treated at hospitals CMS shows as absent or not high volume, Cohen's kappa.
Output: data/interim/volume_misclassification_ny.csv
"""
import numpy as np
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def misclassification(r):
    """r: hospital-year rows with columns year, all_payer, cms_ffs (published CMS FFS; 0 if no published row)."""
    rows = []
    for y, g in r.groupby("year"):
        g = g.sort_values("all_payer", ascending=False)
        rules = {f">={t}/yr": g.all_payer >= t for t in (50, 100, 200)}
        rules["top quartile"] = g.all_payer >= g.all_payer.quantile(0.75)
        for name, truth in rules.items():
            k = int(truth.sum())
            if k == 0:
                continue
            cms_rank = g.cms_ffs.rank(ascending=False, method="first")
            cms_high = (cms_rank <= k) & (g.cms_ffs > 0)
            invisible = truth & (g.cms_ffs <= 0)
            missed = truth & ~cms_high
            rows.append({
                "year": y, "rule": name, "hospitals": len(g), "high_volume_hospitals": k,
                "cms_sensitivity": (truth & cms_high).sum() / k,
                "high_volume_invisible_in_cms": int(invisible.sum()),
                "high_volume_invisible_share": invisible.sum() / k,
                "high_volume_misclassified_share": missed.sum() / k,
                "patients_at_misclassified_high_volume_share": g.loc[missed, "all_payer"].sum() / g.loc[truth, "all_payer"].sum(),
                "false_high_in_cms": int((cms_high & ~truth).sum()),
                "kappa": kappa(truth, cms_high),
                "allpayer_patients_at_cms_invisible_hospitals_share": g.loc[g.cms_ffs <= 0, "all_payer"].sum() / g.all_payer.sum(),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = pd.read_csv(I / "ny_rates_rankings/rank_table.csv")
    r = r[(r.grp == "spine") & (r.all_payer > 0)].copy()
    out = misclassification(r)
    out.to_csv(I / "volume_misclassification_ny.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.year.isin([2013, 2016, 2019, 2022, 2024])].round(3).to_string(index=False))
