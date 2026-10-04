"""What the suppressed CMS spine cells actually hold: exact totals and Indiana ground truth.

Three quantities:
  1. hidden spine volume nationally and by state is exact arithmetic (geography totals minus published rows)
  2. Indiana hospital-level truth shows where those hidden cases sit (state data reveal the 1-10 counts)
  3. what a naive fill of 0, 5 or 10 would get wrong, which is the implicit alternative
Output: data/interim/suppressed_cells/ground_truth.json
"""
import json

import numpy as np
import pandas as pd

from common import ROOT

OUT = ROOT / "data/interim/suppressed_cells"
KMAX = 10

cells = pd.read_parquet(OUT / "cells.parquet")
nat = pd.read_csv(OUT / "nat_drg_year.csv")
st = pd.read_csv(OUT / "state_drg_year.csv")
truth = pd.read_csv(OUT / "indiana_truth.csv", dtype={"ccn": str})

res = {}
ny = nat.groupby("year")[["complete", "published"]].sum()
ny["hidden"] = ny.complete - ny.published
ny["hidden_share"] = ny.hidden / ny.complete
res["national"] = {"by_year": ny.round(4).reset_index().to_dict("records"),
                   "note": "exact: complete geography totals minus published provider rows"}
sy = st.groupby("year").complete_state.sum()
cov = pd.DataFrame({"national_complete": ny.complete, "sum_published_state_rows": sy})
cov["not_covered_by_state_rows"] = cov.national_complete - cov.sum_published_state_rows
cov["share_not_covered"] = cov.not_covered_by_state_rows / cov.national_complete
res["state_coverage"] = cov.round(4).reset_index().to_dict("records")

# Indiana, 2021-2024: the years its payer field separates fee-for-service from Medicare Advantage
ind = cells[(cells.state == "IN") & (cells.year >= 2021) & (~cells.observed)].merge(
    truth[truth.year >= 2021][["year", "ccn", "drg", "state_medicare"]],
    on=["year", "ccn", "drg"], how="left").fillna({"state_medicare": 0})
y = np.clip(ind.state_medicare.values, 0, KMAX)
band = pd.cut(ind.hosp_spine_published, [-1, 0, 10, 50, 200, 1e9],
              labels=["0", "1-10", "11-50", "51-200", "200+"])
res["indiana_2021_2024"] = {
    "suppressed_cells": int(len(ind)), "true_hidden_cases": int(y.sum()),
    "mean_per_suppressed_cell": float(y.mean()),
    "share_of_cells_with_zero": float((y == 0).mean()),
    "by_stratum": ind.assign(t=y).groupby("stratum").t.agg(["size", "mean", "sum"]).round(3).reset_index().to_dict("records"),
    "by_published_volume_band": ind.assign(t=y, band=band).groupby("band", observed=True).t.agg(
        ["size", "mean", "sum"]).round(3).reset_index().rename(columns={"band": "hospital_published_spine_that_year"}).to_dict("records"),
    "share_of_hidden_cases_at_hospitals_with_no_published_spine_row":
        float(ind.assign(t=y).loc[ind.hosp_spine_published == 0, "t"].sum() / y.sum()),
    "naive_fills": {n: {"bias_per_cell": float(v - y.mean()), "mae_per_cell": float(np.abs(v - y).mean()),
                        "total_predicted": float(v * len(y)), "total_true": float(y.sum())}
                    for n, v in [("zero", 0.0), ("five", 5.0), ("ten", 10.0)]},
}
json.dump(res, open(OUT / "ground_truth.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res["indiana_2021_2024"].items() if k != "by_stratum"}, indent=1)[:1200])
print("national hidden share 2013 / 2024:",
      round(ny.hidden_share.iloc[0], 3), "/", round(ny.hidden_share.iloc[-1], 3))
