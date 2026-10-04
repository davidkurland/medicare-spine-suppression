"""Does population aging explain per-enrollee trends? Indirect age standardization.

Age-specific Medicare inpatient rates (FFS + MA, all Medicare beneficiaries) come from Texas PUDF records
2016-2019 (PAT_AGE 5-year bands) divided by Texas Medicare enrollment in the matching CMS enrollment bands.
Expected national discharges per beneficiary in year y = sum_a rate_a x US share of beneficiaries in band a (year y).
The expected index isolates the effect of changing Medicare age composition (including the shrinking under-65
disabled share). Assumes FFS enrollees share the total Medicare age mix (FFS-specific age bands are not published).
Age-adjusted change in complete FFS discharges per FFS enrollee = observed change / expected change.
Texas PAT_AGE: 00-07 <25; 08-11 25-44; 12-15 45-64; 16 65-69 ... 20 85-89; 21 90+;
masked HIV/substance bands 22 0-17, 23 18-44, 24 45-64, 25 65-74, 26 75+ (allocated to the nearest enrollment band).
Outputs: data/interim/age_standardization_rates.csv, age_standardization.csv, age_standardization_adjusted.csv
"""
import importlib

import duckdb
import pandas as pd

from common import DRG_TO_CAT, MSDRG_GROUP, ROOT

tx = importlib.import_module("26_texas")
I = ROOT / "data/interim"
BANDS = ["AGE_LT_25_BENES", "AGE_25_TO_44_BENES", "AGE_45_TO_64_BENES", "AGE_65_TO_69_BENES", "AGE_70_TO_74_BENES",
         "AGE_75_TO_79_BENES", "AGE_80_TO_84_BENES", "AGE_85_TO_89_BENES", "AGE_90_PLUS"]
CODE_TO_BAND = {**{c: "AGE_LT_25_BENES" for c in range(0, 8)}, **{c: "AGE_25_TO_44_BENES" for c in range(8, 12)},
                **{c: "AGE_45_TO_64_BENES" for c in range(12, 16)}, 16: "AGE_65_TO_69_BENES", 17: "AGE_70_TO_74_BENES",
                18: "AGE_75_TO_79_BENES", 19: "AGE_80_TO_84_BENES", 20: "AGE_85_TO_89_BENES", 21: "AGE_90_PLUS",
                22: "AGE_LT_25_BENES", 23: "AGE_25_TO_44_BENES", 24: "AGE_45_TO_64_BENES", 25: "AGE_65_TO_69_BENES", 26: "AGE_75_TO_79_BENES"}


def enrollment():
    e = pd.read_csv(ROOT / "data/raw/cms_enrollment/medicare_monthly_enrollment.csv", dtype=str)
    e = e[(e.MONTH == "Year") & e.BENE_GEO_LVL.isin(["National", "State"]) & ((e.BENE_GEO_LVL == "National") | (e.BENE_COUNTY_DESC == "Total"))]
    for c in BANDS[:-1] + ["AGE_90_TO_94_BENES", "AGE_GT_94_BENES", "TOT_BENES"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    e["AGE_90_PLUS"] = e.AGE_90_TO_94_BENES + e.AGE_GT_94_BENES
    e["year"] = e.YEAR.astype(int)
    e["geo"] = e.BENE_STATE_ABRVTN.where(e.BENE_GEO_LVL == "State", "US")
    return e[["geo", "year"] + BANDS + ["TOT_BENES"]]


if __name__ == "__main__":
    con = duckdb.connect()
    parts = []
    for f in tx.base_files():
        parts.append(con.execute(f"""
          SELECT left(DISCHARGE, 4)::INT AS year, TRY_CAST(PAT_AGE AS INT) AS age_code, TRY_CAST(MS_DRG AS INT) AS msdrg, count(*) AS n
          FROM read_csv('{f}', delim='\t', header=true, all_varchar=true, quote='', escape='', strict_mode=false, null_padding=true)
          WHERE FIRST_PAYMENT_SRC IN ('MA', 'MB', '16') AND left(DISCHARGE, 4)::INT BETWEEN 2016 AND 2019
          GROUP BY ALL""").df())
    d = pd.concat(parts)
    d["category"] = d.msdrg.map(DRG_TO_CAT)
    d["group"] = d.category.map(MSDRG_GROUP)
    d["band"] = d.age_code.map(CODE_TO_BAND)
    d = d.dropna(subset=["group", "band"])
    strata = {"spine": d.group == "spine", "spine_fusion": (d.group == "spine") & ~d.category.fillna("").str.contains("nonfusion"),
              "cranial": d.group == "cranial"}
    en = enrollment()
    tx_en = en[(en.geo == "TX") & en.year.between(2016, 2019)][BANDS].sum()
    us = en[en.geo == "US"].set_index("year").sort_index()
    us_share = us[BANDS].div(us[BANDS].sum(axis=1), axis=0)

    rates, rows = [], []
    dec = pd.read_csv(I / "suppression_exact_national.csv")
    ns = pd.read_csv(I / "national_sensitivity.csv").set_index("year")
    for s, mask in strata.items():
        num = d[mask].groupby("band").n.sum().reindex(BANDS).fillna(0)
        r = num / tx_en
        rates.append(pd.DataFrame({"stratum": s, "band": BANDS, "discharges": num.values, "benes": tx_en.values, "rate_per_100k": (r * 1e5).values}))
        expected = (us_share * r).sum(axis=1)          # expected discharges per beneficiary, by year
        for y in us_share.index:
            rows.append({"stratum": s, "year": y, "expected_per_100k": expected.loc[y] * 1e5,
                         "expected_index_2013": expected.loc[y] / expected.loc[2013] * 100})
    rates = pd.concat(rates)
    rates.to_csv(I / "age_standardization_rates.csv", index=False)
    out = pd.DataFrame(rows)

    g = dec[dec.level == "group"].pivot_table(index="year", columns="stratum", values="total_ffs")
    per = g.div(ns.ffs_part_a_benes, axis=0)
    adj = []
    for s in ("spine", "cranial"):
        e = out[out.stratum == s].set_index("year").expected_per_100k
        for y0, y1 in [(2013, 2024), (2013, 2019), (2019, 2024)]:
            obs = per.loc[y1, s] / per.loc[y0, s] - 1
            exp_ = e.loc[y1] / e.loc[y0] - 1
            adj.append({"stratum": s, "window": f"{y0}-{y1}", "observed_per_enrollee_change": obs,
                        "expected_change_from_age_mix": exp_, "age_adjusted_change": (1 + obs) / (1 + exp_) - 1})
    adj = pd.DataFrame(adj)
    out.to_csv(I / "age_standardization.csv", index=False)
    adj.to_csv(I / "age_standardization_adjusted.csv", index=False)
    pd.set_option("display.width", 220)
    print(rates.pivot_table(index="band", columns="stratum", values="rate_per_100k", sort=False).round(1).to_string())
    print(out.pivot_table(index="year", columns="stratum", values="expected_index_2013").round(1).to_string())
    print(adj.round(3).to_string(index=False))
