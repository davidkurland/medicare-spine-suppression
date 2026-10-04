"""Indiana: hospital-year discharge counts (IDOH APR-DRG PUF aggregates, 2017-2024) vs CMS HCRIS.

Newborns are excluded with APR-DRG 580-640 (MDC 15) because HCRIS S-3 totals exclude the nursery.
Outputs: data/interim/state_compare/IN_hospital_year.parquet, IN_by_year.csv, IN_crosswalk.csv
"""
import re

import pandas as pd

from common import ROOT, agreement, hcris_state, match_to_hcris

RAW = ROOT / "data/raw/states/IN"
OUT = ROOT / "data/interim/state_compare"
OUT.mkdir(parents=True, exist_ok=True)


def load_counts() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("*_APRDRG_IN*.csv")):
        y = int(f.name[:4])
        d = pd.read_csv(f)
        d.columns = [c.upper() for c in d.columns]
        d["newborn"] = d.APRDRG.between(580, 640)
        g = d.groupby("HOSPITAL_ID").apply(lambda x: pd.Series({
            "sp": x.PATS.sum(), "sp_newborn": x.loc[x.newborn, "PATS"].sum()}), include_groups=False)
        frames.append(g.reset_index().assign(year=y))
    c = pd.concat(frames).rename(columns={"HOSPITAL_ID": "id"})
    c["sp_all"] = c.sp
    c["sp"] = c.sp_all - c.sp_newborn
    return c


def load_keys() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("*_Hosp*_ID*.xlsx")) + sorted(RAW.glob("*_HOSPITAL-ID*.xlsx")):
        y = int(f.name[:4])
        d = pd.read_excel(f, sheet_name=0)
        idc = next(c for c in d.columns if re.search(r"id|num", c, re.I))
        nmc = next(c for c in d.columns if re.search(r"name", c, re.I))
        ctc = next(c for c in d.columns if re.search(r"city", c, re.I))
        frames.append(d[[idc, nmc, ctc]].set_axis(["id", "name", "city"], axis=1).assign(year=y))
    return pd.concat(frames).dropna(subset=["id"]).astype({"id": int})


if __name__ == "__main__":
    counts, keys = load_counts(), load_keys()
    latest = keys.sort_values("year").drop_duplicates("id", keep="last")
    xw = match_to_hcris(latest[["id", "name", "city"]], "IN")
    # hand-verified links (specialty orthopaedic hospitals and multi-campus systems the fuzzy match gets wrong)
    ov_path = ROOT / "data/manual/IN_crosswalk_overrides.csv"
    if ov_path.exists():
        ov = pd.read_csv(ov_path, dtype={"ccn": str})
        xw = xw[~xw.id.isin(ov.id)]
        ov = ov.merge(latest[["id", "name", "city"]], on="id", how="left").assign(match_ok=True, name_score=None, same_city=None)
        xw = pd.concat([xw, ov], ignore_index=True)
        print(f"applied {len(ov)} manual overrides")
    xw.to_csv(OUT / "IN_crosswalk.csv", index=False)
    share = counts.merge(xw, on="id").query("match_ok").sp.sum() / counts.sp.sum()
    print(f"IN crosswalk: {xw.match_ok.sum()}/{len(xw)} ids matched, {share:.1%} of discharges")
    print(xw[~xw.match_ok].sort_values("name_score").to_string())

    m = counts.merge(xw[xw.match_ok][["id", "ccn"]], on="id")
    m = m.groupby(["ccn", "year"]).agg(sp=("sp", "sum"), sp_all=("sp_all", "sum"), n_ids=("id", "nunique")).reset_index()
    h = hcris_state("IN", range(2017, 2023))
    h = h[h.coverage_frac > 0.95]
    both = m.merge(h, on=["ccn", "year"])
    both.to_parquet(OUT / "IN_hospital_year.parquet", index=False)
    acute = both[both.fac_type.isin(["STH", "CAH", "CH"])]
    by_year = agreement(acute)
    by_year.to_csv(OUT / "IN_by_year.csv")
    print(by_year.round(3).to_string())
