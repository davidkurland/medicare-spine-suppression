"""Definition check for the multistate headline (HCUPnet CCSR MST013 all-listed vs CMS fusion MS-DRGs).

HCUPnet counts Medicare (FFS + MA, all ages) discharges with ANY listed spine fusion procedure (CCSR MST013);
CMS counts FFS discharges grouped to fusion MS-DRGs. Texas PUDF 2016-2019 records carry the principal and 24
secondary ICD-10-PCS codes, MS-DRG, payer and age, so both definitions can be built on the same records:
  1. validate the MST013 emulation and DRG 460 against HCUPnet's Texas Medicare counts;
  2. overlap: share of MST013 discharges in fusion DRGs and vice versa, by payer;
  3. trend in the MST013 : fusion-DRG ratio, and in the under-65 share of Medicare fusion.
MST013 emulation (ICD-10-PCS fusion roots): 0RG0-0RG2, 0RG4, 0RG6-0RG8, 0RGA (cervical/thoracic/thoracolumbar),
0SG0, 0SG1, 0SG3 (lumbar/lumbosacral). Aggregate reporting only (Texas DUA).
Outputs: data/interim/texas/tx_definition_check.csv, tx_definition_summary.csv. These aggregate counts can
include cells of 1-10; they stay local under data/interim (git-ignored), and any count of 1-10 is suppressed before it
is shown or shared (Texas data use agreement).
"""
import importlib

import duckdb
import pandas as pd

from common import MSDRG_GROUP, DRG_TO_CAT, ROOT

tx = importlib.import_module("26_texas")
I = ROOT / "data/interim"
OUT = I / "texas"
FUSION_DRGS = sorted(d for d, c in DRG_TO_CAT.items() if MSDRG_GROUP.get(c) == "spine" and "fusion" in c and "nonfusion" not in c)
# HCUPnet Texas Medicare discharges (expected payer = Medicare; CCSR MST013 all-listed and MS-DRG 460), as in
# data/raw/hcupnet/hcupnet_long.csv (28_hcupnet.py)
HCUP_TX_MEDICARE = {"MST013": {2016: 15495, 2017: 14844, 2018: 15453, 2019: 15776},
                    "DRG460": {2016: 6807, 2017: 6672, 2018: 5219, 2019: 4751}}

if __name__ == "__main__":
    print("fusion DRGs:", FUSION_DRGS)
    con = duckdb.connect()
    codes = ["PRINC_SURG_PROC_CODE"] + [f"OTH_SURG_PROC_CODE_{i}" for i in range(1, 25)]
    allc = "concat_ws('|', " + ", ".join(f"coalesce(upper(trim({c})), '')" for c in codes) + ")"
    parts = []
    for f in tx.base_files():
        q = f"""
          WITH b AS (
            SELECT left(DISCHARGE, 4)::INT AS year, TRY_CAST(MS_DRG AS INT) AS msdrg,
                   CASE WHEN FIRST_PAYMENT_SRC IN ('MA','MB') THEN 'ffs' WHEN FIRST_PAYMENT_SRC = '16' THEN 'ma' ELSE 'other' END AS payer,
                   CASE WHEN TRY_CAST(PAT_AGE AS INT) BETWEEN 16 AND 21 OR TRY_CAST(PAT_AGE AS INT) IN (25, 26) THEN '65+' ELSE '<65' END AS age,
                   {allc} AS pcs
            FROM read_csv('{f}', delim='\t', header=true, all_varchar=true, quote='', escape='', strict_mode=false, null_padding=true))
          SELECT year, payer, age, msdrg,
                 regexp_matches(pcs, '(^|\\|)(0RG[0124678A]|0SG[013])') AS mst013,
                 msdrg IN ({",".join(map(str, FUSION_DRGS))}) AS fusion_drg,
                 msdrg = 460 AS drg460,
                 count(*) AS n
          FROM b WHERE year BETWEEN 2016 AND 2019 GROUP BY ALL"""
        parts.append(con.execute(q).df())
        print("read", f.split("/")[-1], flush=True)
    d = pd.concat(parts).groupby(["year", "payer", "age", "msdrg", "mst013", "fusion_drg", "drg460"], as_index=False, dropna=False).n.sum()
    d.to_csv(OUT / "tx_definition_check.csv", index=False)

    med = d[d.payer.isin(["ffs", "ma"])]
    rows = []
    for y, g in med.groupby("year"):
        ffs = g[g.payer == "ffs"]
        r = {"year": y,
             "mst013_medicare": g[g.mst013].n.sum(), "hcup_mst013": HCUP_TX_MEDICARE["MST013"][y],
             "drg460_medicare": g[g.drg460].n.sum(), "hcup_drg460": HCUP_TX_MEDICARE["DRG460"][y],
             "fusion_drg_medicare": g[g.fusion_drg].n.sum(),
             "share_mst013_in_fusion_drg": g[g.mst013 & g.fusion_drg].n.sum() / g[g.mst013].n.sum(),
             "share_fusion_drg_with_mst013": g[g.mst013 & g.fusion_drg].n.sum() / g[g.fusion_drg].n.sum(),
             "ratio_mst013_to_fusion_drg": g[g.mst013].n.sum() / g[g.fusion_drg].n.sum(),
             "ratio_mst013_to_fusion_drg_ffs": ffs[ffs.mst013].n.sum() / ffs[ffs.fusion_drg].n.sum(),
             "ratio_mst013_to_fusion_drg_ma": g[(g.payer == "ma") & g.mst013].n.sum() / g[(g.payer == "ma") & g.fusion_drg].n.sum(),
             "under65_share_mst013_medicare": g[g.mst013 & (g.age == "<65")].n.sum() / g[g.mst013].n.sum(),
             "ffs65_fusion_drg": ffs[ffs.fusion_drg & (ffs.age == "65+")].n.sum()}
        r["validation_mst013"] = r["mst013_medicare"] / r["hcup_mst013"]
        r["validation_drg460"] = r["drg460_medicare"] / r["hcup_drg460"]
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "tx_definition_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).T.to_string())
    # what the definition mismatch alone would do to a 2016-2019 comparison
    a, b = s.set_index("year").loc[2016], s.set_index("year").loc[2019]
    print("2016-2019 change  MST013 Medicare:", round(b.mst013_medicare / a.mst013_medicare - 1, 3),
          " fusion-DRG Medicare:", round(b.fusion_drg_medicare / a.fusion_drg_medicare - 1, 3),
          " HCUPnet MST013:", round(b.hcup_mst013 / a.hcup_mst013 - 1, 3))
