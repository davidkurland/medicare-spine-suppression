"""All-payer case-mix shift among inpatients: New York SPARCS 2013-2024.

As uncomplicated spine operations move outpatient, inpatients who remain should be sicker. Any inpatient-only
benchmark (NIS, inpatient NSQIP subsets, hospital inpatient quality metrics) then shows worsening outcomes
without a change in care. Tests: APR Severity of Illness (1 minor - 4 extreme), length of stay, and age mix by
APR-DRG, all payers and Medicare payer, vs a cranial control that did not migrate outpatient.
APR-DRGs: 321 cervical spinal fusion and other back/neck procedures except disc excision/decompression;
304 dorsal and lumbar fusion except for curvature; 310 intervertebral disc excision and decompression;
control 021 craniotomy except for trauma.
Input: data/raw/sparcs/sparcs_<year>.parquet (column names vary by release; APR fields are stable).
Output: data/interim/ny_severity_shift.csv
"""
import duckdb
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
APR = {321: "cervical fusion & other back/neck (321)", 304: "dorsal/lumbar fusion (304)",
       310: "disc excision/decompression (310)", 21: "craniotomy except trauma (021, control)",
       # Joint replacement is computed for reference and not plotted in Figure 4D (the 2020 elective shutdown and the
       # 2021 APR-DRG v38 split confound the NY all-payer series; the national panel, Figure 4C, carries the joint comparison).
       # Comparator line: hip (301) and knee (302) joint replacement; APR-DRG v38 (2021) split each into
       # non-elective/complex (323 hip, 325 knee) and elective (324 hip, 326 knee), folded back in for a continuous series.
       **{c: "joint replacement (301, 302; 323-326 from 2021)" for c in (301, 302, 323, 324, 325, 326)}}

if __name__ == "__main__":
    con = duckdb.connect()
    parts = []
    for y in range(2013, 2025):
        f = ROOT / f"data/raw/sparcs/sparcs_{y}.parquet"
        cols = {c[0] for c in con.execute(f"DESCRIBE SELECT * FROM '{f}'").fetchall()}
        los = '"Length of Stay"'
        d = con.execute(f"""
          SELECT {y} AS year, TRY_CAST("APR DRG Code" AS INT) AS apr, TRY_CAST("APR Severity of Illness Code" AS INT) AS soi,
                 lower(coalesce("Payment Typology 1", '')) LIKE '%medicare%' AS medicare,
                 "Age Group" AS age, TRY_CAST(replace(CAST({los} AS VARCHAR), '120 +', '120') AS DOUBLE) AS los, count(*) AS n
          FROM '{f}'
          WHERE TRY_CAST("APR DRG Code" AS INT) IN ({",".join(map(str, APR))})
          GROUP BY ALL""").df()
        parts.append(d)
        print("read", y, int(d.n.sum()), flush=True)
    d = pd.concat(parts)
    d["label"] = d.apr.map(APR)
    rows = []
    for (y, lab), g in d.groupby(["year", "label"]):
        for scope, s in [("all payers", g), ("Medicare payer", g[g.medicare])]:
            n = s.n.sum()
            if n == 0:
                continue
            rows.append({"year": y, "apr": lab, "scope": scope, "discharges": n,
                         "soi_minor_share": s.loc[s.soi == 1, "n"].sum() / n,
                         "soi_major_extreme_share": s.loc[s.soi >= 3, "n"].sum() / n,
                         "mean_soi": (s.soi * s.n).sum() / s.loc[s.soi.notna(), "n"].sum(),
                         "mean_los": (s.los * s.n).sum() / s.loc[s.los.notna(), "n"].sum()})
    out = pd.DataFrame(rows)
    out.to_csv(I / "ny_severity_shift.csv", index=False)
    pd.set_option("display.width", 250)
    for scope in ("all payers", "Medicare payer"):
        print(scope)
        print(out[out.scope == scope].pivot_table(index="year", columns="apr", values="soi_minor_share").mul(100).round(1).to_string())
        print(out[out.scope == scope].pivot_table(index="year", columns="apr", values="discharges").to_string())
