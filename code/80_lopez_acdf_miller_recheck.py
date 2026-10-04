"""Complete PSPS counts for the series in two published Medicare spine volume studies.

Lopez CD et al, Spine J 2020;20(11):1737-1743. ACDF from the Part B Physician and Other Supplier
public use file, a provider-level file that removes cells under 11 services. Reported a 24.2% rise in
annual volume, 38 877 in 2012 to 48 290 in 2017. The provider-level file series is rebuilt in
81_lopez_acdf_provider_file.py; this script computes the complete PSPS count for the same codes and years.

Miller AK et al, N Am Spine Soc J 2024;17:100314. Ambulatory spine surgery from PSPS, HCPCS 22010-22899
and 62380-63103, reported as raw counts with no population denominator across 2010-2021. Because the
fee-for-service pool shrank over that window, raw counts understate growth per beneficiary.

Output: data/interim/lopez_acdf_miller_recheck.csv
"""
import importlib
import re

import duckdb
import pandas as pd

from common import ROOT

s23 = importlib.import_module("23_setting_migration")
s79 = importlib.import_module("79_bariatric_denominator_check")

LOPEZ_ACDF = ["22551", "22552", "22554", "22585"]


def psps_by_setting(where_code, label):
    """National Part B services by year and setting, with the blank-small-cell imputation."""
    con = duckdb.connect()
    frames = []
    for f in sorted((ROOT / "data/raw/cms_psps").glob("*.csv")):
        year = int(re.search(r"(20\d\d)", f.name).group(1))
        cols = [r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_csv('{f}', all_varchar=true, header=true, sample_size=2000)").fetchall()]
        c = s23.colmap(cols)
        if any(v is None for v in c.values()):
            continue
        q = f"""
          SELECT {year} AS year,
                 CASE WHEN "{c['PLACE_OF_SERVICE_CD']}" = '21' THEN 'inpatient' ELSE 'outpatient/other' END AS setting,
                 TRY_CAST("{c['SUBMITTED_SERVICE_CNT']}" AS DOUBLE)
                   - coalesce(TRY_CAST("{c['DENIED_SERVICES_CNT']}" AS DOUBLE), 0) AS svc,
                 TRY_CAST("{c['NCH_PAYMENT_AMT']}" AS DOUBLE) AS pay
          FROM read_csv('{f}', all_varchar=true, header=true, ignore_errors=true)
          WHERE {where_code}
            AND coalesce("{c['HCPCS_INITIAL_MODIFIER_CD']}", '~') NOT IN ('80','81','82','AS','62')
            AND coalesce("{c['HCPCS_SECOND_MODIFIER_CD']}", '~') NOT IN ('80','81','82','AS','62')
            AND "{c['TYPE_OF_SERVICE_CD']}" = '2'"""
        d = con.execute(q).df()
        g = d.groupby(["year", "setting"]).apply(lambda x: pd.Series({
            "svc_reported": x.svc.sum(skipna=True),
            "pay_reported_rows": x.loc[x.svc.notna(), "pay"].sum(),
            "pay_blank_rows": x.loc[x.svc.isna(), "pay"].sum()}), include_groups=False).reset_index()
        frames.append(g)
        print("read", f.name, label, flush=True)
    d = pd.concat(frames)
    rate = (d.pay_reported_rows / d.svc_reported.where(d.svc_reported > 0)).where(lambda s: s > 0)
    d["services"] = d.svc_reported + (d.pay_blank_rows / rate).fillna(0)
    d["study"] = label
    return d


if __name__ == "__main__":
    con = duckdb.connect()
    # resolve the HCPCS column name once for the range predicate
    f0 = sorted((ROOT / "data/raw/cms_psps").glob("*.csv"))[0]
    cols = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_csv('{f0}', all_varchar=true, header=true, sample_size=2000)").fetchall()]
    hc = s23.colmap(cols)["HCPCS_CD"]

    lopez = psps_by_setting('"' + hc + '" IN (' + ",".join(f"'{c}'" for c in LOPEZ_ACDF) + ')', "Lopez ACDF")
    miller = psps_by_setting(
        f'(("{hc}" >= \'22010\' AND "{hc}" <= \'22899\') OR ("{hc}" >= \'62380\' AND "{hc}" <= \'63103\'))'
        f' AND "{hc}" ~ \'^[0-9]{{5}}$\'', "Miller ambulatory spine")

    out = pd.concat([lopez, miller])
    enr = s79.ffs_enrollment()
    tot = out.groupby(["study", "year"])["services"].sum().reset_index()
    tot["ffs_partb"] = tot.year.map(enr.B_ORGNL_MDCR_BENES)
    tot["per_100k"] = tot.services / tot.ffs_partb * 1e5
    by_set = out.groupby(["study", "year", "setting"])["services"].sum().reset_index()
    by_set.to_csv(ROOT / "data/interim/lopez_acdf_miller_recheck.csv", index=False)

    pd.set_option("display.width", 200)
    for study, w in tot.groupby("study"):
        w = w.set_index("year")
        print(f"\n=== {study} ===")
        print(w[["services", "per_100k"]].round(1).to_string())
        for a, b in [(2013, 2017), (2013, 2021), (2013, 2024)]:
            if a in w.index and b in w.index:
                print(f"  {a}->{b}: complete services {100*(w.services[b]/w.services[a]-1):+.1f}%"
                      f"   per enrollee {100*(w.per_100k[b]/w.per_100k[a]-1):+.1f}%")
    print("\n=== Miller: inpatient vs outpatient complete services ===")
    m = by_set[by_set.study == "Miller ambulatory spine"].pivot_table(index="year", columns="setting", values="services")
    m["outpatient_share_%"] = 100 * m["outpatient/other"] / m.sum(axis=1)
    print(m.round(0).to_string())
