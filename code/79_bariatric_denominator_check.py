"""Does a per-fee-for-service-enrollee denominator remove the Medicare Advantage distortion?

Motivated by Howard et al, JAMA Surg 2026, which reports a 64.8% fall in bariatric surgery per
100,000 fee-for-service Medicare beneficiaries, 2014-2024, and names "migration between fee-for-service
Medicare and Medicare Advantage" as a limitation it cannot resolve.

Dividing by fee-for-service enrollment corrects for the SIZE of the shrinking fee-for-service pool. It
does not correct for its changing COMPOSITION. This script applies the negative-control design used
in the main analysis to the same idea: put procedures through an identical pipeline that share the
denominator but not the exposure. An emergent operation (appendectomy) cannot be displaced by a drug
and is close to demand-insensitive, so whatever its per-enrollee rate does is a floor estimate of what
the denominator alone contributes.

This is an illustration built on PSPS rather than a reproduction: Howard et al used the Physician and
Other Practitioners by Geography and Service file. Both are national Part B counts of the same claims.

Output: data/interim/bariatric_denominator_check.csv
"""
import importlib
import re

import duckdb
import pandas as pd

from common import ROOT

s23 = importlib.import_module("23_setting_migration")

FAMILIES = {
    "Bariatric surgery (Howard et al code set)":
        ["43644", "43645", "43770", "43775", "43843", "43845", "43846", "43847"],
    "CONTROL: appendectomy (44950, 44970)": ["44950", "44970"],
    "CONTROL: inguinal hernia repair (49505, 49650)": ["49505", "49650"],
    "CONTROL: cholecystectomy (47562, 47563, 47564)": ["47562", "47563", "47564"],
    "CONTROL: total knee arthroplasty (27447)": ["27447"],
}


def psps_counts(codes):
    """National Part B primary-surgeon service counts by year, with the blank-small-cell imputation."""
    code_list = ",".join(f"'{c}'" for c in codes)
    con = duckdb.connect()
    frames = []
    for f in sorted((ROOT / "data/raw/cms_psps").glob("*.csv")):
        year = int(re.search(r"(20\d\d)", f.name).group(1))
        cols = [r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_csv('{f}', all_varchar=true, header=true, sample_size=2000)").fetchall()]
        c = s23.colmap(cols)
        if any(v is None for v in c.values()):
            print("skip", f.name)
            continue
        q = f"""
          SELECT {year} AS year, "{c['HCPCS_CD']}" AS cpt,
                 TRY_CAST("{c['SUBMITTED_SERVICE_CNT']}" AS DOUBLE)
                   - coalesce(TRY_CAST("{c['DENIED_SERVICES_CNT']}" AS DOUBLE), 0) AS svc,
                 TRY_CAST("{c['NCH_PAYMENT_AMT']}" AS DOUBLE) AS pay
          FROM read_csv('{f}', all_varchar=true, header=true, ignore_errors=true)
          WHERE "{c['HCPCS_CD']}" IN ({code_list})
            AND coalesce("{c['HCPCS_INITIAL_MODIFIER_CD']}", '~') NOT IN ('80','81','82','AS','62')
            AND coalesce("{c['HCPCS_SECOND_MODIFIER_CD']}", '~') NOT IN ('80','81','82','AS','62')
            AND "{c['TYPE_OF_SERVICE_CD']}" = '2'"""
        d = con.execute(q).df()
        g = d.groupby(["year", "cpt"]).apply(lambda x: pd.Series({
            "svc_reported": x.svc.sum(skipna=True),
            "pay_reported_rows": x.loc[x.svc.notna(), "pay"].sum(),
            "pay_blank_rows": x.loc[x.svc.isna(), "pay"].sum()}), include_groups=False).reset_index()
        frames.append(g)
        print("read", f.name, flush=True)
    d = pd.concat(frames)
    rate = d.pay_reported_rows / d.svc_reported.where(d.svc_reported > 0)
    rate = rate.where(rate > 0)
    d["svc_imputed"] = (d.pay_blank_rows / rate).fillna(0)
    d["services"] = d.svc_reported + d.svc_imputed
    return d


def ffs_enrollment():
    e = pd.read_csv(ROOT / "data/raw/cms_enrollment/medicare_monthly_enrollment.csv", low_memory=False)
    e = e[(e.BENE_GEO_LVL.astype(str).str.lower() == "national")
          & (e.MONTH.astype(str).str.lower() == "year")].copy()
    for c in ["B_ORGNL_MDCR_BENES", "ORGNL_MDCR_BENES", "B_TOT_BENES"]:
        e[c] = pd.to_numeric(e[c], errors="coerce")
    return e.set_index("YEAR")[["B_ORGNL_MDCR_BENES", "ORGNL_MDCR_BENES", "B_TOT_BENES"]]


if __name__ == "__main__":
    codes = sorted({c for v in FAMILIES.values() for c in v})
    d = psps_counts(codes)
    fam = {c: k for k, v in FAMILIES.items() for c in v}
    d["family"] = d.cpt.map(fam)
    tot = d.groupby(["year", "family"])["services"].sum().reset_index()

    enr = ffs_enrollment()
    tot["ffs_partb"] = tot.year.map(enr.B_ORGNL_MDCR_BENES)
    tot["ffs_all"] = tot.year.map(enr.ORGNL_MDCR_BENES)
    tot["rate_per_100k"] = tot.services / tot.ffs_partb * 1e5

    tot = tot[tot.year.between(2014, 2024)].sort_values(["family", "year"])
    tot.to_csv(ROOT / "data/interim/bariatric_denominator_check.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nRate per 100 000 Part B fee-for-service enrollees\n")
    w = tot.pivot_table(index="year", columns="family", values="rate_per_100k")
    print(w.round(1).to_string())

    print("\n2014 -> 2024 change\n")
    for f in w.columns:
        a, b = w[f].loc[2014], w[f].loc[2024]
        print(f"  {f:<48} {a:8.1f} -> {b:8.1f}   {100*(b/a-1):+7.1f}%")

    print("\nDenominator itself, 2014 -> 2024")
    print(f"  Part B fee-for-service enrollees   {enr.B_ORGNL_MDCR_BENES.loc[2014]:>12,.0f}"
          f" -> {enr.B_ORGNL_MDCR_BENES.loc[2024]:>12,.0f}"
          f"   {100*(enr.B_ORGNL_MDCR_BENES.loc[2024]/enr.B_ORGNL_MDCR_BENES.loc[2014]-1):+.1f}%")
    print(f"  All Part B enrollees (FFS + MA)    {enr.B_TOT_BENES.loc[2014]:>12,.0f}"
          f" -> {enr.B_TOT_BENES.loc[2024]:>12,.0f}"
          f"   {100*(enr.B_TOT_BENES.loc[2024]/enr.B_TOT_BENES.loc[2014]-1):+.1f}%")
