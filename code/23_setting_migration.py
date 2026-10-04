"""Where Medicare FFS spine, cranial, and joint replacement procedures are performed: CMS Physician/Supplier Procedure Summary (PSPS).

Counts professional services by CPT and place of service (21 inpatient hospital; 19/22 outpatient hospital;
24 ambulatory surgical center; other), 2013-2024, excluding assistant-at-surgery modifiers (80/81/82/AS)
and co-surgeon claims (62) so each operation is counted once.
The public PSPS files blank service counts in small cells but keep payment amounts. Blank counts are
imputed as blank-row NCH payment / (payment per service in non-blank rows of the same year x CPT x setting);
the imputed share is reported.
Tests the clinical expectation that anterior cervical fusion moved to outpatient settings while posterior
cervical fusion did not.
Output: data/interim/setting_migration.csv
"""
import re

import duckdb
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
FAMILIES = {
    "ACDF (22551)": ["22551"],
    "Anterior cervical interbody, other (22554)": ["22554"],
    "Cervical disc arthroplasty (22856)": ["22856"],
    "Posterior cervical fusion (22600)": ["22600"],
    "C1-C2 posterior fusion (22595)": ["22595"],
    "Cervical laminectomy/laminoplasty (63045, 63050, 63051)": ["63045", "63050", "63051"],
    "Lumbar interbody fusion, posterior/TLIF (22630, 22633)": ["22630", "22633"],
    "Lumbar interbody fusion, additional (22632, 22634)": ["22632", "22634"],
    "Lumbar posterolateral fusion (22612)": ["22612"],
    "Anterior lumbar interbody fusion (22558)": ["22558"],
    "Lumbar discectomy (63030)": ["63030"],
    "Lumbar laminectomy (63047)": ["63047"],
    "Kyphoplasty (22513, 22514)": ["22513", "22514"],
    "Craniotomy for tumor (61510, 61512, 61518)": ["61510", "61512", "61518"],
    "Burr hole/craniotomy for hematoma (61154, 61312)": ["61154", "61312"],
    "Ventriculoperitoneal shunt (62223)": ["62223"],
    "Mechanical thrombectomy (61645)": ["61645"],
    # comparator line: knee left the Inpatient-Only list for CY2018, hip for CY2020; both added to the ASC list 2020-2021
    "Total knee arthroplasty (27447)": ["27447"],
    "Total hip arthroplasty (27130)": ["27130"],
}


def colmap(cols):
    """PSPS column names gained a PSPS_ prefix in later releases."""
    want = ["HCPCS_CD", "TYPE_OF_SERVICE_CD", "HCPCS_INITIAL_MODIFIER_CD", "HCPCS_SECOND_MODIFIER_CD", "PLACE_OF_SERVICE_CD",
            "SUBMITTED_SERVICE_CNT", "DENIED_SERVICES_CNT", "NCH_PAYMENT_AMT"]
    out = {}
    for w in want:
        hit = next((c for c in cols if c.upper() in (w, "PSPS_" + w, "PSPS_HCPCS_" + w)), None)
        if hit is None:
            hit = next((c for c in cols if c.upper().endswith(w)), None)
        out[w] = hit
    return out


if __name__ == "__main__":
    codes = sorted({c for v in FAMILIES.values() for c in v})
    code_list = ",".join(f"'{c}'" for c in codes)
    con = duckdb.connect()
    frames = []
    for f in sorted((ROOT / "data/raw/cms_psps").glob("*.csv")):
        year = int(re.search(r"(20\d\d)", f.name).group(1))
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_csv('{f}', all_varchar=true, header=true, sample_size=2000)").fetchall()]
        c = colmap(cols)
        missing = [k for k, v in c.items() if v is None]
        if missing:
            print(f.name, "missing columns", missing, cols[:25])
            continue
        q = f"""
          SELECT {year} AS year, "{c['HCPCS_CD']}" AS cpt,
                 CASE WHEN "{c['PLACE_OF_SERVICE_CD']}" = '21' THEN 'inpatient'
                      WHEN "{c['PLACE_OF_SERVICE_CD']}" IN ('22','19') THEN 'hospital outpatient'
                      WHEN "{c['PLACE_OF_SERVICE_CD']}" = '24' THEN 'ASC' ELSE 'other' END AS setting,
                 TRY_CAST("{c['SUBMITTED_SERVICE_CNT']}" AS DOUBLE) - coalesce(TRY_CAST("{c['DENIED_SERVICES_CNT']}" AS DOUBLE), 0) AS svc,
                 TRY_CAST("{c['NCH_PAYMENT_AMT']}" AS DOUBLE) AS pay
          FROM read_csv('{f}', all_varchar=true, header=true, ignore_errors=true)
          WHERE "{c['HCPCS_CD']}" IN ({code_list})
            AND coalesce("{c['HCPCS_INITIAL_MODIFIER_CD']}", '~') NOT IN ('80','81','82','AS','62')
            AND coalesce("{c['HCPCS_SECOND_MODIFIER_CD']}", '~') NOT IN ('80','81','82','AS','62')
            AND "{c['TYPE_OF_SERVICE_CD']}" = '2'"""
        d = con.execute(q).df()
        g = d.groupby(["year", "cpt", "setting"]).apply(lambda x: pd.Series({
            "svc_reported": x.svc.sum(skipna=True),
            "pay_reported_rows": x.loc[x.svc.notna(), "pay"].sum(),
            "pay_blank_rows": x.loc[x.svc.isna(), "pay"].sum()}), include_groups=False).reset_index()
        frames.append(g)
        print("read", f.name, len(d), flush=True)
    d = pd.concat(frames)
    rate = d.pay_reported_rows / d.svc_reported.where(d.svc_reported > 0)
    rate = rate.where(rate > 0)          # zero-payment reported rows give no usable payment-per-service rate
    d["svc_imputed"] = (d.pay_blank_rows / rate).fillna(0)
    d["services"] = d.svc_reported + d.svc_imputed
    fam = {c: k for k, v in FAMILIES.items() for c in v}
    d["family"] = d.cpt.map(fam)
    out = d.groupby(["year", "family", "setting"])[["services", "svc_reported", "svc_imputed"]].sum().reset_index()
    out["share"] = out.services / out.groupby(["year", "family"]).services.transform("sum")
    out.to_csv(I / "setting_migration.csv", index=False)
    d.drop(columns=["family"]).to_csv(I / "setting_migration_cpt.csv", index=False)   # per-CPT detail (IPO policy analysis)

    pd.set_option("display.width", 260)
    wide = out.pivot_table(index=["family", "year"], columns="setting", values="services", aggfunc="sum").fillna(0)
    wide["total"] = wide.sum(axis=1)
    wide["inpatient_share"] = wide["inpatient"] / wide["total"]
    wide["imputed_share"] = out.groupby(["family", "year"]).svc_imputed.sum() / out.groupby(["family", "year"]).services.sum()
    print(wide.loc[(slice(None), [2013, 2016, 2019, 2022, 2024]), :].round(3).to_string())
