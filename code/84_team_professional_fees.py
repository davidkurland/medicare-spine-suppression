"""Professional fees for the TEAM cervical simulation, from the Part B Physician/Supplier Procedure Summary (PSPS).

The anchored simulation (46, 51) leaves professional fees and readmissions out of episode spending and absorbs them
in one capture factor. This script measures the professional part: allowed amounts for the surgeon, the assistant
and co-surgeon, and anesthesia, by code and setting, for 2022-2024 (the TEAM baseline years).

Allowed amount = Medicare payment plus beneficiary cost sharing, the basis TEAM uses. In a cell under 11 services
PSPS blanks the service count AND the allowed amount and keeps only the Medicare payment. For blank rows we impute the
count from payment per service, and the allowed amount from the allowed-to-payment ratio, in the unblanked rows of
the same code, setting and line (the count rule is the one in 23_setting_migration).

Cases for a code = plain surgeon lines + co-surgeon lines / 2 (2 surgeons bill 1 operation with modifier 62).
Output: data/interim/psps_fee_lines.csv (code x setting x line), psps_fee_per_case.csv (code x setting)
"""
import re

import duckdb
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
YEARS = (2022, 2023, 2024)
CODES = {
    "22551": "ACDF, first interspace", "22552": "ACDF, each added interspace", "22554": "anterior cervical fusion without discectomy",
    "22585": "anterior fusion, each added interspace (any region)", "22845": "anterior instrumentation, 2-3 segments",
    "22846": "anterior instrumentation, 4-7 segments", "22853": "interbody device, each interspace", "22854": "intervertebral device, corpectomy defect",
    "63081": "cervical corpectomy, first segment", "63082": "cervical corpectomy, each added segment",
    "22600": "posterior cervical fusion, first segment", "22614": "posterior fusion, each added segment (any region)",
    "22840": "posterior nonsegmental instrumentation", "22842": "posterior segmental instrumentation, 3-6 segments",
    "22843": "posterior segmental instrumentation, 7-12 segments", "63045": "cervical laminectomy with foraminotomy, first segment",
    "63048": "laminectomy with foraminotomy, each added segment (any region)", "63015": "cervical laminectomy, more than 2 segments",
    "63001": "cervical laminectomy, 1-2 segments", "63050": "cervical laminoplasty", "63051": "cervical laminoplasty with fixation",
    "00600": "anesthesia, cervical spine", "00670": "anesthesia, extensive spine procedures",
    # lumbar and thoracic lines (TEAM outpatient anchors: 22612 and 22630 map to MS-DRG 451, 22633 to MS-DRG 402)
    "22612": "posterolateral lumbar fusion, first segment", "22610": "posterolateral thoracic fusion, first segment",
    "22630": "posterior lumbar interbody fusion, first interspace", "22632": "posterior lumbar interbody fusion, each added interspace",
    "22633": "combined posterolateral and interbody lumbar fusion, first interspace", "22634": "combined lumbar fusion, each added interspace",
    "22558": "anterior lumbar interbody fusion, first interspace", "22556": "anterior thoracic interbody fusion, first interspace",
    "63047": "lumbar laminectomy with foraminotomy, first segment", "63052": "decompression with lumbar interbody fusion, first segment",
    "63053": "decompression with lumbar interbody fusion, each added segment", "00630": "anesthesia, lumbar spine"}
ASSIST = ("80", "81", "82", "AS")
ANES_PRIMARY = ("AA", "AD", "QK", "QY", "QZ")     # 1 of these per anesthetic; QX is the directed CRNA's second line


def lines():
    con, out = duckdb.connect(), []
    codes = ",".join(f"'{c}'" for c in CODES)
    for f in sorted((ROOT / "data/raw/cms_psps").glob("*.csv")):
        year = int(re.search(r"(20\d\d)", f.name).group(1))
        if year not in YEARS:
            continue
        q = f"""
        SELECT {year} AS year, HCPCS_CD AS code,
               CASE WHEN PLACE_OF_SERVICE_CD = '21' THEN 'inpatient' WHEN PLACE_OF_SERVICE_CD IN ('22', '19') THEN 'hospital outpatient'
                    WHEN PLACE_OF_SERVICE_CD = '24' THEN 'ASC' ELSE 'other' END AS setting,
               CASE WHEN TYPE_OF_SERVICE_CD = '7' AND coalesce(HCPCS_INITIAL_MODIFIER_CD, '~') IN {ANES_PRIMARY} THEN 'anesthesia, primary line'
                    WHEN TYPE_OF_SERVICE_CD = '7' THEN 'anesthesia, second line'
                    WHEN coalesce(HCPCS_INITIAL_MODIFIER_CD, '~') IN {ASSIST} OR coalesce(HCPCS_SECOND_MODIFIER_CD, '~') IN {ASSIST} OR TYPE_OF_SERVICE_CD = '8' THEN 'assistant'
                    WHEN coalesce(HCPCS_INITIAL_MODIFIER_CD, '~') = '62' OR coalesce(HCPCS_SECOND_MODIFIER_CD, '~') = '62' THEN 'co-surgeon'
                    WHEN TYPE_OF_SERVICE_CD = '2' THEN 'surgeon' ELSE 'other' END AS line,
               TRY_CAST(PSPS_SUBMITTED_SERVICE_CNT AS DOUBLE) - coalesce(TRY_CAST(PSPS_DENIED_SERVICES_CNT AS DOUBLE), 0) AS svc,
               TRY_CAST(PSPS_ALLOWED_CHARGE_AMT AS DOUBLE) AS allowed,
               TRY_CAST(PSPS_NCH_PAYMENT_AMT AS DOUBLE) AS pay
        FROM read_csv('{f}', all_varchar=true, header=true, ignore_errors=true)
        WHERE HCPCS_CD IN ({codes})"""
        out.append(con.execute(q).df())
        print("read", f.name, flush=True)
    d = pd.concat(out)
    g = d.groupby(["code", "setting", "line"]).apply(lambda x: pd.Series({
        "svc_reported": x.svc.sum(skipna=True), "allowed_reported": x.loc[x.svc.notna(), "allowed"].sum(),
        "pay_reported": x.loc[x.svc.notna(), "pay"].sum(), "pay_blank": x.loc[x.svc.isna(), "pay"].sum()}), include_groups=False).reset_index()
    pay_per_svc = g.pay_reported / g.svc_reported.where(g.svc_reported > 0)
    allowed_per_pay = g.allowed_reported / g.pay_reported.where(g.pay_reported > 0)
    g["services"] = g.svc_reported + (g.pay_blank / pay_per_svc).fillna(0)
    g["allowed"] = g.allowed_reported + (g.pay_blank * allowed_per_pay).fillna(0)
    g["blank_share_of_dollars"] = g.pay_blank / (g.pay_blank + g.pay_reported).where((g.pay_blank + g.pay_reported) > 0)
    g["what"] = g.code.map(CODES)
    return g[["code", "what", "setting", "line", "services", "allowed", "blank_share_of_dollars"]]


def per_case(g):
    """Per operation: cases, dollars for the surgeon side (surgeon + co-surgeon + assistant) and per added unit."""
    rows = []
    for (code, setting), x in g[g.setting.isin(["inpatient", "hospital outpatient"])].groupby(["code", "setting"]):
        s = x.set_index("line")
        get = lambda l, c: float(s.loc[l, c]) if l in s.index else 0.0
        if code.startswith("00"):
            cases = get("anesthesia, primary line", "services")
            total = get("anesthesia, primary line", "allowed") + get("anesthesia, second line", "allowed")
        else:
            cases = get("surgeon", "services") + get("co-surgeon", "services") / 2
            total = get("surgeon", "allowed") + get("co-surgeon", "allowed") + get("assistant", "allowed")
        if cases <= 0:
            continue
        rows.append({"code": code, "what": CODES[code], "setting": setting, "cases_3yr": cases, "allowed_per_case": total / cases,
                     "surgeon_only_per_case": (get("surgeon", "allowed") / get("surgeon", "services")) if get("surgeon", "services") else float("nan"),
                     "assistant_share_of_dollars": get("assistant", "allowed") / total if total else float("nan")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    g = lines()
    g.to_csv(I / "psps_fee_lines.csv", index=False)
    p = per_case(g)
    p.to_csv(I / "psps_fee_per_case.csv", index=False)
    pd.set_option("display.width", 220)
    w = p.pivot_table(index=["code", "what"], columns="setting", values=["cases_3yr", "allowed_per_case"]).round(0)
    print(w.to_string())
    tot = g[g.setting.isin(["inpatient", "hospital outpatient"])]
    print("\nshare of payment dollars in blanked rows, inpatient and hospital outpatient:", round((tot.blank_share_of_dollars * tot.allowed).sum() / tot.allowed.sum(), 3))
