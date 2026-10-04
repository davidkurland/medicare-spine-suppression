"""Negative control for the missing-hospital finding: do high-volume JOINT REPLACEMENT hospitals vanish from the
public file the way high-volume spine hospitals do?

Same method as 40 (New York) and 43 (Indiana), same misclassification function, same hospital units:
  all-payer inpatient joint replacement discharges by hospital-year from state records, against published CMS FFS
  joint replacement discharges by CCN-year from the public provider file. A high-volume hospital is "invisible" when
  it has >= 100 all-payer joint replacement discharges but no published CMS joint replacement row.
Definitions:
  New York SPARCS  APR-DRG 301 hip, 302 knee joint replacement; APR-DRG v38 (2021) split these into 323-326, folded back in.
  Indiana IDOH     MS-DRGs 469, 470, 521, 522.
  CMS FFS          MS-DRGs 469, 470, 521, 522 (521/522 carved out of 469/470 in FY2024).
Joint replacement volume per hospital is large, so the expectation is that few high-volume hospitals disappear: the
spine finding should be specific to a line whose volume is spread thin across hospitals and severity tiers.
Mechanism test (primary): counting every missing row mixes suppression with hospitals that are absent from the file
(PPS-exempt hospitals) or that our crosswalk could not link. The crosswalk was built for the neurosurgical
cohort, so orthopedic-heavy campuses billing under a parent CCN are unlinked for joints but not for spine. The fair
test is therefore: among high-volume hospitals that appear in the CMS file that year, the share whose every cell for
the service line is suppressed. Computed identically for spine (from the NY rank table and Indiana hospital volumes),
cranial and joint replacement.
Data: New York SPARCS is a public-use de-identified discharge file and the Indiana files are hospital-level counts;
outputs are hospital-year counts.
Outputs: data/interim/volume_misclassification_joints_ny.csv, volume_misclassification_joints_in.csv,
         data/interim/suppressed_high_volume_hospitals.csv
"""
import importlib
import re

import duckdb
import pandas as pd

from common import ROOT

mc = importlib.import_module("40_volume_misclassification")
hs = importlib.import_module("52_hidden_share_team_episodes")
I = ROOT / "data/interim"
JOINT_DRG = [469, 470, 521, 522]
JOINT_APR = [301, 302, 323, 324, 325, 326]


def cms_joint_by_ccn(states):
    """Published CMS FFS joint replacement discharges by CCN and year for the given states."""
    rows = []
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1", low_memory=False)
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        ccn, st = hs.col(d, "Prvdr_CCN"), hs.col(d, "State_Abrvtn")
        drg, n = hs.col(d, "DRG_Cd"), hs.col(d, "Tot_Dschrgs")
        d = d[[ccn, st, drg, n]].rename(columns={ccn: "ccn", st: "state", drg: "drg", n: "n"})
        d["drg"] = pd.to_numeric(d.drg, errors="coerce"); d["n"] = pd.to_numeric(d.n, errors="coerce")
        d = d[d.state.isin(states) & d.drg.isin(JOINT_DRG)]
        d["ccn"] = d.ccn.str.zfill(6)
        rows.append(d.groupby(["state", "ccn"]).n.sum().rename("cms_ffs").reset_index().assign(year=y))
    return pd.concat(rows, ignore_index=True)


def new_york(cms):
    src = f"read_parquet('{ROOT}/data/interim/sparcs/*/*.parquet', union_by_name=true)"
    d = duckdb.connect().execute(f"""
        SELECT year, CAST(pfi AS BIGINT) AS pfi, count(*) AS all_payer
        FROM {src}
        WHERE year BETWEEN 2013 AND 2024 AND TRY_CAST(apr_drg AS INT) IN ({",".join(map(str, JOINT_APR))}) AND pfi IS NOT NULL
        GROUP BY year, CAST(pfi AS BIGINT)""").df()
    # PFI -> analysis unit exactly as the spine analysis (CCN crosswalk, then manual consolidation)
    xw = pd.read_csv(I / "ny_pfi_ccn_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok == True].drop_duplicates("pfi")[["pfi", "ccn"]]
    d = d.merge(xw, on="pfi", how="left")
    d["unit"] = d.ccn.fillna("pfi" + d.pfi.astype(int).astype(str))
    cons = pd.read_csv(ROOT / "data/manual/NY_ccn_consolidation.csv", dtype=str)
    for r in cons.itertuples():
        m = (d.unit == r.from_unit) & (d.year >= int(r.start_year))
        d.loc[m, "unit"] = r.to_ccn
    t = d.groupby(["year", "unit"]).all_payer.sum().reset_index()
    c = cms[cms.state == "NY"].rename(columns={"ccn": "unit"})[["year", "unit", "cms_ffs"]]
    t = t.merge(c, on=["year", "unit"], how="left").fillna({"cms_ffs": 0})
    return t[t.all_payer > 0]


def indiana(cms):
    parts = []
    for f in sorted((ROOT / "data/raw/states/IN").glob("*_MSDRG_IN*.csv")):
        d = pd.read_csv(f)
        d.columns = [c.upper() for c in d.columns]
        d = d[pd.to_numeric(d.MSDRG, errors="coerce").isin(JOINT_DRG)]
        parts.append(d.groupby("HOSPITAL_ID").PATS.sum().rename("all_payer").reset_index().assign(year=int(f.name[:4])))
    h = pd.concat(parts).rename(columns={"HOSPITAL_ID": "id"})
    xw = pd.read_csv(I / "state_compare/IN_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok == True][["id", "ccn"]]
    m = h.merge(xw, on="id", how="left").dropna(subset=["ccn"])
    m["ccn"] = m.ccn.str.zfill(6)
    m = m.groupby(["ccn", "year"]).all_payer.sum().reset_index()
    c = cms[cms.state == "IN"][["year", "ccn", "cms_ffs"]]
    m = m.merge(c, on=["year", "ccn"], how="left").fillna({"cms_ffs": 0})
    return m[m.all_payer > 0]


def indiana_neuro(group):
    """Indiana all-payer discharges by CCN for a neuro group (as 43_indiana_misclassification), with CMS published rows."""
    from common import DRG_TO_CAT, MSDRG_GROUP
    parts = []
    for f in sorted((ROOT / "data/raw/states/IN").glob("*_MSDRG_IN*.csv")):
        d = pd.read_csv(f)
        d.columns = [c.upper() for c in d.columns]
        d["group"] = pd.to_numeric(d.MSDRG, errors="coerce").map(DRG_TO_CAT).map(MSDRG_GROUP)
        parts.append(d[d.group == group].groupby("HOSPITAL_ID").PATS.sum().rename("all_payer").reset_index().assign(year=int(f.name[:4])))
    h = pd.concat(parts).rename(columns={"HOSPITAL_ID": "id"})
    xw = pd.read_csv(I / "state_compare/IN_crosswalk.csv", dtype={"ccn": str})
    xw = xw[xw.match_ok == True][["id", "ccn"]]
    m = h.merge(xw, on="id", how="left").dropna(subset=["ccn"]).groupby(["ccn", "year"]).all_payer.sum().reset_index()
    cms = pd.read_parquet(I / "cms_neuro_hospital_year.parquet")
    cms = cms[(cms.state == "IN") & (cms.group == group)].groupby(["ccn", "year"]).discharges.sum().rename("cms_ffs").reset_index()
    m = m.merge(cms, on=["ccn", "year"], how="left").fillna({"cms_ffs": 0})
    return m[m.all_payer > 0]


def in_cms_file():
    """(year, ccn) pairs present in the public provider file for any DRG."""
    seen = set()
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        c = pd.read_csv(f, dtype=str, encoding="latin-1", usecols=lambda x: "CCN" in x).iloc[:, 0].str.zfill(6)
        seen.update((y, u) for u in c.unique())
    return seen


def suppressed_share(t, unit_col, seen, line, state, threshold=100):
    """Among hospitals with >= threshold all-payer cases that appear in the CMS file, share with no published row."""
    h = t[t.all_payer >= threshold].copy()
    h["in_file"] = [(y, str(u).zfill(6)) in seen for y, u in zip(h.year, h[unit_col])]
    rows = []
    for y, g in h.groupby("year"):
        f = g[g.in_file]
        rows.append(dict(state=state, line=line, year=y, high_volume=len(g), in_file=len(f),
                         suppressed=int((f.cms_ffs <= 0).sum()),
                         suppressed_share=(f.cms_ffs <= 0).mean() if len(f) else float("nan"),
                         absent_or_unlinked=int((~g.in_file).sum())))
    return rows


if __name__ == "__main__":
    cms = cms_joint_by_ccn(["NY", "IN"])
    pd.set_option("display.width", 220)
    cols = ["year", "high_volume_hospitals", "high_volume_invisible_in_cms", "high_volume_invisible_share",
            "allpayer_patients_at_cms_invisible_hospitals_share"]
    for name, fn in [("ny", new_york), ("in", indiana)]:
        out = mc.misclassification(fn(cms))
        out.to_csv(I / f"volume_misclassification_joints_{name}.csv", index=False)
        print(f"\n{name.upper()}  joint replacement, >=100/yr")
        print(out[out.rule == ">=100/yr"][cols].round(3).to_string(index=False))

    seen = in_cms_file()
    rows = []
    rows += suppressed_share(new_york(cms), "unit", seen, "Joint replacement", "New York")
    rows += suppressed_share(indiana(cms), "ccn", seen, "Joint replacement", "Indiana")
    ny_sp = pd.read_csv(I / "ny_rates_rankings/rank_table.csv")
    ny_sp = ny_sp[(ny_sp.grp == "spine") & (ny_sp.all_payer > 0)]
    rows += suppressed_share(ny_sp, "unit", seen, "Spine", "New York")
    in_sp = pd.read_csv(I / "in_hospital_spine_volume.csv", dtype={"ccn": str})
    rows += suppressed_share(in_sp, "ccn", seen, "Spine", "Indiana")
    # cranial comparator line; same units, same restriction to hospitals in the file
    ny_cr = pd.read_csv(I / "ny_rates_rankings/rank_table.csv")
    ny_cr = ny_cr[(ny_cr.grp == "cranial") & (ny_cr.all_payer > 0)]
    rows += suppressed_share(ny_cr, "unit", seen, "Cranial", "New York")
    rows += suppressed_share(indiana_neuro("cranial"), "ccn", seen, "Cranial", "Indiana")
    sup = pd.DataFrame(rows)
    sup.to_csv(I / "suppressed_high_volume_hospitals.csv", index=False)
    print("\nMECHANISM TEST: high-volume hospitals in the CMS file with every cell suppressed")
    print(sup[sup.year.isin([2013, 2017, 2019, 2022, 2024])].pivot_table(
        index="year", columns=["state", "line"], values="suppressed_share").mul(100).round(1).to_string())
    print(sup[sup.year == 2024].to_string(index=False))
