"""Indiana analyses: linkage reproducibility, payer coding, split decomposition, rates, open cranial.

Imports load_counts and load_keys from 07_indiana and load, payer_map and pcs_group from 10_indiana_neuro.

1. Automated vs verified linkage: regenerate the automated crosswalk (common.match_to_hcris without
   overrides), hospital-level Spearman rho (state Medicare vs CMS FFS within published cells) with
   bootstrap 95% CIs over hospitals, compared with the automated-crosswalk agreement values in
   data/manual/IN_rank_agreement_automated_crosswalk.csv; sensitivity
   excluding the 3 "probable" overrides.
2. Linkage coverage by match tier and year.
3. Classification of the 18 manual overrides (evidence from HCRIS names/addresses, IDOH keys, web profiles).
4. Payer coding: key labels by year; published-cell CMS:state FFS ratio (aggregate and across cells);
   hospital-level ratio vs hospital MA share; MA share of enrollment vs of neurosurgical inpatients;
   CMS "by Geography" Indiana FFS totals (include provider-suppressed cells) vs state FFS totals.
5. Decomposition with "suppressed" split into (a) state count <11, (b) state count >=11 (mismatch),
   (c) linked hospital absent from the whole CMS provider file; plus scenario rows.
6. Per-beneficiary rates, 2021-2024.
7. Open vs endovascular cranial from the principal ICD-10-PCS table.

CMS data years are calendar years (data.cms.gov catalog metadata), matching Indiana's calendar-year PUF.
Outputs: data/manual/IN_crosswalk_automated.csv, data/interim/indiana_linkage/*.csv
"""
import importlib
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import DRG_TO_CAT, MSDRG_GROUP, ROOT, hcris_state, match_to_hcris

RAW = ROOT / "data/raw/states/IN"
SC = ROOT / "data/interim/state_compare"
OUT = ROOT / "data/interim/indiana_linkage"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000
RNG = np.random.default_rng(20260915)

ind07 = importlib.import_module("07_indiana")          # load_counts (APR-DRG all-cause), load_keys
ind10 = importlib.import_module("10_indiana_neuro")    # load (MSDRG/PROC with payer), payer_map, pcs_group

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


# ----------------------------------------------------------------------------------------------
# crosswalks
# ----------------------------------------------------------------------------------------------
def build_crosswalks():
    """Automated (fuzzy only) and verified (fuzzy + manual overrides, as in 07_indiana.py)."""
    keys = ind07.load_keys()
    latest = keys.sort_values("year").drop_duplicates("id", keep="last")
    auto = match_to_hcris(latest[["id", "name", "city"]], "IN")
    auto["ccn"] = auto.ccn.astype("string")
    auto.to_csv(ROOT / "data/manual/IN_crosswalk_automated.csv", index=False)

    ov = pd.read_csv(ROOT / "data/manual/IN_crosswalk_overrides.csv", dtype={"ccn": str})
    # the 3 overrides that rest on address and name-history evidence rather than a name match carry "probable-link" in
    # their note; they are kept in the primary analysis and dropped in the sensitivity scenarios
    ov["probable"] = ov.note.str.contains("probable", case=False)
    ver = auto[~auto.id.isin(ov.id)]
    ov_rows = ov.merge(latest[["id", "name", "city"]], on="id", how="left").assign(match_ok=True)
    ver = pd.concat([ver, ov_rows.drop(columns="probable")], ignore_index=True)

    # match tier for each IDOH id under the verified crosswalk
    prob_ids = set(ov.loc[ov.probable, "id"])
    def tier(r):
        if r.id in prob_ids:
            return "override_probable"
        if r.id in set(ov.id):
            return "override_manual"
        if not r.match_ok:
            return "unlinked"
        return "auto_high_name" if r.name_score >= 0.85 else "auto_other"
    ver["tier"] = ver.apply(tier, axis=1)

    # check the regenerated verified crosswalk against the one 07_indiana.py wrote
    old = pd.read_csv(SC / "IN_crosswalk.csv", dtype={"ccn": str})
    a = old[old.match_ok].set_index("id").ccn
    b = ver[ver.match_ok].set_index("id").ccn.astype(str)
    diff = pd.concat([a.rename("ccn_saved"), b.rename("ccn_regenerated")], axis=1)
    diff = diff[diff.ccn_saved.fillna("") != diff.ccn_regenerated.fillna("")]
    print(f"verified crosswalk regenerated: {len(diff)} id->ccn differences vs IN_crosswalk.csv")
    if len(diff):
        print(diff.to_string())
    # automated vs verified differences (which ids the overrides change)
    ad = auto.set_index("id")[["name", "ccn", "cms_name", "name_score", "match_ok"]]
    changed = ad.join(ver.set_index("id")[["ccn", "tier"]].rename(columns={"ccn": "ccn_verified"}))
    changed = changed[changed.ccn.astype(str) != changed.ccn_verified.astype(str)]
    print("ids whose CCN differs between automated and verified crosswalks:")
    print(changed.to_string())
    return auto, ver, ov, latest


# ----------------------------------------------------------------------------------------------
# state x CMS cells (same logic as 10_indiana_neuro.py, parameterized by crosswalk)
# ----------------------------------------------------------------------------------------------
def cms_rows_in():
    rows = pd.read_parquet(ROOT / "data/interim/cms_neuro_drg_rows.parquet")
    rows = rows[(rows.state == "IN") & rows.category.map(MSDRG_GROUP).isin(["cranial", "spine"])]
    return rows.assign(grp=rows.category.map(MSDRG_GROUP)).rename(columns={"drg": "MSDRG"})


def build_cells(med, xw, rows, drop_ccns=()):
    """med: Medicare neuro rows (ffs_like), xw: id->ccn. Returns year x grp x ccn x MSDRG cells."""
    m = med.merge(xw[xw.match_ok][["id", "ccn"]].astype({"ccn": str}), on="id")
    st = m.groupby(["year", "grp", "ccn", "MSDRG"]).PATS.sum().rename("state_medicare")
    cm = rows[~rows.ccn.isin(drop_ccns)].groupby(["year", "grp", "ccn", "MSDRG"]).n.sum().rename("cms_ffs")
    c = pd.concat([st, cm], axis=1).reset_index()
    return c[~c.ccn.isin(drop_ccns)]


def boot_spearman(a, b, n=N_BOOT):
    """Percentile bootstrap CI for Spearman rho, resampling hospitals (vectorized rank correlation)."""
    k = len(a)
    idx = RNG.integers(0, k, size=(n, k))
    ra, rb = rankdata(a[idx], axis=1), rankdata(b[idx], axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    r = np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)
    return np.nanpercentile(r, 2.5), np.nanpercentile(r, 97.5)


def rank_agreement(cells, label):
    """Hospital-level rho within published cells, computed as in the linkage sensitivity eFigure (17_supplement_figures.py)."""
    out = []
    for (g, y), d in cells[cells.year >= 2017].groupby(["grp", "year"]):
        hp = d.dropna(subset=["cms_ffs"]).fillna({"state_medicare": 0}).groupby("ccn")[["cms_ffs", "state_medicare"]].sum()
        a, b = hp.cms_ffs.to_numpy(float), hp.state_medicare.to_numpy(float)
        rho = spearmanr(a, b).statistic if len(hp) >= 3 else np.nan
        lo, hi = boot_spearman(a, b) if len(hp) >= 3 else (np.nan, np.nan)
        out.append({"crosswalk": label, "grp": g, "year": y, "n_hospitals": len(hp),
                    "n_hospitals_state_zero": int((b == 0).sum()), "rho": rho, "rho_lo": lo, "rho_hi": hi})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
# CMS provider-file presence (all DRGs) and geography totals
# ----------------------------------------------------------------------------------------------
def cms_provider_ccns():
    pres = {}
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        d = pd.read_csv(f, usecols=["Rndrng_Prvdr_CCN"], dtype=str, encoding="latin-1")
        pres[y] = set(d.Rndrng_Prvdr_CCN.str.strip().str.zfill(6))
    return pres


def cms_geo_indiana():
    frames = []
    for f in sorted((ROOT / "data/raw/cms_mupinp_geo").glob("*.CSV")):
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        d = d[(d.Rndrng_Prvdr_Geo_Lvl == "State") & (d.Rndrng_Prvdr_Geo_Desc == "Indiana")]
        d = d.assign(year=y, drg=d.DRG_Cd.astype(int), n=pd.to_numeric(d.Tot_Dschrgs))
        frames.append(d[["year", "drg", "n"]])
    g = pd.concat(frames)
    g["grp"] = g.drg.map(DRG_TO_CAT).map(MSDRG_GROUP)
    return g[g.grp.isin(["spine", "cranial"])].groupby(["year", "grp"]).n.sum().rename("cms_geo_state_total")


def enrollment_in():
    e = pd.read_csv(ROOT / "data/raw/cms_enrollment/medicare_monthly_enrollment.csv",
                    usecols=["YEAR", "MONTH", "BENE_GEO_LVL", "BENE_STATE_ABRVTN", "TOT_BENES",
                             "A_ORGNL_MDCR_BENES", "A_MA_AND_OTH_BENES"], dtype={"MONTH": str})
    e = e[(e.MONTH == "Year") & (e.BENE_GEO_LVL == "State") & (e.BENE_STATE_ABRVTN == "IN")]
    e = e.rename(columns={"YEAR": "year", "TOT_BENES": "tot_benes", "A_ORGNL_MDCR_BENES": "a_ffs_benes",
                          "A_MA_AND_OTH_BENES": "a_ma_benes"})[["year", "tot_benes", "a_ffs_benes", "a_ma_benes"]]
    e = e.astype(float).astype({"year": int})
    e["ma_enroll_share"] = e.a_ma_benes / (e.a_ma_benes + e.a_ffs_benes)
    return e.set_index("year")


# ----------------------------------------------------------------------------------------------
# 5. decomposition with split suppressed group
# ----------------------------------------------------------------------------------------------
def decompose(ms_neuro, xw, rows, present, exclude_ids=()):
    """Assign every Indiana Medicare neuro inpatient (2021-2024) to a mutually exclusive group."""
    d = ms_neuro[(ms_neuro.year >= 2021) & ~ms_neuro.id.isin(exclude_ids)]
    linked = xw[xw.match_ok][["id", "ccn"]].astype({"ccn": str})
    out = []
    for (y, g), dd in d.groupby(["year", "grp"]):
        ffs = dd[dd.payer == "medicare_ffs"]
        adv = dd.loc[dd.payer == "medicare_adv", "PATS"].sum()
        fl = ffs.merge(linked, on="id")
        st = fl.groupby(["ccn", "MSDRG"]).PATS.sum().rename("state")
        cm = rows[(rows.year == y) & (rows.grp == g)].groupby(["ccn", "MSDRG"]).n.sum().rename("cms")
        c = pd.concat([st, cm], axis=1).reset_index()
        c = c[c.state.notna()]                              # state cells; CMS-only cells carry no state patients
        pub = c.cms.notna()
        absent = ~pub & ~c.ccn.isin(present[y])
        a = ~pub & ~absent & (c.state < 11)
        b = ~pub & ~absent & (c.state >= 11)
        r = {"year": y, "grp": g, "advantage": adv,
             "published": c.loc[pub, "state"].sum(),
             "supp_a_lt11": c.loc[a, "state"].sum(),
             "supp_b_ge11_mismatch": c.loc[b, "state"].sum(),
             "supp_c_hosp_absent": c.loc[absent, "state"].sum(),
             "unlinked": ffs.PATS.sum() - fl.PATS.sum(),
             "cms_reported": rows[(rows.year == y) & (rows.grp == g) & rows.ccn.isin(c.loc[pub, "ccn"])]
                             .merge(c.loc[pub, ["ccn", "MSDRG"]], on=["ccn", "MSDRG"]).n.sum(),
             "cells_a": int(a.sum()), "cells_b": int(b.sum()), "cells_c": int(absent.sum()),
             "hosp_absent_ccns": ";".join(sorted(c.loc[absent, "ccn"].unique()))}
        r["ffs_total"] = ffs.PATS.sum()
        out.append(r)
    return pd.DataFrame(out)


FFS_GROUPS = ["published", "supp_a_lt11", "supp_b_ge11_mismatch", "supp_c_hosp_absent", "unlinked"]


def finish(df, scenario, supp_def):
    df = df.copy()
    df["scenario"] = scenario
    df["total_medicare"] = df.advantage + df[FFS_GROUPS].sum(1)
    cols = {"all_suppressed": ["supp_a_lt11", "supp_b_ge11_mismatch", "supp_c_hosp_absent"],
            "a_plus_c": ["supp_a_lt11", "supp_c_hosp_absent"],
            "a_only": ["supp_a_lt11"]}[supp_def]
    df["suppressed_definition"] = "+".join(cols)
    df["suppressed_total"] = df[cols].sum(1)
    for k in ["advantage"] + FFS_GROUPS + ["suppressed_total", "cms_reported"]:
        df[f"{k}_share"] = df[k] / df.total_medicare
    return df


if __name__ == "__main__":
    # ------------------------------------------------------------------ data
    auto, ver, ov, latest = build_crosswalks()
    prob_ids = set(ov.loc[ov.probable, "id"])
    prob_ccns = set(ov.loc[ov.probable, "ccn"])
    ms = ind10.load("MSDRG").rename(columns={"HOSPITAL_ID": "id"})
    ms["grp"] = ms.MSDRG.map(DRG_TO_CAT).map(MSDRG_GROUP)
    neuro = ms[ms.grp.isin(["spine", "cranial"])].copy()
    # 10_indiana_neuro.py: FFS from 2021 (payer split available); all Medicare before 2021
    neuro["ffs_like"] = np.where(neuro.year >= 2021, neuro.payer == "medicare_ffs",
                                 neuro.payer.isin(["medicare_ffs", "medicare_any", "medicare_adv"]))
    med = neuro[neuro.ffs_like]
    rows = cms_rows_in()

    # ------------------------------------------------------------------ 1. rank agreement
    cells_auto = build_cells(med, auto, rows)
    cells_ver = build_cells(med, ver, rows)
    cells_noprob = build_cells(med, ver[~ver.id.isin(prob_ids)], rows, drop_ccns=prob_ccns)
    ra = pd.concat([rank_agreement(cells_auto, "automated"), rank_agreement(cells_ver, "verified"),
                    rank_agreement(cells_noprob, "verified_excl_probable")])
    # Verified-crosswalk rho recomputed from the cells file written by 10_indiana_neuro.py, and the
    # automated-crosswalk values from data/manual
    saved_cells = pd.read_csv(SC / "IN_neuro_vs_cms_cells.csv", dtype={"ccn": str})
    saved = pd.DataFrame([{"grp": g, "year": y, "rho_verified_saved": spearmanr(
        *d.dropna(subset=["cms_ffs"]).fillna({"state_medicare": 0}).groupby("ccn")[["cms_ffs", "state_medicare"]].sum().T.values).statistic}
        for (g, y), d in saved_cells[saved_cells.year >= 2017].groupby(["grp", "year"])])
    auto_ref = pd.read_csv(ROOT / "data/manual/IN_rank_agreement_automated_crosswalk.csv")
    ra.to_csv(OUT / "rank_agreement.csv", index=False)
    wide = ra.pivot_table(index=["grp", "year"], columns="crosswalk", values=["rho", "n_hospitals"]).round(3)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    cmp_ = wide.reset_index().merge(auto_ref, on=["grp", "year"]).merge(saved, on=["grp", "year"])
    cmp_["diff_auto_code_minus_table"] = (cmp_.rho_automated - cmp_.rho_hospital_automated).round(3)
    cmp_["diff_verified_regenerated_minus_saved"] = (cmp_.rho_verified - cmp_.rho_verified_saved).round(3)
    cmp_.to_csv(OUT / "rank_agreement_comparison.csv", index=False)
    print("\n== 1. rank agreement ==")
    print(ra.round(3).to_string())
    print(cmp_.to_string())

    # ------------------------------------------------------------------ 2. linkage coverage
    counts = ind07.load_counts()                                   # all-cause (APR-DRG table), newborns flagged
    tiers = ver[["id", "tier"]]
    cov = []
    allc = counts.merge(tiers, on="id", how="left").fillna({"tier": "unlinked"})
    for meas, col in [("all_cause_excl_newborn", "sp"), ("all_cause_incl_newborn", "sp_all")]:
        cov.append(allc.groupby(["year", "tier"])[col].sum().rename("n").reset_index().assign(measure=meas))
    nm = neuro[neuro.payer.str.startswith("medicare")].merge(tiers, on="id", how="left").fillna({"tier": "unlinked"})
    for g in ["spine", "cranial"]:
        cov.append(nm[nm.grp == g].groupby(["year", "tier"]).PATS.sum().rename("n").reset_index()
                   .assign(measure=f"{g}_medicare_all"))
        cov.append(nm[(nm.grp == g) & (nm.payer == "medicare_ffs")].groupby(["year", "tier"]).PATS.sum()
                   .rename("n").reset_index().assign(measure=f"{g}_medicare_ffs"))
    cov = pd.concat(cov)
    # complete the measure x year x tier grid so empty tiers appear as explicit zeros
    grid = cov[["measure", "year"]].drop_duplicates().merge(pd.DataFrame({"tier": sorted(ver.tier.unique())}), how="cross")
    cov = grid.merge(cov, on=["measure", "year", "tier"], how="left").fillna({"n": 0})
    cov["share"] = cov.n / cov.groupby(["measure", "year"]).n.transform("sum")
    cov = cov[["measure", "year", "tier", "n", "share"]].sort_values(["measure", "year", "tier"])
    cov.to_csv(OUT / "linkage_coverage.csv", index=False)
    print("\n== 2. linkage coverage (share linked = 1 - unlinked) ==")
    print(cov.pivot_table(index=["measure", "year"], columns="tier", values="share").round(4).to_string())
    print("ids by tier:", ver.tier.value_counts().to_dict())

    # ------------------------------------------------------------------ 3. override classification
    # Evidence: IDOH key names (2017-2024), HCRIS legal names/addresses (hcris_hospital_cy.parquet), and
    # public CCN profiles (name and address history; ahd.com, healthcare4ppl.com, hospital websites; accessed September 2026).
    cls = {
        31: ("multicampus/shared CCN", "Sibling hospital in Lutheran Health Network (Fort Wayne). Fuzzy match gave it Lutheran Downtown's CCN 150047; its own CCN 150017 is at 7950 W Jefferson Blvd in HCRIS."),
        33: ("renamed/rebranded", "IDOH key names 'St. Joseph Hospital' and later 'Lutheran Downtown'. HCRIS names for CCN 150047 go from ST JOSEPH MEDICAL CENTER to LUTHERAN DOWNTOWN HOSPITAL (702 Van Buren St). Corrected together with id 31."),
        126: ("specialty orthopedic hospital", "Parkview Ortho Hospital has its own CCN 150167 (ORTHOPAEDIC HOSPT.AT PARKVIEW, 11130 Parkview Circle, 37 beds). The fuzzy match folded it into Parkview Hospital."),
        154: ("specialty orthopedic hospital", "OrthoIndy Hospital; CMS legal name INDIANA ORTHOPAEDIC HOSPITAL LLC (8450 Northwest Blvd, Indianapolis, 38 beds)."),
        166: ("specialty orthopedic hospital", "The Orthopedic Hospital of Lutheran Health Network; CMS legal name LUTHERAN MUSCULOSKELETAL CENTER (7952 W Jefferson Blvd, adjacent to 150017)."),
        430: ("specialty orthopedic hospital", "Franciscan Health Orthopedic Hospital Carmel (IDOH key 2024 only); CCN 150193 in HCRIS from 2022 (10777 Illinois St, 20 beds)."),
        138: ("probable", "Community Hospital East. CCN 150074 is at 1500 N Ritter Ave (HCRIS; the same CCN and address appear on public profiles such as ahd.com and healthcare4ppl.com). HCRIS lists only system legal names (COMMUNITY HEALTH NETWORK INC). Public CCN profiles (name and address history) support this link."),
        139: ("probable", "Community Hospital North. CCN 150169 is at 7150 Clearvista Dr (HCRIS; HCRIS name history includes COMMUNITY HOSPITAL NORTH; IDOH-hosted 2019 cost report is titled Community-Hospital-North-150169). Public CCN profiles (name and address history) support this link."),
        105: ("renamed/rebranded", "Saint Joseph Health System Mishawaka Medical Center; CMS legal name ST. JOSEPHS REG MED CENTER S. BEND at 5215 Holy Cross Pkwy, Mishawaka (CCN 150012 confirmed on public profiles)."),
        113: ("renamed/rebranded", "IDOH key names 'Porter Regional Hospital' and later 'Northwest Health-Porter'; CCN 150035 (Valparaiso)."),
        94: ("renamed/rebranded", "Brand name Reid Health vs legal name REID HOSPITAL & HEALTH CARE SERVICES; CCN 150048 (Richmond)."),
        155: ("renamed/rebranded", "'Indiana University Health North Hospital' vs CMS 'IU HEALTH NORTH HOSPITAL' (formerly CLARIAN NORTH); CCN 150161 (Carmel). Only the brand prefix differs."),
        162: ("renamed/rebranded", "'Indiana University Health Arnett Hospital' vs 'IU HEALTH ARNETT HOSPITAL'; CCN 150173 (Lafayette). Only the brand prefix differs."),
        152: ("renamed/rebranded", "'Indiana University Health West Hospital' vs 'IU HEALTH WEST HOSPITAL'; CCN 150158 (Avon). Only the brand prefix differs."),
        38: ("multicampus/shared CCN", "Methodist Hospitals Northlake Campus (Gary). Northlake and Southlake report under the single system CCN 150002 (HCRIS address 600 Grant St, Gary)."),
        49: ("multicampus/shared CCN", "Methodist Hospitals Southlake Campus (Merrillville). Shares CCN 150002 with Northlake."),
        58: ("probable", "Parkview Noble Hospital. CCN 150146 is at 401 Sawyer Rd, Kendallville (HCRIS legal name COMMUNITY HOSPITAL OF NOBLE CTY INC; the same CCN and address appear on public profiles such as ahd.com, healthcare4ppl.com and payerprice.com). Public CCN profiles (name and address history) support this link."),
        89: ("renamed/rebranded", "IDOH 'The Women's Hospital' (Newburgh) vs CMS DEACONESS WOMENS HOSPITAL; CCN 150149 (4199 Gateway Blvd, Newburgh)."),
    }
    h = hcris_state("IN").sort_values("year").drop_duplicates("ccn", keep="last").set_index("ccn")
    oc = ov.merge(latest[["id", "name", "city"]], on="id", how="left")
    oc["category"] = oc.id.map(lambda i: cls[i][0])
    oc["evidence"] = oc.id.map(lambda i: cls[i][1])
    oc["hcris_address"] = oc.ccn.map(lambda c: f"{h.loc[c, 'address']}, {h.loc[c, 'city']}" if c in h.index else "")
    oc["hcris_beds"] = oc.ccn.map(h.beds)
    auto_i = auto.set_index("id")
    oc["automated_ccn"] = oc.id.map(auto_i.ccn)
    oc["automated_cms_name"] = oc.id.map(auto_i.cms_name)
    oc = oc.rename(columns={"name": "idoh_name", "note": "override_note"})
    oc[["id", "idoh_name", "city", "ccn", "cms_name", "category", "evidence", "hcris_address", "hcris_beds",
        "automated_ccn", "automated_cms_name", "override_note"]].to_csv(OUT / "override_classification.csv", index=False)
    print("\n== 3. override classes ==", oc.category.value_counts().to_dict())

    # ------------------------------------------------------------------ 4. payer coding
    print("\n== 4. payer key labels ==")
    labels = []
    for y in range(2017, 2025):
        f = next(RAW.glob(f"{y}_PAY*"))
        k = pd.read_excel(f, sheet_name=0)
        k = k.set_axis(["code", "label"] + list(k.columns[2:]), axis=1).sort_values("code")
        mp = ind10.payer_map(y)
        for r in k.itertuples():
            labels.append({"year": y, "file": f.name, "code": int(r.code), "label": str(r.label).strip(),
                           "mapped_to": mp[int(r.code)]})
            print(f"{y} {int(r.code):>2} -> {mp[int(r.code)]:<13} | {str(r.label).strip()}")
    pd.DataFrame(labels).to_csv(OUT / "payer_key_labels.csv", index=False)

    # published-cell ratio: aggregate and across cells (verified crosswalk, FFS 2021+)
    c21 = cells_ver[cells_ver.year >= 2021]
    pub = c21.dropna(subset=["cms_ffs"]).fillna({"state_medicare": 0})
    agg = pub.groupby(["year", "grp"]).agg(cells=("cms_ffs", "size"), cms=("cms_ffs", "sum"), state=("state_medicare", "sum"))
    agg["cms_over_state"] = agg.cms / agg.state
    pos = pub[pub.state_medicare > 0].assign(r=lambda d: d.cms_ffs / d.state_medicare)
    dist = pos.groupby(["year", "grp"]).r.describe(percentiles=[.1, .25, .5, .75, .9])
    dist["share_cms_gt_state"] = pub.groupby(["year", "grp"]).apply(lambda d: (d.cms_ffs > d.state_medicare).mean(), include_groups=False)
    dist["share_equal"] = pub.groupby(["year", "grp"]).apply(lambda d: (d.cms_ffs == d.state_medicare).mean(), include_groups=False)
    dist["share_cms_lt_state"] = pub.groupby(["year", "grp"]).apply(lambda d: (d.cms_ffs < d.state_medicare).mean(), include_groups=False)
    dist["cells_state_zero"] = pub.groupby(["year", "grp"]).apply(lambda d: int((d.state_medicare == 0).sum()), include_groups=False)
    ratio = agg.join(dist.add_prefix("cellratio_"))
    # CMS geography totals include provider cells suppressed in the provider file (IPPS hospitals in IN)
    geo = cms_geo_indiana()
    stffs = neuro[neuro.payer == "medicare_ffs"].groupby(["year", "grp"]).PATS.sum().rename("state_ffs_statewide")
    stffs_link = neuro[(neuro.payer == "medicare_ffs") & neuro.id.isin(ver[ver.match_ok].id)] \
        .groupby(["year", "grp"]).PATS.sum().rename("state_ffs_linked")
    ratio = ratio.join(geo).join(stffs).join(stffs_link)
    ratio["geo_over_state_statewide"] = ratio.cms_geo_state_total / ratio.state_ffs_statewide
    # CMS rows at CCNs with no IDOH-reporting hospital (e.g., Pinnacle Hospital 150166, Crown Point, which is
    # absent from every IDOH hospital key) inflate CMS relative to the state; remove them from both CMS measures
    linked_ccns = set(ver.loc[ver.match_ok, "ccn"].astype(str))
    cms_nonidoh = rows[(rows.year >= 2021) & ~rows.ccn.isin(linked_ccns)]
    ratio["cms_rows_at_nonIDOH_ccns"] = cms_nonidoh.groupby(["year", "grp"]).n.sum()
    ratio["nonIDOH_ccns"] = cms_nonidoh.groupby(["year", "grp"]).ccn.agg(lambda s: ";".join(sorted(s.unique())))
    ratio = ratio.fillna({"cms_rows_at_nonIDOH_ccns": 0})
    ratio["cms_over_state_linked_ccns"] = (ratio.cms - ratio.cms_rows_at_nonIDOH_ccns) / ratio.state
    ratio["geo_minus_nonIDOH_over_state_statewide"] = (ratio.cms_geo_state_total - ratio.cms_rows_at_nonIDOH_ccns) / ratio.state_ffs_statewide
    ratio.to_csv(OUT / "payer_cms_state_ratio.csv")
    print(ratio.round(3).to_string())

    # hospital-level ratio vs MA share (MA share among all-cause Medicare and among neuro Medicare)
    xl = ver[ver.match_ok][["id", "ccn"]].astype({"ccn": str})
    msl = ms[(ms.year >= 2021) & ms.payer.isin(["medicare_ffs", "medicare_adv"])].merge(xl, on="id")
    ma_all = msl.pivot_table(index=["year", "ccn"], columns="payer", values="PATS", aggfunc="sum", fill_value=0)
    ma_all = (ma_all.medicare_adv / ma_all.sum(1)).rename("ma_share_allcause")
    nl = msl[msl.grp.isin(["spine", "cranial"])].pivot_table(index=["year", "grp", "ccn"], columns="payer",
                                                           values="PATS", aggfunc="sum", fill_value=0)
    nl = (nl.medicare_adv / nl.sum(1)).rename("ma_share_neuro")
    hp = pub.groupby(["year", "grp", "ccn"])[["cms_ffs", "state_medicare"]].sum()
    hp = hp[hp.state_medicare > 0].assign(log_ratio=lambda d: np.log(d.cms_ffs / d.state_medicare))
    hp = hp.join(ma_all, on=["year", "ccn"]).join(nl)
    hp.to_csv(OUT / "payer_hospital_ratio_ma.csv")
    corr = []
    for key, d in list(hp.groupby(["year", "grp"])) + [(("pooled", g), d) for g, d in hp.groupby("grp")]:
        for mcol in ["ma_share_allcause", "ma_share_neuro"]:
            dd = d.dropna(subset=[mcol])
            s = spearmanr(dd.log_ratio, dd[mcol])
            big = dd[dd.cms_ffs >= 30]                       # less noisy hospitals
            s2 = spearmanr(big.log_ratio, big[mcol]) if len(big) >= 5 else None
            corr.append({"year": key[0], "grp": key[1], "ma_measure": mcol, "n_hospitals": len(dd), "rho": s.statistic,
                         "p": s.pvalue, "n_hosp_cms_ge30": len(big), "rho_cms_ge30": s2.statistic if s2 else np.nan,
                         "p_cms_ge30": s2.pvalue if s2 else np.nan})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / "payer_hospital_ma_corr.csv", index=False)
    print(corr.round(3).to_string())

    # MA share of enrollment vs of neuro inpatients, implied MA:FFS inpatient rate ratio
    enr = enrollment_in()
    pm = neuro[(neuro.year >= 2021) & neuro.payer.isin(["medicare_ffs", "medicare_adv"])] \
        .pivot_table(index=["year", "grp"], columns="payer", values="PATS", aggfunc="sum").reset_index()
    pm = pm.merge(enr, left_on="year", right_index=True).merge(agg.cms_over_state.reset_index(), on=["year", "grp"])
    pm["ma_inpatient_share"] = pm.medicare_adv / (pm.medicare_adv + pm.medicare_ffs)
    odds = lambda s: s / (1 - s)
    pm["implied_ma_ffs_rate_ratio"] = odds(pm.ma_inpatient_share) / odds(pm.ma_enroll_share)
    ffs_resc = pm.medicare_ffs * pm.cms_over_state
    pm["ma_inpatient_share_rescaled"] = 1 - ffs_resc / (pm.medicare_adv + pm.medicare_ffs)
    pm["implied_rate_ratio_rescaled"] = odds(pm.ma_inpatient_share_rescaled) / odds(pm.ma_enroll_share)
    pm.to_csv(OUT / "ma_enrollment_vs_inpatients.csv", index=False)
    print(pm.round(4).to_string())

    # ------------------------------------------------------------------ 5. decomposition + scenarios
    present = cms_provider_ccns()
    base = decompose(neuro, ver, rows, present)
    noprob_unl = decompose(neuro, ver[~ver.id.isin(prob_ids)], rows, present)
    noprob_exc = decompose(neuro, ver, rows, present, exclude_ids=prob_ids)
    def rescale(df, ratio, label):
        """Scale every state FFS group by ratio (per year x grp); MA absorbs the difference (total fixed)."""
        df = df.copy()
        r_ = df.set_index(["year", "grp"]).index.map(ratio).to_numpy(float)
        total = df.advantage + df[FFS_GROUPS].sum(1)
        df[FFS_GROUPS] = df[FFS_GROUPS].mul(r_, axis=0)
        df["advantage"] = total - df[FFS_GROUPS].sum(1)
        df["rescale_ratio"], df["rescale_ratio_source"] = r_, label
        return df

    # three payer ratios: (i) primary definition, all CMS-published IN cells (state FFS 0 where absent);
    # (ii) only published cells with state FFS > 0 at linked hospitals; (iii) CMS by-Geography Indiana FFS
    # total (includes provider-suppressed cells; IPPS hospitals only) over statewide state FFS
    r_all = agg.cms_over_state.to_dict()
    r_pos = (base.cms_reported / base.published).set_axis(pd.MultiIndex.from_frame(base[["year", "grp"]])).to_dict()
    r_geo = ratio.geo_minus_nonIDOH_over_state_statewide.to_dict()
    resc = rescale(base, r_all, "published cells, all")
    resc_pos = rescale(base, r_pos, "published cells with state FFS > 0")
    resc_geo = rescale(base, r_geo, "(CMS geography total - CMS rows at non-IDOH CCNs) / state FFS statewide")
    # Deaconess Gateway Hospital (IDOH id 756, Newburgh) is fuzzy-linked to the Heart Hospital at Deaconess
    # Gateway CCN 150175, which left the CMS provider file after DY2018; it is all of group (c) in 2023-2024.
    # Sensitivity: relink it to Deaconess Hospital Inc (150082), the system CCN of Deaconess Midtown (unverified).
    ver_gw = ver.copy()
    ver_gw.loc[ver_gw.id == 756, "ccn"] = "150082"
    gw = decompose(neuro, ver_gw, rows, present)
    # all adjustments at once (lower bound for MA and suppressed shares)
    pub_np = cells_noprob[cells_noprob.year >= 2021].dropna(subset=["cms_ffs"]).fillna({"state_medicare": 0})
    r_np = (pub_np.groupby(["year", "grp"]).cms_ffs.sum() / pub_np.groupby(["year", "grp"]).state_medicare.sum()).to_dict()
    comb = rescale(noprob_exc, r_np, "published cells, all, probable CCNs excluded")
    sc = pd.concat([
        finish(base, "base (primary definition)", "all_suppressed"),
        finish(resc, "payer-rescaled (published-cell ratio)", "all_suppressed"),
        finish(resc_pos, "payer-rescaled (cells with state FFS > 0)", "all_suppressed"),
        finish(resc_geo, "payer-rescaled (CMS geography total)", "all_suppressed"),
        finish(noprob_unl, "probable links dropped (treated as unlinked)", "all_suppressed"),
        finish(noprob_exc, "probable-link hospitals excluded", "all_suppressed"),
        finish(base, "mismatch cells not suppressed (a+c)", "a_plus_c"),
        finish(base, "strict suppression only (a)", "a_only"),
        finish(gw, "Deaconess Gateway relinked to 150082 (unverified)", "all_suppressed"),
        finish(comb, "combined: rescaled + probable excluded + strict (a)", "a_only"),
    ], ignore_index=True)
    lead = ["scenario", "year", "grp", "total_medicare", "advantage"] + FFS_GROUPS + ["suppressed_definition", "suppressed_total", "cms_reported"]
    sc = sc[lead + [c for c in sc.columns if c not in lead]]
    sc.to_csv(OUT / "decomposition_scenarios.csv", index=False)
    print("\n== 5. decomposition scenarios ==")
    show = ["scenario", "year", "grp", "total_medicare", "advantage", "published", "supp_a_lt11", "supp_b_ge11_mismatch",
            "supp_c_hosp_absent", "unlinked", "cms_reported", "advantage_share", "published_share", "suppressed_total_share"]
    print(sc[show].round(3).to_string())
    rng = sc[sc.year.isin([2021, 2024])].groupby(["year", "grp"]).agg(
        ma_share_min=("advantage_share", "min"), ma_share_max=("advantage_share", "max"),
        suppressed_share_min=("suppressed_total_share", "min"), suppressed_share_max=("suppressed_total_share", "max"),
        published_share_min=("published_share", "min"), published_share_max=("published_share", "max"))
    rng.to_csv(OUT / "decomposition_ranges.csv")
    print(rng.round(3).to_string())
    print("hospital-absent CCNs by year:", base.set_index(["year", "grp"]).hosp_absent_ccns.to_dict())

    # ------------------------------------------------------------------ 6. per-beneficiary rates
    cms_pub = rows.groupby(["year", "grp"]).n.sum().rename("cms_in_published")
    rt = neuro[neuro.payer.isin(["medicare_ffs", "medicare_adv"]) & (neuro.year >= 2021)] \
        .pivot_table(index=["year", "grp"], columns="payer", values="PATS", aggfunc="sum")
    rt = rt.join(cms_pub).join(geo).reset_index().merge(enr, left_on="year", right_index=True)
    rt["state_medicare"] = rt.medicare_ffs + rt.medicare_adv
    rt["state_medicare_per100k_tot_benes"] = 1e5 * rt.state_medicare / rt.tot_benes
    rt["state_medicare_per100k_partA_benes"] = 1e5 * rt.state_medicare / (rt.a_ffs_benes + rt.a_ma_benes)
    rt["state_ffs_per100k_ffs_partA"] = 1e5 * rt.medicare_ffs / rt.a_ffs_benes
    rt["state_ma_per100k_ma_partA"] = 1e5 * rt.medicare_adv / rt.a_ma_benes
    rt["cms_published_per100k_ffs_partA"] = 1e5 * rt.cms_in_published / rt.a_ffs_benes
    rt["cms_geo_total_per100k_ffs_partA"] = 1e5 * rt.cms_geo_state_total / rt.a_ffs_benes
    rt.to_csv(OUT / "per_beneficiary_rates.csv", index=False)
    print("\n== 6. per-beneficiary ==")
    print(rt.round(1).to_string())

    # ------------------------------------------------------------------ 7. open vs endovascular cranial
    # Principal ICD-10-PCS rule (5th character approach 3 = percutaneous, 4 = percutaneous endoscopic):
    #   endo_core : 03 C/V/L/7 (extirpation, restriction, occlusion, dilation) on body part G (intracranial
    #               artery), or 05 C/V/L on body part L (intracranial vein), approach 3/4.
    #   endo_broad: endo_core + 03 C/V/L on K/L (internal carotid) or P/Q (vertebral), approach 3/4; these
    #               body parts are not exclusively intracranial (cervical ICA/vertebral segments).
    # Body parts in table 03: only G is wholly intracranial. H/J are common carotid, S/T temporal and
    # U/V thyroid arteries (extracranial). K/L and P/Q have intracranial segments.
    #   open_vascular: 03 C/V/L on G with open approach 0 (e.g., aneurysm clipping 03VG0CZ).
    #   pcs_cranial  : 10_indiana_neuro.pcs_group == "cranial" (00-table surgical operations on brain/cranial nerves).
    pc = ind10.load("PROC")
    code = pc.PROCEDURE_1.astype(str).str.upper().str.strip()
    core = code.str.match(r"^03[CVL7]G[34]") | code.str.match(r"^05[CVL]L[34]")
    broad = core | code.str.match(r"^03[CVL][KLPQ][34]")
    openv = code.str.match(r"^03[CVL]G0")
    pc["cls"] = np.select([core, broad, openv, pc.PROCEDURE_1.map(ind10.pcs_group) == "cranial"],
                          ["endo_core", "endo_broad_only", "open_vascular", "pcs_cranial"], "other")
    pc["pay"] = np.where(pc.payer.str.startswith("medicare"), "medicare", "non_medicare")
    pcs_n = pc[pc.cls != "other"].pivot_table(index="year", columns="cls", values="PATS", aggfunc="sum", fill_value=0)
    pcs_med = pc[(pc.cls != "other") & (pc.pay == "medicare")].pivot_table(index="year", columns="cls", values="PATS",
                                                                          aggfunc="sum", fill_value=0).add_prefix("medicare_")
    drgb = ms.assign(band=pd.cut(ms.MSDRG, [19, 22, 24, 27, 30, 33], labels=["drg020_022", "drg023_024", "drg025_027", "x", "drg031_033"]))
    drgb = drgb[drgb.band.isin(["drg020_022", "drg023_024", "drg025_027", "drg031_033"])]
    dn = drgb.pivot_table(index="year", columns="band", values="PATS", aggfunc="sum", observed=True)
    dm = drgb[drgb.payer.str.startswith("medicare")].pivot_table(index="year", columns="band", values="PATS",
                                                                 aggfunc="sum", observed=True).add_prefix("medicare_")
    oc7 = pcs_n.join(pcs_med).join(dn).join(dm)
    oc7["drg020_027"] = oc7[["drg020_022", "drg023_024", "drg025_027"]].sum(1)
    oc7["medicare_drg020_027"] = oc7[["medicare_drg020_022", "medicare_drg023_024", "medicare_drg025_027"]].sum(1)
    oc7["endo_core_share_of_drg020_027"] = oc7.endo_core / oc7.drg020_027
    oc7["endo_broad_share_of_drg020_027"] = (oc7.endo_core + oc7.endo_broad_only) / oc7.drg020_027
    oc7["medicare_endo_core_share_of_drg020_027"] = oc7.medicare_endo_core / oc7.medicare_drg020_027
    oc7["open_cranial_est_drg020_027_minus_endo_core"] = oc7.drg020_027 - oc7.endo_core
    oc7["medicare_open_cranial_est"] = oc7.medicare_drg020_027 - oc7.medicare_endo_core
    oc7.to_csv(OUT / "open_cranial.csv")
    print("\n== 7. open cranial ==")
    print(oc7.round(3).T.to_string())
    top = pc[pc.cls.str.startswith("endo")].groupby("PROCEDURE_1").PATS.sum().sort_values(ascending=False).head(15)
    print("top endovascular principal codes (all years):", top.to_dict())
