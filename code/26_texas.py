"""Texas PUDF 2016-2019 (record level): Medicare FFS vs Medicare Advantage neurosurgical inpatients, validated
against CMS's exact Texas FFS totals; anterior vs posterior cervical fusion.

Payer (FIRST_PAYMENT_SRC): MA = Medicare Part A (FFS), MB = Medicare Part B (FFS), 16 = HMO Medicare Risk
(Medicare Advantage). Age (PAT_AGE): codes 16-21 = 65+, 25-26 = 65+ (HIV/substance masking bands).
Files contain stray quote characters, so quoting is disabled when reading.
Principal ICD-10-PCS (PRINC_SURG_PROC_CODE) families; 7th character of fusion codes gives approach/column
(0 anterior approach anterior column; 1 posterior approach posterior column; J posterior approach anterior column).
Texas masks hospitals with <50 discharges into THCIC_ID 999999; statewide totals are unaffected.
Used under the Texas PUDF data use agreement (aggregate reporting only). Citation: Texas Hospital Inpatient
Discharge Public Use Data File, 2016-2019. Texas Department of State Health Services, Austin, Texas.
Outputs: data/interim/texas/*.csv. These aggregate counts can include cells of 1-10; they stay local under
data/interim (git-ignored), and any count of 1-10 is suppressed before it is shown or shared (Texas data use agreement).
"""
import glob
import re

import duckdb
import pandas as pd

from common import DRG_TO_CAT, MSDRG_GROUP, ROOT

I = ROOT / "data/interim"
OUT = I / "texas"
OUT.mkdir(parents=True, exist_ok=True)
TX = ROOT / "data/raw/states/TX"

PCS_FAMILY = """
  CASE
    WHEN regexp_matches(pc, '^0RG[124]..0') THEN 'cervical_fusion_anterior'
    WHEN regexp_matches(pc, '^0RG[0124]..[1J]') THEN 'cervical_fusion_posterior'
    WHEN regexp_matches(pc, '^0RG[678A]') THEN 'thoracic_fusion'
    WHEN regexp_matches(pc, '^0SG[013]..0') THEN 'lumbar_fusion_anterior'
    WHEN regexp_matches(pc, '^0SG[013]..[1J]') THEN 'lumbar_fusion_posterior'
    WHEN regexp_matches(pc, '^0R[BT][359B]') OR regexp_matches(pc, '^0S[BT][24]') THEN 'discectomy'
    WHEN regexp_matches(pc, '^00N[WXY]') OR regexp_matches(pc, '^01N[18BR]') THEN 'decompression'
    WHEN regexp_matches(pc, '^00[BCT][0-9A-D]') THEN 'cranial_excision'
    WHEN regexp_matches(pc, '^0016') THEN 'csf_shunt'
    WHEN regexp_matches(pc, '^03C[GHJKQSU]') THEN 'intracranial_thrombectomy'
  END"""


def base_files():
    files = []
    for f in sorted(glob.glob(str(TX / "PUDF*" / "**" / "*"), recursive=True)):
        name = f.lower()
        if re.search(r"base.?1|base_data_1|base1", name) and name.endswith((".txt", ".tab", ".tsv", ".dat")):
            files.append(f)
    return files


if __name__ == "__main__":
    files = base_files()
    print(len(files), "base #1 files")
    con = duckdb.connect()
    parts = []
    for f in files:
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_csv('{f}', delim='\t', header=true, all_varchar=true, sample_size=5000, quote='', escape='', strict_mode=false, null_padding=true)").fetchall()]
        need = {"DISCHARGE", "THCIC_ID", "PAT_AGE", "FIRST_PAYMENT_SRC", "MS_DRG", "PRINC_SURG_PROC_CODE"}
        if not need.issubset(set(cols)):
            print("skip", f, "missing", need - set(cols))
            continue
        q = f"""
          SELECT left(DISCHARGE, 4)::INT AS year, DISCHARGE AS quarter, THCIC_ID AS thcic_id,
                 TRY_CAST(MS_DRG AS INT) AS msdrg, upper(trim(PRINC_SURG_PROC_CODE)) AS pc,
                 CASE WHEN FIRST_PAYMENT_SRC IN ('MA','MB') THEN 'medicare_ffs'
                      WHEN FIRST_PAYMENT_SRC = '16' THEN 'medicare_adv' ELSE 'other' END AS payer,
                 TRY_CAST(PAT_AGE AS INT) BETWEEN 16 AND 21 OR TRY_CAST(PAT_AGE AS INT) IN (25, 26) AS age65
          FROM read_csv('{f}', delim='\t', header=true, all_varchar=true, quote='', escape='', strict_mode=false, null_padding=true)"""
        d = con.execute(f"""
          WITH b AS ({q})
          SELECT year, payer, age65, msdrg, {PCS_FAMILY} AS pcs_family, count(*) AS n
          FROM b GROUP BY ALL""").df()
        parts.append(d)
        print("read", f, int(d.n.sum()), flush=True)
    d = pd.concat(parts)
    d["category"] = d.msdrg.map(DRG_TO_CAT)
    d["group"] = d.category.map(MSDRG_GROUP)
    d.to_csv(OUT / "tx_counts_long.csv", index=False)

    pd.set_option("display.width", 250)
    # 1. DRG groups by payer (all ages and 65+)
    g = d[d.group.isin(["spine", "cranial"])]
    by_payer = g.pivot_table(index=["group", "year"], columns="payer", values="n", aggfunc="sum").fillna(0)
    by_payer["medicare_total"] = by_payer.medicare_ffs + by_payer.medicare_adv
    by_payer["ma_share"] = by_payer.medicare_adv / by_payer.medicare_total
    by_payer65 = g[g.age65].pivot_table(index=["group", "year"], columns="payer", values="n", aggfunc="sum").fillna(0)
    by_payer["ma_share_65plus"] = by_payer65.medicare_adv / (by_payer65.medicare_adv + by_payer65.medicare_ffs)

    # 2. validation vs CMS exact Texas FFS totals (geography) and published provider rows
    st = pd.read_csv(I / "suppression_exact_state.csv")
    st = st[(st.st == "TX") & st.group.isin(["spine", "cranial"])].set_index(["group", "year"])
    by_payer = by_payer.join(st[["state_total_published_cells", "provider_published"]])
    by_payer["tx_ffs_vs_cms_total"] = by_payer.medicare_ffs / by_payer.state_total_published_cells
    by_payer["visible_share_of_medicare"] = by_payer.provider_published / by_payer.medicare_total
    by_payer.reset_index().to_csv(OUT / "tx_drg_groups_by_payer.csv", index=False)
    print(by_payer.round(3).to_string())

    # 3. PCS families by payer (principal procedure), anterior vs posterior cervical
    p = d.dropna(subset=["pcs_family"]).pivot_table(index=["pcs_family", "year"], columns="payer", values="n", aggfunc="sum").fillna(0)
    p["all"] = p.sum(axis=1)
    p["ma_share_of_medicare"] = p.medicare_adv / (p.medicare_adv + p.medicare_ffs)
    p.reset_index().to_csv(OUT / "tx_pcs_families_by_payer.csv", index=False)
    print(p.round(3).to_string())
