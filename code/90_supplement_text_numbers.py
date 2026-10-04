"""Saves the numbers quoted in the Supplement text that no other output file states, each with its source.

Output: data/interim/supplement_text_numbers.csv
"""
import json

import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
R = ROOT / "data/interim/suppressed_cells"

if __name__ == "__main__":
    rows = []
    add = lambda item, value, source: rows.append({"item": item, "value": value, "source": source})
    gt = json.load(open(R / "ground_truth.json"))
    add("Indiana 2021-2024 suppressed spine cells", gt["indiana_2021_2024"]["suppressed_cells"], "ground_truth.json: indiana_2021_2024.suppressed_cells")
    add("Indiana 2021-2024 discharges in suppressed spine cells", gt["indiana_2021_2024"]["true_hidden_cases"], "ground_truth.json: indiana_2021_2024.true_hidden_cases")
    cov = [1 - c["share_not_covered"] for c in gt["state_coverage"]]
    add("published state rows as a share of national spine volume, lowest year", min(cov), "ground_truth.json: state_coverage")
    add("published state rows as a share of national spine volume, highest year", max(cov), "ground_truth.json: state_coverage")
    sm = pd.read_csv(I / "setting_migration.csv")
    sp = sm[(sm.year >= 2013) & ~sm.family.str.contains("arthroplasty \\(27|Craniotomy|hematoma|shunt|thrombectomy", case=False, regex=True)]
    add("PSPS services imputed from payments, spine procedure families, 2013-2024", sp.svc_imputed.sum() / sp.services.sum(), "setting_migration.csv: svc_imputed / services")
    add("PSPS services imputed from payments, all families in the file, 2013-2024", sm[sm.year >= 2013].svc_imputed.sum() / sm[sm.year >= 2013].services.sum(), "setting_migration.csv: svc_imputed / services")
    import glob
    sh_ = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/cms_mupinp_geo/*.CSV"))):
        d = pd.read_csv(f, encoding="latin-1", dtype=str)
        d = d[d.Rndrng_Prvdr_Geo_Lvl == "National"]
        drg, n = pd.to_numeric(d.DRG_Cd, errors="coerce"), pd.to_numeric(d.Tot_Dschrgs, errors="coerce")
        sh_.append(n[drg.isin([23, 25, 26, 27])].sum() / n[drg.isin([20, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33])].sum())
    add("MS-DRGs 23 and 25-27 as a share of complete Medicare FFS cranial discharges, lowest year", min(sh_), "CMS by Geography and Service files, national rows")
    add("MS-DRGs 23 and 25-27 as a share of complete Medicare FFS cranial discharges, highest year", max(sh_), "CMS by Geography and Service files, national rows")
    pn = pd.read_csv(I / "multistate_panel.csv")
    add("states with any HCUPnet year in the panel", pn.st.nunique(), "multistate_panel.csv")
    add("most states in a single year", int(pn.groupby("year").st.nunique().max()), "multistate_panel.csv")
    add("fewest states in a single year", int(pn.groupby("year").st.nunique().min()), "multistate_panel.csv")
    sh = pd.read_csv(I / "suppressed_high_volume_hospitals.csv").query("state == 'New York' and year == 2024").set_index("line")
    for line in ("Spine", "Joint replacement", "Cranial"):
        r = sh.loc[line]
        add(f"New York 2024 high-volume {line.lower()} hospitals: all / in file / suppressed / absent or unlinked", f"{int(r.high_volume)} / {int(r.in_file)} / {int(r.suppressed)} / {int(r.absent_or_unlinked)}", "suppressed_high_volume_hospitals.csv")
    ss = pd.read_csv(I / "provider_file_search_summary.csv", index_col=0).n
    add("literature search: records screened", int(ss["records screened"]), "provider_file_search_summary.csv; queries and seeds in data/manual/provider_file_search/query_log.md (34 queries, 6 seed papers, dates from 2014/01)")
    out = pd.DataFrame(rows)
    out.to_csv(I / "supplement_text_numbers.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
    print(out.to_string(index=False))
