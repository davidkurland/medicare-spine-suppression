"""Structured search for spine studies built on CMS public summary files (eMethods 12).

The search was run in October 2026: 34 PubMed queries plus related-article pulls for 6 seed papers gave 386 unique
records (data/manual/provider_file_search/query_log.md). Titles and abstracts were screened against written criteria;
37 were included and 17 marked unsure (screened_records.csv). The 54 were then classified from the abstract into the
groups below. Full texts were not read, so a study whose abstract does not name its file can be misplaced, and the
count of provider-file studies is a floor.

  A  provider-level file (rows removed under 11 beneficiaries or services); reports spine procedure volume or counts
  B  provider-level file; payments, charges or surgeon characteristics only
  C  national summary files (Part B National Summary, PSPS): totals are complete, so provider-row suppression does not apply
  D  cannot be classified from the abstract

Output: data/interim/provider_file_search.csv, provider_file_search_summary.csv
"""
import pandas as pd

from common import ROOT

D = ROOT / "data/manual/provider_file_search"
A = "39332029 38158598 37612890 32562771 32534133 42775937 41848006 41649280 42531462 42554751 41399566 42585156 26425400 40689986 42599313 34563716 41642296 32925685 42412075".split()
B = "28650884 33225298 26076440 28338572 25822546 42428039".split()
C = ("39773683 37306413 36700751 35728787 41520188 42787814 41649844 41317013 41000488 40831911 42683525 38370335 35872133 32417502 30562331 26546300 "
     "37192412 27799375 39891936 40828204 38743853 40236876 28778222 41838245 35652764 32291587").split()
DD = "34524764 41565466 35239289".split()
RECHECKED = {"32562771": "ACDF, 2013-2017", "32534133": "lumbar discectomy, 2013-2017", "42531462": "ACDF, 2013-2023", "42585156": "lumbar laminectomy, 2013-2023",
             "42412075": "cervical disc arthroplasty, 2016-2022"}
LAB = {"A": "Provider-level file; reports procedure volume", "B": "Provider-level file; payments or surgeon characteristics only",
       "C": "National summary file (no provider-row suppression)", "D": "Not classifiable from the abstract"}

if __name__ == "__main__":
    r = pd.read_csv(D / "screened_records.csv", dtype=str).fillna("")
    cls = {**{i: "A" for i in A}, **{i: "B" for i in B}, **{i: "C" for i in C}, **{i: "D" for i in DD}}
    kept = r[r.screen_decision.isin(["include", "unsure"])].copy()
    assert set(kept.pmid) == set(cls), (set(kept.pmid) ^ set(cls))
    kept["group"] = kept.pmid.map(cls)
    kept["group_label"] = kept.group.map(LAB)
    kept["rechecked"] = kept.pmid.map(RECHECKED).fillna("")
    kept = kept.sort_values(["group", "year", "first_author"])
    kept[["pmid", "doi", "first_author", "year", "journal", "title", "file_named_in_abstract", "group", "group_label", "rechecked"]].to_csv(ROOT / "data/interim/provider_file_search.csv", index=False)
    s = pd.Series({"records screened": len(r), "excluded at title and abstract": int((r.screen_decision == "exclude").sum()), "retained for classification": len(kept),
                   **{LAB[g]: int((kept.group == g).sum()) for g in "ABCD"}, "provider-file volume series rechecked": len(RECHECKED)})
    s.rename("n").to_csv(ROOT / "data/interim/provider_file_search_summary.csv", header=True)
    print(s.to_string())
