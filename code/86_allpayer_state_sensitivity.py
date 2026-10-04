"""How far the 35-state all-payer fusion result depends on individual states.

HCUPnet state counts are statistics reported by each state's data organization. A state whose all-Medicare count
(FFS plus Medicare Advantage) fell MORE than its complete FFS count is suspect: Medicare Advantage volume would have had
to fall while Medicare Advantage enrollment grew. A state bootstrap does not capture that kind of reporting error, so
we report the pooled change with such states removed, with extreme states removed, and the median state.
Output: data/interim/allpayer_state_sensitivity.csv, allpayer_by_state.csv
"""
import importlib

import numpy as np
import pandas as pd

from common import ROOT

r82 = importlib.import_module("82_uncertainty_sensitivity")
I = ROOT / "data/interim"

if __name__ == "__main__":
    w = r82.balanced(pd.read_csv(I / "multistate_panel.csv").query("category == 'fusion'"))
    p = w.pivot_table(index="st", columns="year", values=["hcup_medicare", "cms_ffs_total", "cms_ffs_published", "TOT_BENES"])
    st = pd.DataFrame({"allpayer_count": p["hcup_medicare"][2023] / p["hcup_medicare"][2016] - 1,
                       "allpayer_per_beneficiary": (p["hcup_medicare"][2023] / p["TOT_BENES"][2023]) / (p["hcup_medicare"][2016] / p["TOT_BENES"][2016]) - 1,
                       "cms_complete_count": p["cms_ffs_total"][2023] / p["cms_ffs_total"][2016] - 1,
                       "cms_published_count": p["cms_ffs_published"][2023] / p["cms_ffs_published"][2016] - 1,
                       "allpayer_2016": p["hcup_medicare"][2016]})
    st["allpayer_fell_more_than_ffs"] = st.allpayer_count < st.cms_complete_count
    # an all-Medicare count below the complete FFS count in a compared year cannot be right if the 2 sources cover the same hospitals
    st["allpayer_below_ffs_in_a_year"] = (p["hcup_medicare"][2016] < p["cms_ffs_total"][2016]) | (p["hcup_medicare"][2023] < p["cms_ffs_total"][2023])
    st.to_csv(I / "allpayer_by_state.csv")
    rows = []

    def add(label, keep):
        r = r82.rate_changes(w[w.st.isin(keep)])
        rows.append({"states": label, "n_states": len(keep), **{k: r[k] for k in ("allpayer_count", "allpayer_per_beneficiary", "cms_complete_count", "cms_published_count", "cms_published_per_ffs_enrollee")}})
    add("all", list(st.index))
    add("without states where all-Medicare fell more than complete FFS", list(st.index[~st.allpayer_fell_more_than_ffs]))
    add("without states where all-Medicare was below complete FFS in 2016 or 2023", list(st.index[~st.allpayer_below_ffs_in_a_year]))
    add("without states changing more than 40% either way", list(st.index[st.allpayer_count.abs() <= 0.40]))
    add("without the 5 largest states", list(st.sort_values("allpayer_2016").index[:-5]))
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"states": "median state (unweighted)", "n_states": len(st), "allpayer_count": st.allpayer_count.median(),
                         "allpayer_per_beneficiary": st.allpayer_per_beneficiary.median(), "cms_complete_count": st.cms_complete_count.median(),
                         "cms_published_count": st.cms_published_count.median(), "cms_published_per_ffs_enrollee": np.nan}
    out.to_csv(I / "allpayer_state_sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print("\nstate range, all-payer count: %s %.1f%% to %s %.1f%%" % (st.allpayer_count.idxmin(), 100 * st.allpayer_count.min(), st.allpayer_count.idxmax(), 100 * st.allpayer_count.max()))
    print("states where all-Medicare fell more than complete FFS:", sorted(st.index[st.allpayer_fell_more_than_ffs]))
    print("states where published fell more than all-Medicare:", int((st.cms_published_count < st.allpayer_count).sum()), "of", len(st))
