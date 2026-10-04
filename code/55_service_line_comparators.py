"""Three service lines as comparators for the measurement claim, plus two supporting analyses.

Service lines (2013-2024, national Medicare FFS):
  Spine            partial migration, split by anatomy: some operations have an outpatient anchor, others do not
  Joint replacement migration essentially complete (knee off the inpatient-only list 2018, hip 2020)
  Cranial          migration never applied

For each line: published provider-file rows, complete geography totals, and complete per FFS Part A enrollee.
The gap between published and complete is the measurement artifact; it is large for spine, small for joints.

Positive control: a published independent reconstruction of the joint-replacement series from the same public files
(Pean C. The Hospital Is Losing the Joint. The Techy Surgeon, 18 Sep 2026) reports 542,513 discharges in 2017
falling to 127,759 in 2024. The script compares the complete totals computed here for the same DRGs and years.

Composition: change in average Medicare payment per inpatient discharge decomposed into within-DRG payment
change and case-mix shift, exactly (Shapley-symmetric shift-share, no residual):
    dP = SUM_d [ ds_d * (p_d0 + p_d1)/2 ]  +  SUM_d [ (s_d0 + s_d1)/2 * dp_d ]
         \_____ case mix _____/               \_____ within-DRG rate _____/
Payments are nominal; the rate term therefore carries annual IPPS update factors as well as real price change.

Outputs: data/interim/service_line_comparators.csv, data/interim/joints_positive_control.csv,
         data/interim/payment_composition.csv
"""
import importlib
import re

import numpy as np
import pandas as pd

from common import ROOT

hs = importlib.import_module("52_hidden_share_team_episodes")
I = ROOT / "data/interim"

SPINE = [453, 454, 455, 456, 457, 458, 459, 460, 471, 472, 473,
         402, 426, 427, 428, 429, 430, 447, 448, 450, 451,
         28, 29, 30, 490, 491, 518, 519, 520]
JOINT = [469, 470, 521, 522]
CRANIAL = [20, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33]
LINES = {"Spine": SPINE, "Joint replacement": JOINT, "Cranial": CRANIAL}


def payments():
    """National average Medicare payment and discharges by DRG and year, from the geography file."""
    rows = []
    for f in sorted((ROOT / "data/raw/cms_mupinp_geo").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        lvl = hs.col(d, "Geo_Lvl")
        d = d[d[lvl].str.strip().str.lower() == "national"]
        drg, n, pay = hs.col(d, "DRG_Cd"), hs.col(d, "Tot_Dschrgs"), hs.col(d, "Avg_Mdcr_Pymt")
        d = d[[drg, n, pay]].rename(columns={drg: "drg", n: "n", pay: "pay"})
        for c in ("drg", "n", "pay"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["year"] = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        rows.append(d.dropna(subset=["drg", "n"]))
    return pd.concat(rows, ignore_index=True)


def decompose(p, drgs, y0, y1):
    """Exact shift-share of the change in average Medicare payment per discharge."""
    d = p[p.drg.isin(drgs) & p.year.isin([y0, y1])]
    w = d.pivot_table(index="drg", columns="year", values=["n", "pay"]).fillna(0.0)
    n0, n1 = w[("n", y0)], w[("n", y1)]
    p0, p1 = w[("pay", y0)], w[("pay", y1)]
    s0, s1 = n0 / n0.sum(), n1 / n1.sum()
    # a DRG absent in one year contributes through its share only; carry the other year's payment across
    p0 = p0.where(p0 > 0, p1)
    p1 = p1.where(p1 > 0, p0)
    mix = ((s1 - s0) * (p0 + p1) / 2).sum()
    rate = ((s0 + s1) / 2 * (p1 - p0)).sum()
    P0, P1 = (s0 * p0).sum(), (s1 * p1).sum()
    return dict(year_start=y0, year_end=y1, avg_pymt_start=P0, avg_pymt_end=P1, change=P1 - P0,
                case_mix=mix, within_drg_rate=rate, case_mix_pct=100 * mix / (P1 - P0),
                rate_pct=100 * rate / (P1 - P0))


if __name__ == "__main__":
    pub, tot = hs.published(), hs.complete()
    ffs = pd.read_csv(I / "national_sensitivity.csv").set_index("year").ffs_part_a_benes

    rows = []
    for name, drgs in LINES.items():
        for y in range(2013, 2025):
            pb = pub.reindex([(y, d) for d in drgs]).sum()
            tt = tot.reindex([(y, d) for d in drgs]).sum()
            rows.append(dict(line=name, year=y, published=pb, complete=tt,
                             hidden_share=1 - pb / tt, per_1k_ffs=1000 * tt / ffs.get(y, float("nan"))))
    s = pd.DataFrame(rows)
    s.to_csv(I / "service_line_comparators.csv", index=False)

    j = s[(s.line == "Joint replacement") & s.year.isin([2017, 2024])].set_index("year")
    pc = pd.DataFrame([
        dict(source="This study, complete geography totals", y2017=j.complete[2017], y2024=j.complete[2024]),
        dict(source="Independent reconstruction (Pean 2026)", y2017=542513, y2024=127759),
    ])
    pc["change_pct"] = 100 * (pc.y2024 / pc.y2017 - 1)
    pc.to_csv(I / "joints_positive_control.csv", index=False)

    # how much of each line's PUBLISHED decline is suppression rather than surgery
    sup = []
    for name in LINES:
        d = s[s.line == name].set_index("year")
        lp = np.log(d.published[2024] / d.published[2013])
        lt = np.log(d.complete[2024] / d.complete[2013])
        sup.append(dict(line=name, published_change=d.published[2024] / d.published[2013] - 1,
                        complete_change=d.complete[2024] / d.complete[2013] - 1,
                        suppression_pct_of_published_change=100 * (lp - lt) / lp))
    sup = pd.DataFrame(sup)
    sup.to_csv(I / "suppression_share_by_line.csv", index=False)
    # the same shares ending in 2023, the last year before the FY2025 fusion DRG regrouping reached the file
    sup23 = []
    for name in LINES:
        d = s[s.line == name].set_index("year")
        lp = np.log(d.published[2023] / d.published[2013]); lt = np.log(d.complete[2023] / d.complete[2013])
        sup23.append(dict(line=name, published_change=d.published[2023] / d.published[2013] - 1, complete_change=d.complete[2023] / d.complete[2013] - 1,
                          suppression_pct_of_published_change=100 * (lp - lt) / lp))
    pd.DataFrame(sup23).to_csv(I / "suppression_share_by_line_2013_2023.csv", index=False)
    print("\nSUPPRESSION SHARE OF THE PUBLISHED CHANGE\n", sup.round(3).to_string(index=False))

    # cell census for the DRGs used as worked examples: how many hospitals still print a row
    cells = []
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1", low_memory=False)
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        drg, n = hs.col(d, "DRG_Cd"), hs.col(d, "Tot_Dschrgs")
        d = d[[drg, n]].rename(columns={drg: "drg", n: "n"})
        d["drg"] = pd.to_numeric(d.drg, errors="coerce"); d["n"] = pd.to_numeric(d.n, errors="coerce")
        for t in (471, 472, 473, 518):
            g = d[d.drg == t].dropna(subset=["n"])
            cells.append(dict(drg=t, year=y, hospitals_with_row=len(g), published=g.n.sum(),
                              complete=float(tot.get((y, t), float("nan"))),
                              median_cell=g.n.median(), cells_11_to_15=int(((g.n >= 11) & (g.n <= 15)).sum())))
    cc = pd.DataFrame(cells)
    cc["hidden_share"] = 1 - cc.published / cc.complete
    cc.to_csv(I / "drg_cell_census.csv", index=False)
    print("\nCELL CENSUS (worked-example DRGs)\n",
          cc[cc.drg.isin([473, 518]) & cc.year.isin([2016, 2019, 2024])].round(3).to_string(index=False))

    p = payments()
    comp = pd.DataFrame([dict(line=k, **decompose(p, v, 2013, 2024)) for k, v in LINES.items()])
    comp.to_csv(I / "payment_composition.csv", index=False)

    print(s[s.year.isin([2013, 2017, 2024])].to_string(index=False))
    print("\nPOSITIVE CONTROL\n", pc.to_string(index=False))
    print("\nPAYMENT COMPOSITION 2013-2024\n", comp.round(1).to_string(index=False))
