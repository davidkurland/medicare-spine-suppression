"""Hidden share of Medicare FFS discharges (suppressed in the public provider file) for each TEAM episode type,
2013-2024, with spinal fusion and cranial surgery for comparison.

Hidden share = 1 - (sum of published provider rows) / (national geography total), by DRG group and year.
TEAM episode-type DRGs (FY2025 grouper): spinal fusion 402, 426-430, 447-448, 450-451, 471-473 (pre-FY2025 453-455,
459-460 and 471-473); lower extremity joint replacement 469-470, 521-522; surgical hip/femur fracture 480-482;
CABG 231-236; major bowel 329-331.
Outputs: data/interim/hidden_share_team_episodes.csv, manuscript/figures/eFigure_hidden_share_episodes.png/.pdf
"""
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
FIG = ROOT / "manuscript/figures"
FIG.mkdir(parents=True, exist_ok=True)
GROUPS = {
    "Spinal fusion": [453, 454, 455, 459, 460, 471, 472, 473, 402, 426, 427, 428, 429, 430, 447, 448, 450, 451],
    "Joint replacement": [469, 470, 521, 522],
    "Hip/femur fracture": [480, 481, 482],
    "CABG": [231, 232, 233, 234, 235, 236],
    "Major bowel": [329, 330, 331],
    "Cranial (comparator)": [20, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33],
}
COL = {"Spinal fusion": "#D55E00", "Joint replacement": "#0072B2", "Hip/femur fracture": "#009E73", "CABG": "#CC79A7",
       "Major bowel": "#E69F00", "Cranial (comparator)": "#7F7F7F"}
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.6,
                     "pdf.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight"})


def col(df, *keys):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keys):
            return c
    raise KeyError(keys)


def published():
    rows = []
    for f in sorted((ROOT / "data/raw/cms_mupinp").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        drg, n = col(d, "DRG_Cd"), col(d, "Tot_Dschrgs")
        d = d[[drg, n]].rename(columns={drg: "drg", n: "n"})
        d["drg"] = pd.to_numeric(d.drg, errors="coerce"); d["n"] = pd.to_numeric(d.n, errors="coerce")
        d["year"] = y
        rows.append(d)
    return pd.concat(rows).groupby(["year", "drg"]).n.sum().rename("published")


def complete():
    rows = []
    for f in sorted((ROOT / "data/raw/cms_mupinp_geo").glob("*.CSV")):
        d = pd.read_csv(f, dtype=str, encoding="latin-1")
        y = 2000 + int(re.search(r"DY(\d\d)", f.name).group(1))
        lvl = col(d, "Geo_Lvl")
        d = d[d[lvl].str.strip().str.lower() == "national"]
        drg, n = col(d, "DRG_Cd"), col(d, "Tot_Dschrgs")
        d = d[[drg, n]].rename(columns={drg: "drg", n: "n"})
        d["drg"] = pd.to_numeric(d.drg, errors="coerce"); d["n"] = pd.to_numeric(d.n, errors="coerce")
        d["year"] = y
        rows.append(d)
    return pd.concat(rows).groupby(["year", "drg"]).n.sum().rename("total")


if __name__ == "__main__":
    m = pd.concat([published(), complete()], axis=1).reset_index()
    m["group"] = None
    for g, drgs in GROUPS.items():
        m.loc[m.drg.isin(drgs), "group"] = g
    t = m.dropna(subset=["group"]).groupby(["group", "year"])[["published", "total"]].sum().reset_index()
    t["hidden_share"] = 1 - t.published / t.total
    t.to_csv(I / "hidden_share_team_episodes.csv", index=False)
    w = t.pivot(index="year", columns="group", values="hidden_share")
    print((w * 100).round(1).to_string())
    print("\nnational totals 2024:", t[t.year == 2024].set_index("group").total.astype(int).to_dict())

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ends = []
    for g in GROUPS:
        s = w[g].dropna()
        ax.plot(s.index, s * 100, color=COL[g], marker="o", ms=2.3, lw=1.6 if g in ("Spinal fusion", "Cranial (comparator)") else 1.3)
        ends.append([s.index[-1] + 0.3, s.iloc[-1] * 100, g])
    ends.sort(key=lambda e: e[1])
    for j in range(1, len(ends)):
        ends[j][1] = max(ends[j][1], ends[j - 1][1] + 3.2)
    for x, y, g in ends:
        ax.text(x, y, f"{g}  {w[g].dropna().iloc[-1] * 100:.0f}%", color=COL[g], fontsize=6.5, va="center")
    ax.set_xlim(2012.6, 2025.4)
    ax.set_xticks([2013, 2016, 2019, 2022, 2024])
    ax.set_ylim(0, max(w.max()) * 100 + 5)
    ax.set_ylabel("FFS discharges hidden by suppression, %")
    ax.set_title("Hidden share of complete FFS volume, by TEAM episode type", loc="left", fontweight="bold")
    fig.savefig(FIG / "eFigure_hidden_share_episodes.png"); fig.savefig(FIG / "eFigure_hidden_share_episodes.pdf")
    print("saved")
