"""Measured additions to simulated TEAM episode spending: professional fees and readmissions.

Episode spending is the DRG payment plus post-acute payments plus the two measured additions below, which are added
to every episode before the simulation is anchored to CMS's published prices. The capture factor in
51_team_cms_anchor.py covers only what remains unmeasured (other Part B services, payment standardization).

  professional fees  per-case bundles of Part B allowed amounts, 2022-2024, from 84_team_professional_fees.py.
                     A fee for a given code is the same in both settings; cases differ in how much is done.
  readmissions       30-day rate x payment per readmission, data/manual/team_readmission_inputs.json.

Every assumption a bundle needs is in ASSUME with a low, base and high value. Quantities PSPS can show are taken
from PSPS instead (added interspaces per ACDF, added levels per interbody fusion, decompression with interbody
fusion, cages per interspace).

Switches: set_level(fee=..., readmit=..., enabled=...) with fee in {low, base, high, off}, readmit likewise.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"
STATE = {"enabled": True, "fee": "base", "readmit": "base"}

ASSUME = {  # (low, base, high)
    "plate_per_acdf": (0.80, 0.94, 1.00),               # 0.94 = outpatient (22845+22846)/(22551+22554), where nearly every anterior plate is cervical
    "pcf_added_segments": (3.0, 4.5, 6.0),              # posterior cervical fusion spans 4 to 7 levels (authors' clinical estimate); 22614 is shared, so PSPS cannot show it
    "posterior_instrumented": (0.85, 0.95, 1.00),       # PSPS: posterior instrumentation codes / posterior fusion primaries = 0.97
    "pcf_with_decompression": (0.30, 0.60, 0.90),
    "pcf_decompression_added_segments": (2.0, 3.0, 5.0),
    "plf_with_decompression": (0.50, 0.70, 0.90),       # 63047 is shared with stand-alone laminectomy
    "alif_plate": (0.10, 0.30, 0.60),
    "added_levels_multilevel_interbody": (1.0, 1.5, 2.0),      # MS-DRGs 426-428: multilevel lumbar fusion is typically 2 to 3 levels (authors' clinical estimate)
    "added_levels_multilevel_posterolateral": (1.0, 1.5, 2.0), # MS-DRGs 447-448: same
    "cage_scale": (0.70, 1.00, 1.14),                   # multiplies the cage rate PSPS implies (0.88 inpatient, 0.85 outpatient)
}


def set_level(fee=None, readmit=None, enabled=None):
    if fee is not None:
        STATE["fee"] = fee
    if readmit is not None:
        STATE["readmit"] = readmit
    if enabled is not None:
        STATE["enabled"] = enabled


@lru_cache(maxsize=1)
def _table():
    return pd.read_csv(I / "psps_fee_per_case.csv", dtype={"code": str}).set_index(["code", "setting"])


def _a(key):
    return ASSUME[key][{"low": 0, "base": 1, "high": 2}[STATE["fee"]]]


IP, OP = "inpatient", "hospital outpatient"


def A(code, s):
    return float(_table().loc[(code, s), "allowed_per_case"])


def N(code, s):
    return float(_table().loc[(code, s), "cases_3yr"]) if (code, s) in _table().index else 0.0


def observed(s):
    """Quantities PSPS shows directly, by setting."""
    o = {"acdf_added": N("22552", s) / N("22551", s), "tlif_added": N("22634", s) / N("22633", s), "plif_added": N("22632", s) / N("22630", s),
         "alif_added": N("22585", s) / (N("22558", s) + N("22554", s) + N("22556", s)),
         "interbody_decompressed": N("63052", s) / (N("22633", s) + N("22630", s)), "decompression_added": N("63053", s) / N("63052", s),
         "plf_share_of_single_column": N("22612", s) / (N("22612", s) + N("22630", s)), "corpectomy_share": N("22554", s) / (N("22551", s) + N("22554", s))}
    spaces = (N("22551", s) * (1 + o["acdf_added"]) + N("22633", s) * (1 + o["tlif_added"]) + N("22630", s) * (1 + o["plif_added"])
              + N("22558", s) * (1 + o["alif_added"]))
    o["cage_per_interspace"] = min(1.0, N("22853", s) / spaces)
    pcf = 3.0 if s == OP else ASSUME["pcf_added_segments"][1]
    o["plf_added_all_cases"] = max(0.0, (N("22614", s) - pcf * N("22600", s)) / (N("22612", s) + N("22610", s)))
    return o


def cervical_fees():
    out = {}
    for s, key in [(IP, "acdf_ip"), (OP, "acdf_op")]:
        o = observed(s); cage = min(1.0, o["cage_per_interspace"] * _a("cage_scale"))
        acdf = A("22551", s) + o["acdf_added"] * A("22552", s) + _a("plate_per_acdf") * A("22845", s) + cage * (1 + o["acdf_added"]) * A("22853", s)
        corp = (A("63081", s) + A("22554", s) + (N("63082", s) / N("63081", s)) * A("63082", s) + _a("plate_per_acdf") * A("22845", s) + A("22854", s))
        out[key] = (1 - o["corpectomy_share"]) * acdf + o["corpectomy_share"] * corp + A("00670", s)
    dec = A("63045", IP) + _a("pcf_decompression_added_segments") * A("63048", IP)
    out["pcf_ip"] = (A("22600", IP) + _a("pcf_added_segments") * A("22614", IP) + _a("posterior_instrumented") * A("22842", IP)
                     + _a("pcf_with_decompression") * dec + A("00670", IP))
    return out


def lumbar_fee(kind, approach, s, added=None):
    """kind: comb_single (MS-DRG 402), comb_multi (426-428), single (450, 451), multi (447, 448).
    approach: posterior, anterior, thoracic, other. `added` overrides the added levels (used for outpatient cases)."""
    o = observed(s); cage = min(1.0, o["cage_per_interspace"] * _a("cage_scale")); inst = _a("posterior_instrumented")
    multi = kind in ("comb_multi", "multi")
    if added is None:
        added = {"comb_single": 0.0, "single": 0.0, "comb_multi": _a("added_levels_multilevel_interbody"), "multi": _a("added_levels_multilevel_posterolateral")}[kind]
    rod = A("22842", s) if (multi or added >= 1) else A("22840", s)
    dec_ib = o["interbody_decompressed"] * (A("63052", s) + min(added, 2.0) * A("63053", s))
    tlif = A("22633", s) + added * A("22634", s) + cage * (1 + added) * A("22853", s) + inst * rod + dec_ib
    plif = A("22630", s) + added * A("22632", s) + cage * (1 + added) * A("22853", s) + inst * rod + dec_ib
    plf = A("22612", s) + added * A("22614", s) + inst * rod + _a("plf_with_decompression") * (A("63047", s) + added * A("63048", s))
    alif = A("22558", s) + added * A("22585", s) + cage * (1 + added) * A("22853", s)
    if approach == "posterior":
        w = o["plf_share_of_single_column"]
        fee = tlif if kind.startswith("comb") else w * plf + (1 - w) * plif
    elif approach == "anterior":      # anterior interbody, with a posterior fusion on the combined lines
        fee = alif + (A("22612", s) + added * A("22614", s) + inst * rod if kind.startswith("comb") else _a("alif_plate") * A("22845", s))
    elif approach == "thoracic":
        fee = A("22610", IP) + added * A("22614", s) + inst * rod + _a("plf_with_decompression") * (A("63047", s) + added * A("63048", s))
    else:
        return 0.5 * (lumbar_fee(kind, "posterior", s, added) + lumbar_fee(kind, "anterior", s, added))
    return fee + A("00670", s)


@lru_cache(maxsize=8)
def _readmit(level):
    if level == "off":
        return 0.0, 0.0
    v = json.load(open(ROOT / "data/manual/team_readmission_inputs.json"))["values"]
    pay = v["payment_per_readmission"][level]
    return pay * v["rate_acdf_inpatient"][level], pay * v["rate_acdf_outpatient"][level]   # posterior fusion takes the inpatient rate in every scenario


def readmission(setting):
    ip, op = _readmit(STATE["readmit"])
    return ip if setting == IP else op


# ---------------------------------------------------------------- what the simulation scripts call
def cervical_inpatient(d):
    """Dollars to add to each inpatient cervical episode (rows of the Texas cohort; uses d.family)."""
    if not STATE["enabled"]:
        return np.zeros(len(d))
    add = np.full(len(d), readmission(IP))
    if STATE["fee"] != "off":
        f = cervical_fees(); fam = d.family.values
        ant, post = (fam == "cervical_anterior"), (fam == "cervical_posterior")
        mix = post.sum() / max(post.sum() + ant.sum(), 1)
        add += np.select([ant, post], [f["acdf_ip"], f["pcf_ip"]], (1 - mix) * f["acdf_ip"] + mix * f["pcf_ip"])
    return add


def cervical_outpatient():
    if not STATE["enabled"]:
        return 0.0
    return readmission(OP) + (cervical_fees()["acdf_op"] if STATE["fee"] != "off" else 0.0)


KIND = {402: "comb_single", 426: "comb_multi", 427: "comb_multi", 428: "comb_multi", 450: "single", 451: "single", 447: "multi", 448: "multi"}
APPROACH = {"lumbar_posterior": "posterior", "lumbar_anterior": "anterior", "thoracic_other": "thoracic"}


def lumbar_inpatient(d):
    """Dollars to add to each inpatient noncervical episode (uses d.drg25 and d.family)."""
    if not STATE["enabled"]:
        return np.zeros(len(d))
    add = np.full(len(d), readmission(IP))
    if STATE["fee"] != "off":
        key = list(zip(d.drg25.map(KIND).fillna("single"), d.family.map(APPROACH).fillna("other")))
        fees = {k: lumbar_fee(k[0], k[1], IP) for k in set(key)}
        add += np.array([fees[k] for k in key])
    return add


def lumbar_outpatient(drg):
    """Outpatient anchors: 22633 maps to MS-DRG 402; 22612 and 22630 map to MS-DRG 451. TEAM maps every outpatient case to
    the single-level line whatever its length, so the added levels observed in outpatient claims are kept."""
    if not STATE["enabled"]:
        return 0.0
    if STATE["fee"] == "off":
        return readmission(OP)
    o = observed(OP)
    if drg == 402:
        fee = lumbar_fee("comb_single", "posterior", OP, added=o["tlif_added"])
    else:
        # added levels differ by code, so the PLF fee is rebuilt with the observed added levels
        fee = lumbar_fee("single", "posterior", OP, added=o["plf_added_all_cases"])
    return fee + readmission(OP)


def summary():
    rows = []
    f = cervical_fees()
    rows += [("cervical", "outpatient ACDF", f["acdf_op"]), ("cervical", "inpatient ACDF", f["acdf_ip"]), ("cervical", "inpatient posterior cervical fusion", f["pcf_ip"])]
    for kind, lab in [("comb_single", "MS-DRG 402, single-level combined"), ("single", "MS-DRGs 450-451, single level"),
                      ("comb_multi", "MS-DRGs 426-428, multilevel combined"), ("multi", "MS-DRGs 447-448, multilevel")]:
        for ap in ("posterior", "anterior"):
            rows.append(("lumbar", f"{lab}, inpatient {ap}", lumbar_fee(kind, ap, IP)))
    rows += [("lumbar", "outpatient TLIF (22633)", lumbar_outpatient(402) - readmission(OP)), ("lumbar", "outpatient posterolateral or PLIF (22612, 22630)", lumbar_outpatient(451) - readmission(OP)),
             ("both", "readmission, inpatient episode", readmission(IP)), ("both", "readmission, outpatient episode", readmission(OP))]
    return pd.DataFrame(rows, columns=["model", "group", "dollars"])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out = None
    for lv in ("low", "base", "high"):
        set_level(fee=lv, readmit=lv)
        s = summary().rename(columns={"dollars": lv})
        out = s if out is None else out.merge(s, on=["model", "group"])
    set_level(fee="base", readmit="base")
    out.to_csv(I / "fee_bundles.csv", index=False)
    print(out.round(0).to_string(index=False))
    print("\nobserved in PSPS, inpatient:", {k: round(v, 2) for k, v in observed(IP).items()})
    print("observed in PSPS, outpatient:", {k: round(v, 2) for k, v in observed(OP).items()})
