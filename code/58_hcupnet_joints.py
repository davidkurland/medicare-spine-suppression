"""HCUPnet pull for the joint replacement and cranial comparator lines (companion to 28_hcupnet.py, which writes
hcupnet_long.csv).

Same source, method, states, and years as 28_hcupnet.py: HCUPnet (AHRQ) State Inpatient Databases, discharges by
state x year x expected payer, MS-DRG 469 and 470 (principal). MS-DRGs 521 and 522 begin in FY2024 and fall outside
the 2016-2023 comparison window. Used under the HCUPnet Data Use Agreement: aggregate use only, no analysis
of cells of 10 or fewer, no attempt to identify individuals or establishments. Suppressed cells stay flagged in the output and are excluded downstream
(59_multistate_joints.py).
Citation: HCUPnet, Healthcare Cost and Utilization Project. Agency for Healthcare Research and Quality, Rockville, MD.
https://datatools.ahrq.gov/hcupnet
Also pulls the cranial comparator (python 58_hcupnet_joints.py cranial): MS-DRGs 23 and 25-27, the only cranial DRGs in
HCUPnet's curated MS-DRG list (20-22, 24 and 31-33 are not offered); they cover about 80% of Medicare cranial volume.
Exact labels as listed in the HCUPnet Diagnoses/Procedures filter; DRGs 23 and 469 carry a trailing space.
Output: data/raw/hcupnet/hcupnet_joints_long.csv, hcupnet_cranial_long.csv
"""
import importlib
import sys
import time

import pandas as pd
import requests

h = importlib.import_module("28_hcupnet")
GROUPS = {
    "joints": {469: "469: Hip/knee joint replac/attach low extrem w/MCC ",      # trailing space is HCUPnet's
               470: "470: Hip/knee joint replac/attach low extrem w/o MCC"},
    "cranial": {23: "23: Craniotomy w/device impl/acute CNS PDx w/MCC ",         # trailing space is HCUPnet's
                25: "25: Craniotomy/endovasc intracranial proc w/MCC",
                26: "26: Craniotomy/endovasc intracranial proc w/CC",
                27: "27: Craniotomy/endovasc intracranial proc w/o CC/MCC"},
}
GROUP = next((a for a in sys.argv[1:] if a in GROUPS), "joints")
WANT = GROUPS[GROUP]

if __name__ == "__main__":
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (academic research; aggregate HCUPnet statistics)"
    h.bootstrap_labels(s)   # opens the HCUPnet session
    drg = dict(WANT)   # labels are fixed above because the HCUPnet page does not expose the filter domain
    print("labels used:", drg, flush=True)
    good = {}
    for code, lab in drg.items():
        d = h.fetch({"State": "Arizona", "Year Filter": "2019", "Characteristicsgroup": "Expected Payer", "Label Calc (copy)": lab}, s)
        print("validate", code, len(d), flush=True)
        if len(d):
            good[code] = lab
        time.sleep(1.5)
    if "--probe" in sys.argv or not good:
        sys.exit(0 if good else 1)
    frames = []
    for st in h.STATES:
        d = h.fetch({"State": st, "Year Filter": h.YEARS, "Characteristicsgroup": "Expected Payer",
                     "Label Calc (copy)": ",".join(good.values())}, s)
        frames.append(d.assign(classification="MS-DRG"))
        print(st, len(d), flush=True)
        time.sleep(1.5)
    long = pd.concat([f for f in frames if len(f)], ignore_index=True)
    long.to_csv(h.OUT / f"hcupnet_{GROUP}_long.csv", index=False)
    print(long.shape)
