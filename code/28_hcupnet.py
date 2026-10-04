"""HCUPnet (AHRQ) state inpatient statistics: all-payer SID discharges by state x year x category x expected payer.

Used under the HCUPnet Data Use Agreement: aggregate use only, no analysis of cells of 10 or fewer,
no attempts to identify individuals or establishments. Citation: HCUPnet, Healthcare Cost and Utilization
Project. Agency for Healthcare Research and Quality, Rockville, MD. https://datatools.ahrq.gov/hcupnet
Access: Tableau Server view CSV export with URL filters, one state per request, polite pauses.
"Medicare" = fee-for-service + Medicare Advantage (first-listed expected payer), all ages.
Categories: procedure CCSR (all-listed, operating room; 2016+) and MS-DRG (principal; curated list).
Output: data/raw/hcupnet/hcupnet_long.csv. The extracts for the comparator lines in the multistate panel
(hcupnet_joints_long.csv, hcupnet_cranial_long.csv) are written by 58_hcupnet_joints.py.
"""
import io
import re
import time
import urllib.parse

import pandas as pd
import requests

from common import ROOT

OUT = ROOT / "data/raw/hcupnet"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://dataviz.ahrq.gov"
WB, VIEW = "HCUPnet_Analysis_SID_Classifications_v2_1", "DescriptiveStats"
STATES = ["Alaska", "Arizona", "Arkansas", "California", "Colorado", "Delaware", "Florida", "Hawaii", "Illinois", "Indiana",
          "Iowa", "Kansas", "Kentucky", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
          "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
          "North Carolina", "North Dakota", "Oklahoma", "Oregon", "Rhode Island", "South Carolina", "South Dakota",
          "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
YEARS = ",".join(str(y) for y in range(2013, 2025))
CCSR_WANTED = {"MST013": "spine", "MST016": "spine", "CNS008": "spine", "CNS007": "cranial", "CNS004": "cranial", "CNS010": "cranial"}
DRG_WANTED = {23: "cranial", 25: "cranial", 26: "cranial", 27: "cranial", 29: "spine", 30: "spine", 454: "spine", 455: "spine",
              460: "spine", 472: "spine", 473: "spine", 519: "spine", 520: "spine"}
CCSR_TYPE = "Procedure--Clinical Classification Software Refined (CCSR) (operating room only)"
CCSR_FALLBACK = {"MST013": "MST013: Spine fusion", "MST016": "MST016: Vertebral discectomy",
                 "CNS008": "CNS008: Spinal cord decompression", "CNS007": "CNS007: CNS excision procedures",
                 "CNS004": "CNS004: Intracranial epidural and subdural space drainage",
                 "CNS010": "CNS010: Cerebrospinal fluid shunt procedures"}


def q(params: dict) -> str:
    return "&".join(f"{urllib.parse.quote(k)}={urllib.parse.quote(v, safe=',')}" for k, v in params.items())


def fetch(params: dict, s: requests.Session) -> pd.DataFrame:
    url = f"{BASE}/views/{WB}/{VIEW}.csv?{q(params)}"
    for attempt in range(3):
        r = s.get(url, timeout=120)
        if r.status_code == 200 and len(r.content) > 5:
            return pd.read_csv(io.StringIO(r.text), dtype=str)
        time.sleep(10 * (attempt + 1))
    return pd.DataFrame()


def bootstrap_labels(s: requests.Session) -> set:
    """Read filter domains (DRG / CCSR labels) from the Tableau bootstrap response."""
    s.get(f"{BASE}/views/{WB}/{VIEW}?:embed=yes&:showVizHome=no", timeout=60)
    r = s.post(f"{BASE}/vizql/w/{WB}/v/{VIEW}/startSession/viewing?:embed=yes", timeout=60)
    sid = r.headers.get("x-session-id")
    if not sid:
        return set()
    b = s.post(f"{BASE}/vizql/w/{WB}/v/{VIEW}/bootstrapSession/sessions/{sid}", timeout=120, data={
        "worksheetPortSize": '{"w":1200,"h":1000}', "dashboardPortSize": '{"w":1200,"h":1000}',
        "clientDimension": '{"w":1200,"h":1000}', "renderMapsClientSide": "true", "isBrowserRendering": "true",
        "browserRenderingThreshold": "100", "formatDataValueLocally": "false", "navType": "Nav", "navSrc": "Boot",
        "devicePixelRatio": "1", "clientRenderPixelLimit": "25000000", "sheet_id": "Descriptive Stats",
        "showParams": '{"checkpoint":false,"refresh":false,"refreshUnmodified":false}', "filterTileSize": "200",
        "locale": "en_US", "language": "en", "verboseMode": "false", ":session_feature_flags": "{}", "keychain_version": "1"})
    return set(re.findall(r'"(\d{1,3}: [^"\\]+)"', b.text)) | set(re.findall(r'"([A-Z]{3}\d{3}: [^"\\]+)"', b.text))


if __name__ == "__main__":
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (academic research; aggregate HCUPnet statistics)"
    labels = bootstrap_labels(s)
    drg_labels = {int(l.split(":")[0]): l for l in labels if re.match(r"^\d{1,3}: ", l) and int(l.split(":")[0]) in DRG_WANTED}
    ccsr_labels = {l.split(":")[0]: l for l in labels if re.match(r"^[A-Z]{3}\d{3}: ", l) and l.split(":")[0] in CCSR_WANTED}
    for k, v in CCSR_FALLBACK.items():
        ccsr_labels.setdefault(k, v)
    print("DRG labels found:", drg_labels)
    print("CCSR labels:", ccsr_labels)
    if 460 not in drg_labels:
        drg_labels[460] = "460: Spinal fusion except cervical without MCC"

    # validate each label on one state-year; drop labels that return nothing
    good_drg, good_ccsr = {}, {}
    for code, lab in drg_labels.items():
        d = fetch({"State": "Arizona", "Year Filter": "2019", "Characteristicsgroup": "Expected Payer", "Label Calc (copy)": lab}, s)
        if len(d):
            good_drg[code] = lab
        time.sleep(1.5)
    for code, lab in ccsr_labels.items():
        d = fetch({"State": "Arizona", "Year Filter": "2019", "Characteristicsgroup": "Expected Payer", "Classification Type": CCSR_TYPE,
                   "All Listed or Principal": "All-Listed", "Label Calc (copy)": lab}, s)
        if len(d):
            good_ccsr[code] = lab
        time.sleep(1.5)
    print("validated DRG:", sorted(good_drg), "validated CCSR:", sorted(good_ccsr), flush=True)

    frames = []
    for st in STATES:
        if good_drg:
            d = fetch({"State": st, "Year Filter": YEARS, "Characteristicsgroup": "Expected Payer",
                       "Label Calc (copy)": ",".join(good_drg.values())}, s)
            frames.append(d.assign(classification="MS-DRG"))
            time.sleep(1.5)
        if good_ccsr:
            d = fetch({"State": st, "Year Filter": YEARS, "Characteristicsgroup": "Expected Payer", "Classification Type": CCSR_TYPE,
                       "All Listed or Principal": "All-Listed", "Label Calc (copy)": ",".join(good_ccsr.values())}, s)
            frames.append(d.assign(classification="CCSR"))
            time.sleep(1.5)
        print(st, sum(len(f) for f in frames[-2:]), flush=True)
    long = pd.concat([f for f in frames if len(f)], ignore_index=True)
    long.to_csv(OUT / "hcupnet_long.csv", index=False)
    print(long.shape)
