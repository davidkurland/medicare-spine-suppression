# Where Did the Spine Surgery Go?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23134875.svg)](https://doi.org/10.5281/zenodo.23134875)

Analysis code, hospital crosswalks, and download manifests for:

> Kurland DB, Sastry R, Niu T, Oyelese A. Where Did the Spine Surgery Go? Small-Cell Suppression, Medicare
> Advantage, and Pooled Episode Pricing in Medicare Inpatient Spine Surgery, 2013-2024.

Medicare's public hospital file shows inpatient spine surgery falling by half from 2013 to 2024. The study
measures how much of that fall reflects fewer operations, which cases remain inpatient, and how the
Transforming Episode Accountability Model (TEAM) prices them when outpatient and inpatient cases share one
target.

## What is here

| Path | Contents |
|---|---|
| `code/` | The analysis and figure scripts, a shared helper module, and pinned requirements |
| `data/manual/` | Hospital crosswalks, the hand-verified linkage corrections, simulation inputs, and the literature search log |
| `data/source_manifests/` | The download URLs for the public CMS and New York files |

## What is not here, and why

**No source data.** Two sources are governed by data use agreements that permit aggregate reporting only, so
no record-level file and no intermediate derived from one is distributed here:

- **Texas Hospital Inpatient Discharge Public Use Data File**, 2016-2019. Texas Department of State Health
  Services. Obtain from the Texas Health Care Information Collection under its data use agreement:
  https://www.dshs.texas.gov/center-health-statistics/texas-health-care-information-collection
- **HCUPnet**, Agency for Healthcare Research and Quality. Aggregate use only, no analysis of cells of 10 or
  fewer, no attempt to identify individuals or establishments: https://datatools.ahrq.gov/hcupnet

The remaining sources are public and free:

- CMS Medicare Inpatient Hospitals **by Provider and Service** (the suppressed hospital-level file)
- CMS Medicare Inpatient Hospitals **by Geography and Service** (complete totals)
- CMS Medicare Monthly Enrollment
- CMS Physician/Supplier Procedure Summary (PSPS), Part B services by place of service
- CMS Physician & Other Practitioners by Provider and Service, via the data.cms.gov API
- CMS Healthcare Cost Report Information System (HCRIS)
- CMS OPPS final rule addenda (Addendum B and the Inpatient-Only list in Addendum E), TEAM performance year 1
  target prices and specifications, and the CMS-HCC ICD-10-CM mappings
- New York SPARCS de-identified hospital inpatient discharges and facility files: https://health.data.ny.gov
- Indiana hospital discharge data files: https://www.in.gov/health/oda/hospital-discharge-data/
- Florida FloridaHealthFinder query tool: https://quality.healthfinder.fl.gov/QueryTool/QTResults

Place downloads under `data/raw/<source>/`, following the folder names in `data/source_manifests/`. Three
inputs come from interactive tools rather than file downloads: the HCUPnet statistics (`28_hcupnet.py` and
`58_hcupnet_joints.py` query the public HCUPnet interface and write `data/raw/hcupnet/`), the Florida counts
(`27_florida.py` queries FloridaHealthFinder; save the tool's report-choice list as
`data/raw/states/FL/fl_reportChoices.json` first), and the Part B provider-file series for the published-study
rechecks (`81_lopez_acdf_provider_file.py` and `87_dinesh_recheck.py` call the data.cms.gov API).

Intermediate outputs regenerate into `data/interim/`, figures into `manuscript/figures/` (the diagnostic plots of
`05_ny_external_validity.py` into `figures/`). These folders are excluded from version control. `data/interim/` holds record-level Texas files and Indiana hospital cells of
1 to 10 discharges; keep it local and never share it.

## The crosswalks are the part you cannot regenerate

Linking state hospital identifiers to CMS certification numbers cannot be done reliably by name matching.
Specialty orthopedic hospitals and multi-campus systems break it. `data/manual/` holds the corrections that
were verified by hand:

- `IN_crosswalk_automated.csv` and `IN_crosswalk_overrides.csv`: the automated Indiana match (which
  `24_indiana_linkage_payer.py` regenerates) and the 18 hand-verified corrections applied on top of it. Rank concordance between state and CMS spine volume rises from
  0.07-0.23 under automated matching alone to 0.76-0.88 after these corrections. Three corrections verified
  by address are dropped in sensitivity scenarios.
- `NY_ccn_consolidation.csv`: New York facilities that begin reporting under a parent system's CMS
  certification number partway through the study period.
- `IN_rank_agreement_automated_crosswalk.csv`: rank agreement under the automated crosswalk alone, the statistics that
  justify the corrections; `24_indiana_linkage_payer.py` recomputes it and writes the comparison.

## Three things a reuser should know before trusting a Medicare volume series

1. **The hospital and Part B provider files delete any cell below 11 cases and leave no marker.** The
   provider reads as zero. Coverage moves with volume, so the published series exaggerates change in
   whichever direction volume is heading. Use the **by Geography and Service** totals, which are complete,
   or reconstruct PSPS counts from the payment fields, which survive when the service count is blanked.
2. **Medicare Advantage enrollees never appear in fee-for-service files.** Their share of Part A
   beneficiaries rose from 28% in 2013 to 50% in 2024. Report rates per enrollee, and state the denominator.
3. **A per-enrollee denominator fixes the size of the pool, not its composition.** Test the residual with a
   negative control: a procedure that shares the denominator but cannot share the exposure.

## Pipeline

Scripts read `code/common.py`, which resolves paths relative to the project root. Each script's header
describes what it does and what it writes. The numeric prefix gives a working order.

| Stage | Scripts |
|---|---|
| State data and hospital linkage (New York, Indiana) | `01`, `03`, `04`, `07`, `09`, `10`, `12`, `24`, `25`, `43`, `61` |
| National CMS panel, exact suppression, and the CY2024 split year | `08`, `16`, `21`, `21b`, `22`, `52`, `93`, `97` |
| Surgical setting (PSPS) and the Inpatient-Only list | `23`, `33`, `38`, `95` |
| Texas, Florida, and the multistate all-payer comparison | `26`-`29`, `34`, `35`, `44`, `58`, `59`, `86` |
| Case mix and hospitals missing from the public file | `39`-`41`, `55`, `57` |
| TEAM pricing simulation | `team_addons.py`, `45`, `46`, `51`, `53`, `56`, `84`, `85`, `92`, `94` |
| Uncertainty and sensitivity (runs after the TEAM scripts) | `82` |
| Hospital-level panel of suppressed cells, for the Indiana ground-truth check | `70`, `75` |
| Rechecks of published series and the structured literature search | `79`-`81`, `87`, `88`, `96` |
| State data quality and specification checks (cost-report completeness, field quality, specification curve, rank concordance, Indiana decomposition) | `02`, `05`, `06`, `13`-`15` |
| Figures | `30`, `31`, `42`, `47` (main); `17`, `33`, `52`, `56` (Supplement) |
| Numbers quoted in the Supplement text, with sources | `90` |

Before submission the full pipeline was rerun from the raw downloads in a fresh workspace, and the
regenerated outputs matched the values the paper reports.

## Running it

Python 3.10. Install the pinned dependencies with `pip install -r code/requirements.txt`. `duckdb` reads
the large CMS CSV files without loading them into memory.

## Use of a large language model

The analysis scripts, tables, and figures were written with a large language model assistant (Claude, Opus 5,
Opus 5.5, and Fable 5.1, in Claude Code; Anthropic PBC) under the direction of the corresponding author, as
disclosed in the paper's Methods and Acknowledgment. The authors set the study questions, data sources,
definitions, and analytic approach. The model generated no data, and every number in the paper traces to a
script output built from the public files.

## Citing

Please cite the paper and the archived release of this code:

> Kurland DB, Sastry R, Niu T, Oyelese A. Code and crosswalks for: Where Did the Spine Surgery Go? Small-Cell
> Suppression, Medicare Advantage, and Pooled Episode Pricing in Medicare Inpatient Spine Surgery, 2013-2024.
> Version v1.0.0. Zenodo; 2026. doi:10.5281/zenodo.23134875

The concept DOI 10.5281/zenodo.23134874 always resolves to the latest version.

## Licence

Code: MIT (`LICENSE`). Crosswalks, manifests, and other files under `data/`: CC BY 4.0 (`LICENSE-data`).

## Contact

David B. Kurland, MD, PhD. Department of Neurosurgery, Brown University Health and The Warren Alpert Medical
School of Brown University, 593 Eddy Street, Providence, RI 02903. dkurland@brownhealth.org
