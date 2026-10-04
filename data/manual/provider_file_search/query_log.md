# Query log: spine studies drawing on CMS public summary files

Run date: 2026-10-02. Source: PubMed. Date filter 2014/01 to 2026/10, max 50 hits per query. found_by in the CSV records the first query (or seed-related pull) that returned each PMID; later duplicate hits are not listed.

| # | Query | Hits (returned / total) |
|---|---|---|
| Q1 | "Medicare"[MeSH] AND "Spinal Fusion"[MeSH] AND trends | 34 / 34 |
| Q2 | Medicare AND "public use file" AND spine | 1 / 1 |
| Q3 | "Physician and Other Supplier" AND spine | 3 / 3 |
| Q4 | "Provider Utilization and Payment" AND fusion | 50 / 140 (NOISY: PubMed ANDs the loose words provider, utilization, payment, fusion; includes many non-spine fusion/provider papers; top 50 kept for screening) |
| Q5 | Medicare AND reimbursement AND utilization AND "anterior cervical discectomy" | 20 / 20 |
| Q6 | Medicare AND "Part B" AND spine AND trends | 21 / 21 |
| Q7 | "Physician/Supplier Procedure Summary" AND spine | 14 / 14 |
| Q8 | Medicare AND utilization AND reimbursement AND lumbar | 50 / 57 (7 not retrieved) |
| Q9 | Medicare AND "disc arthroplasty" AND utilization | 10 / 10 |
| Q10 | Medicare AND kyphoplasty AND utilization AND trends | 8 / 8 |
| Q11 | Medicare AND sacroiliac AND utilization | 24 / 24 |
| Q12 | Medicare AND laminectomy AND reimbursement AND trends | 6 / 6 |
| Q13 | Medicare AND "spine surgeons" AND reimbursement | 19 / 19 |
| Q14 | Medicare AND CMS AND "spine surgery" AND volume AND hospital | 5 / 5 |
| Q15 | Medicare AND "Spinal Cord Stimulation" AND utilization | 16 / 16 |
| Q16 | "Medicare Inpatient Hospitals by Provider and Service" AND spinal | 21 / 21 (loose AND of words) |
| Q17 | "Inpatient Charge Data" AND spine | 0 |
| Q18 | "Medicare Physician & Other Practitioners" AND spine | 8 / 8 |
| Q19 | "Medicare"[MeSH] AND "Insurance, Health, Reimbursement"[MeSH] AND "Spinal Fusion"[MeSH] | 21 / 21 |
| Q20 | "Centers for Medicare and Medicaid Services, U.S."[MeSH] AND neurosurgeon AND reimbursement | 2 / 2 |
| Q21 | Medicare AND vertebroplasty AND utilization | 26 / 26 |
| Q22 | Medicare AND "ambulatory surgery center" AND spine | 12 / 12 |
| Q23 | Medicare AND "cervical disc arthroplasty" | 12 / 12 |
| Q24 | Medicare AND "surgeon volume" AND "spinal fusion" | 0 |
| Q25 | Medicare AND "hospital volume" AND "Spinal Fusion"[MeSH] | 2 / 2 (both already found) |
| Q26 | Medicare AND neurosurgery AND "Part B" AND payments | 15 / 15 |
| Q27 | Medicare AND "lumbar decompression" AND utilization | 26 / 26 |
| Q28 | Medicare AND "minimally invasive" AND "spinal fusion" AND trends | 2 / 2 (both already found) |
| Q29 | "Medicare Provider Utilization and Payment Data" AND "Spinal Fusion"[MeSH] | 17 / 17 |
| Q30 | "Medicare" AND "Orthopedic Procedures"[MeSH] AND spine AND "payment" | 50 / 55 (5 not retrieved) |
| Q31 | Medicare AND "spinal fusion" AND "geographic variation" AND "Diskectomy"[MeSH] | 1 / 1 (already found) |
| Q32 | Medicare AND "Spinal Fusion"[MeSH] AND "procedure volume" AND surgeons | 2 / 2 (both already found) |
| Q33 | Medicare AND "Spinal Fusion"[MeSH] AND inpatient AND "diagnosis-related groups"[MeSH] | 2 / 2 (both already found) |
| Q34 | Medicare AND "spinal cord stimulator" AND physicians AND payments | 2 / 2 (1 new) |
| Q0 | "Medicare Provider Utilization and Payment Data" (no spine term) | 50 / 1090. DROPPED as off-topic: all specialties, no spine restriction. Not in the CSV. |

Seed lookups (not counted as queries): DOI search 10.1016/j.spinee.2020.05.558 returned PMID 32534133 (Lopez lumbar fusion and discectomy). Khanna author search returned PMID 28338572 (Spine 2017;42:1648, confirmed by metadata). Menger author search returned PMID 26425400 (Surg Neurol Int 2015;6 Suppl 14:S391, confirmed by metadata).

## Seed related-article pulls
PubMed's related-articles list runs to several hundred PMIDs per seed; the first 20 (excluding the seed itself) were taken, in PubMed similarity order.
- 32562771 (Lopez ACDF): S1. 
- 32534133 (Lopez lumbar): S2.
- 42412075 (Ng cervical disc arthroplasty): S3.
- 38370335 (Miller ambulatory spine): S4.
- 28338572 (Khanna TLF): S5.
- 26425400 (Menger): S6.
- Seed-related PMIDs that fall before the 2014/01 start were NOT added, because related-article pulls cannot be date-filtered: 23972533, 22425784, 22343278, 21412380, 22430649, 22145188, 7721585, 18246016, 20489676, 23623637, 23532118. Related PMIDs already found by a Q query carry the Q label.

## Notes
- found_by shows first-found query only; use the pmid column for dedupe.
- Q4, Q16 and the S-pulls carry the most off-topic papers; screening should expect noise there.
- Many hits (e.g., Khanna 2017 on DRG 460 CMS data; Menger 2015 on named-surgeon CPT counts) need the abstract or full text to confirm which file was used. Screening therefore required the abstract to name or plainly describe the file.
