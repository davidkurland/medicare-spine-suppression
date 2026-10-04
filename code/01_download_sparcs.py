"""Download SPARCS de-identified inpatient PUFs (2009-2024) and convert each to parquet.

All columns are kept as strings in the raw parquet; harmonisation happens in 04_sparcs_harmonize.py.
The CSV is deleted after a successful conversion (re-downloadable from health.data.ny.gov).
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "sparcs"
RAW.mkdir(parents=True, exist_ok=True)

DATASETS = {
    2009: "q6hk-esrj", 2010: "mtfm-rxf4", 2011: "pyhr-5eas", 2012: "u4ud-w55t",
    2013: "npsr-cm47", 2014: "rmwa-zns4", 2015: "82xm-y6g8", 2016: "gnzp-ekau",
    2017: "22g3-z7e7", 2018: "yjgt-tq93", 2019: "4ny4-j5zv", 2020: "nxi5-zj9x",
    2021: "tg3i-cinn", 2022: "5dtw-tffi", 2023: "46xm-urtu", 2024: "sf4k-39ay",
}


def fetch(year: int) -> str:
    pq = RAW / f"sparcs_{year}.parquet"
    if pq.exists():
        return f"{year}: exists"
    csv = RAW / f"sparcs_{year}.csv"
    url = f"https://health.data.ny.gov/api/views/{DATASETS[year]}/rows.csv?accessType=DOWNLOAD"
    if not csv.exists():
        subprocess.run(["curl", "-s", "--retry", "5", "-m", "7200", url, "-o", str(csv)], check=True)
    con = duckdb.connect()
    # some releases contain malformed rows (unbalanced quotes); keep them as far as possible and log how many were lost
    con.execute(
        f"COPY (SELECT * FROM read_csv('{csv}', all_varchar=true, header=true, delim=',', quote='\"', "
        f"escape='\"', strict_mode=false, null_padding=true, ignore_errors=true, store_rejects=true)) "
        f"TO '{pq}' (FORMAT parquet, COMPRESSION zstd)"
    )
    rejects = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
    if rejects:
        print(f"{year}: {rejects} malformed rows rejected", flush=True)
    n = con.execute(f"SELECT count(*) FROM '{pq}'").fetchone()[0]
    csv.unlink()
    return f"{year}: {n:,} rows"


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or list(DATASETS)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for msg in ex.map(fetch, years):
            print(msg, flush=True)
