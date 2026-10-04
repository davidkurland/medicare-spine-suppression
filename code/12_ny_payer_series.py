"""New York SPARCS spine and cranial inpatients by expected payer and year (all ages).

Imported by 31_figure2_states.py; reads the harmonized SPARCS partitions written by 04_sparcs_harmonize.py.
"""
import duckdb
import pandas as pd

from common import ROOT


def ny_payer_series() -> pd.DataFrame:
    return duckdb.connect().execute(f"""
      WITH c AS (SELECT year, pay1, CASE WHEN TRY_CAST(apr_drg AS INT) IN (20,21,22) THEN 'cranial'
                                        WHEN TRY_CAST(apr_drg AS INT) IN (23,303,304,310,321) THEN 'spine' END grp
                 FROM read_parquet('{ROOT}/data/interim/sparcs/*/*.parquet', union_by_name=true))
      SELECT grp, year, count(*) all_payer,
             sum(regexp_matches(pay1,'(?i)^medicare')::INT) medicare,
             sum(regexp_matches(pay1,'(?i)^medicaid')::INT) medicaid,
             sum(regexp_matches(pay1,'(?i)private|blue cross|insurance company|managed care')::INT) commercial,
             sum(regexp_matches(pay1,'(?i)misc|workers|other|federal|champus|corrections')::INT) other
      FROM c WHERE grp IS NOT NULL GROUP BY ALL ORDER BY grp, year""").df()


if __name__ == "__main__":
    print(ny_payer_series().to_string(index=False))
