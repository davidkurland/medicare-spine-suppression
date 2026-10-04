"""Complete PSPS counts behind the rechecks of Lopez et al (lumbar discectomy, CPT 63030) and Ng et al (cervical disc
arthroplasty, CPT 22856), taken from the primary-surgeon series (setting_migration_cpt.csv, all settings).

The reported values and the provider-file and geography-file counts were entered by hand from each paper and file
(data/manual/published_paper_checks_entered.csv). This script fills the "PSPS complete" rows from the
primary-surgeon series, which excludes ASC facility lines, and leaves the other rows as entered.
Output: data/interim/published_paper_checks.csv
"""
import pandas as pd

from common import ROOT

I = ROOT / "data/interim"

if __name__ == "__main__":
    entered = pd.read_csv(ROOT / "data/manual/published_paper_checks_entered.csv")
    x = pd.read_csv(I / "setting_migration_cpt.csv", dtype={"cpt": str})
    tot = x.groupby(["cpt", "year"]).services.sum()
    from_series = {("lopez_discectomy", f"complete_{y}"): tot[("63030", y)] for y in (2013, 2017, 2022)}
    from_series.update({("ng_cda", f"complete_cda_{y}"): tot[("22856", y)] for y in (2016, 2022)})
    out = entered.copy()
    for (chk, meas), v in from_series.items():
        i = out.index[(out.check == chk) & (out.measure == meas)]
        assert len(i) == 1, (chk, meas)
        out.loc[i, "value"] = round(float(v))
    out.to_csv(I / "published_paper_checks.csv", index=False)
    print(pd.concat([entered.value.rename("as entered"), out.value.rename("from series"), out.measure], axis=1).to_string())
