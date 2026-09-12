# Selected maize line recommendation example

These are the first ten advanced lines per pool from the frozen selected model: C1 parent plus progeny ridge and C2 flat ridge, both centered against the complete planned site roster. Selection uses forecast scores and plot costs, not outcomes. This excerpt illustrates the decision output; it is not the complete field-testing plan.

The full allocation advances 757 C1 lines using 3,994 plots and 861 C2 lines using 4,698 plots. Each selected line receives one plot at every scheduled site. A breeder should review seed/tester availability, diversity and missing-data flags before using the plan.

Scores and intervals are in field-adjusted bu/acre, not absolute location yields or pure GCA. Intervals use 2007 residuals; they are prediction intervals for relative trial performance, not confidence intervals for genetic value or evidence of stable performance in every environment. All 2008 research is retrospective, with prior exposure to outcomes. Overall nominal 90% coverage was 90.81% in C1 and 87.63% in C2.

## C1

| Line | Pool rank | Predicted advantage | 90% prediction interval | Plots | Needs review |
|---|---:|---:|---|---:|---|
| C1.427.34 | 1 | +7.843 | -8.883 to +23.438 | 5 | Yes |
| C1.380.154 | 2 | +7.173 | -9.554 to +22.768 | 5 | Yes |
| C1.427.16 | 3 | +7.132 | -9.595 to +22.727 | 5 | Yes |
| C1.401.7 | 4 | +7.038 | -9.688 to +22.634 | 6 | No |
| C1.379.88 | 5 | +6.989 | -9.737 to +22.585 | 6 | No |
| C1.401.81 | 6 | +6.465 | -10.261 to +22.060 | 6 | No |
| C1.430.65 | 7 | +6.177 | -10.550 to +21.772 | 5 | Yes |
| C1.401.30 | 8 | +6.137 | -10.589 to +21.733 | 6 | No |
| C1.439.5 | 9 | +5.977 | -10.750 to +21.572 | 5 | Yes |
| C1.401.21 | 10 | +5.960 | -10.766 to +21.556 | 6 | No |

## C2

| Line | Pool rank | Predicted advantage | 90% prediction interval | Plots | Needs review |
|---|---:|---:|---|---:|---|
| C2.436.55 | 1 | +2.501 | -13.979 to +17.178 | 7 | Yes |
| C2.424.46 | 2 | +2.351 | -14.128 to +17.029 | 4 | No |
| C2.386.25 | 3 | +2.330 | -14.149 to +17.008 | 2 | No |
| C2.436.48 | 4 | +2.280 | -14.200 to +16.957 | 7 | Yes |
| C2.424.142 | 5 | +2.228 | -14.251 to +16.905 | 5 | No |
| C2.447.158 | 6 | +2.199 | -14.280 to +16.876 | 6 | Yes |
| C2.447.59 | 7 | +2.196 | -14.283 to +16.873 | 6 | Yes |
| C2.424.21 | 8 | +2.168 | -14.311 to +16.846 | 6 | No |
| C2.447.163 | 9 | +2.139 | -14.340 to +16.816 | 6 | Yes |
| C2.447.154 | 10 | +2.107 | -14.372 to +16.785 | 6 | Yes |

## Provenance and reproduction

- [Full-precision example CSV](selected_shortlist.csv)
- [Policy, source/artifact hashes and verification record](selected_shortlist_provenance.json)

The source run is `outputs/layout_selected_verified`. All six forecast source hashes match main commit `65511837ecbdcda2254f059caf3d7175e50429be`. The current policy differs from the run only in its descriptive status. Numeric model predictions and roster adjustment were independently reconstructed; intervals and the complete plot allocation were checked before exporting this excerpt. No model was refitted for the export.

Retrospective metrics were recomputed using saved historical outcomes after confirming equivalence of all candidate scores and advancement flags to the evaluated run. Those historical outcome files, full rankings and raw data remain local. Hashes document provenance; the small example alone cannot reproduce full-data training or verify the private artifacts.

To reproduce from the supplied complete dataset, run the selected forecast as described in the [README](../README.md), filter each `C*_rankings.csv` to `advance == True`, and take the first ten rows in saved rank order. Compare the numeric columns, allowing floating-point tolerance across environments. The CSV records full precision; the tables round to three decimals.
