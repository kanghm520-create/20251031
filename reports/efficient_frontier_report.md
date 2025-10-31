# Efficient Frontier Analysis Report

**Project:** Samsung, Apple, and NVIDIA Price Dataset  
**Date:** 2025-10-31
**Prepared by:** Data Analytics Team

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Analytical Context](#analytical-context)
3. [Dataset Overview](#dataset-overview)
4. [Descriptive Statistics](#descriptive-statistics)
5. [Efficient Frontier Methodology](#efficient-frontier-methodology)
6. [Implementation and Reproducibility](#implementation-and-reproducibility)
7. [Results and Interpretation](#results-and-interpretation)
8. [Operational Insights](#operational-insights)
9. [Recommendations and Next Steps](#recommendations-and-next-steps)
10. [Appendix A: Key Metrics Table](#appendix-a-key-metrics-table)
11. [Appendix B: Script Entry Points](#appendix-b-script-entry-points)

---

## Executive Summary
The analysis focused on understanding the historical performance and portfolio implications of three technology leaders—Samsung Electronics (005930.KS), Apple Inc. (AAPL), and NVIDIA Corporation (NVDA)—using the multi-ticker dataset stored in `temp.csv`. The work proceeded in three iterative phases: initial dataset documentation, exploratory cross-ticker analytics, and full Markowitz efficient frontier modelling with reproducible tooling. This report consolidates those efforts into a single, three-page narrative.

Key outcomes include:

- Comprehensive documentation of the dataset structure, coverage, and usage guidelines to accelerate onboarding for future analysts.
- Per-ticker descriptive statistics and cross-ticker correlation analysis that reveal distinctive liquidity profiles and price dynamics across U.S. and Korean markets.
- Construction of a long-only efficient frontier, with annualized return and volatility estimates derived from aligned trading calendars, and identification of minimum-variance and maximum-Sharpe portfolios.
- Delivery of a reproducible Python workflow (`scripts/generate_efficient_frontier.py`) capable of regenerating the efficient frontier visualization and associated metrics on demand.

Collectively, these deliverables equip stakeholders with both descriptive and prescriptive perspectives on the dataset, supporting tasks ranging from data quality checks to strategic asset allocation scenario analysis.

## Analytical Context
The initiative originated from a requirement to transform a raw CSV export into actionable insights for investment decision-making. Initial user feedback requested high-level documentation to contextualize the dataset, followed by specific interest in efficient frontier analysis and accompanying visuals to compare risk-return trade-offs among the three equities. Subsequent iterations refined the analysis to incorporate a risk-free rate assumption, deliver reproducibility scripts, and provide a more detailed interpretation of the output. This report reflects the cumulative learnings from that iterative dialogue.

Three guiding questions framed the work:

1. **What is the scope and structure of the dataset?** Precise knowledge of header semantics, ticker coverage, and temporal span ensures analysts can correctly ingest and transform the data.
2. **How do the individual assets behave historically?** Statistical summaries of price levels and traded volumes reveal each ticker's central tendencies and extremities, informing both risk assessment and model calibration.
3. **What portfolios strike attractive risk-return balances?** By modelling the efficient frontier, we quantify the benefits of diversification across the three equities and identify allocations that align with differing investor appetites.

## Dataset Overview
The `temp.csv` file encapsulates 516 trading days between 2023-10-16 and 2025-10-10. Each row corresponds to a synchronized trading day across all available tickers and includes the following metrics: Close, High, Low, Open, and Volume. The dataset employs a three-line header that maps each column to a metric-ticker pair, enabling multi-level selection in analytical tools such as pandas.

Ticker coverage is not uniform due to exchange-specific holidays. Samsung provides 482 complete rows, while Apple and NVIDIA each provide 499. Analysts should therefore align on the intersection of trading days when computing joint statistics, as was done for the efficient frontier. Monetary values remain in the securities' home currencies (Korean won for Samsung and U.S. dollars for Apple and NVIDIA), and volumes are reported as raw share counts without normalization.

Data quality considerations uncovered during documentation include:

- **Heterogeneous scale of volumes:** NVIDIA routinely trades hundreds of millions of shares per day, dwarfing Samsung's tens of millions. Comparative analytics should apply scaling or log transformations.
- **Currency differences:** Returns calculations sidestep currency conversion by focusing on percentage changes, but absolute price levels should not be compared without normalization.
- **Calendar alignment:** Non-overlapping holidays and trading suspensions require explicit filtering to avoid introducing `NaN` values in multi-asset computations.

## Descriptive Statistics
Descriptive analytics provided the first layer of insight. Average closing prices clustered around 66,472 KRW for Samsung, 209.52 USD for Apple, and 115.60 USD (post-split) for NVIDIA. Extreme values highlighted Samsung's strong rally into October 2025 and NVIDIA's meteoric rise, culminating in a 192.57 USD peak. Volume analysis further emphasized NVIDIA's outsized market participation, with daily peaks exceeding 1.1 billion shares.

Cross-ticker correlations of daily closing prices revealed nuanced relationships:

- **AAPL vs. NVDA:** A strong positive correlation (0.733) indicates synchronized movements among the two U.S. tech giants, likely driven by shared macroeconomic and sector catalysts.
- **005930.KS vs. AAPL:** A moderate negative correlation (-0.375) suggests that Samsung's Korea-listed shares provided diversification benefits relative to Apple during the sample window.
- **005930.KS vs. NVDA:** A mildly negative correlation (-0.189) reinforces Samsung's role as a partial hedge against NVIDIA's volatility.

These descriptive measures set expectations for the efficient frontier analysis: diversification gains should emerge from combining Samsung with the U.S.-listed equities, while pure U.S. portfolios may remain tightly coupled.

## Efficient Frontier Methodology
The efficient frontier analysis transformed daily closing prices into simple returns, filtered to the common trading calendar, and annualized the resulting series using a 252-trading-day convention. We imposed long-only weights summing to 100%, reflecting a realistic constraint for many institutional investors. Portfolio risk was computed via the covariance matrix of annualized returns, while expected returns were derived from the mean of the annualized series.

Two optimization targets were evaluated:

1. **Global Minimum-Variance Portfolio (GMV):** Minimizes volatility subject to the long-only and full-investment constraints.
2. **Maximum Sharpe Ratio Portfolio:** Maximizes excess return per unit of risk relative to a 2% annual risk-free rate, again under the same constraints.

To trace the frontier, the workflow sampled 100 target return levels between the GMV and the highest individual asset return, solving a quadratic program for each. The resulting risk-return pairs form the smooth frontier curve visualized in the final plot.

Assumptions and considerations:

- **Risk-Free Rate:** A constant 2% annual rate reflects a conservative proxy for contemporary U.S. Treasury yields, enabling Sharpe ratio comparisons.
- **No Shorting:** Short positions were excluded to maintain operational simplicity and align with regulatory limitations common in certain jurisdictions.
- **Reinvestment:** Returns assume full reinvestment with no transaction costs, taxes, or slippage.

## Implementation and Reproducibility
To operationalize the analysis, we introduced `scripts/generate_efficient_frontier.py`, a command-line tool built on pandas, NumPy, and Matplotlib. The script encapsulates data ingestion, cleaning, optimization, and visualization. Key capabilities include:

- **CLI Interface:** Users can specify the number of frontier points and the risk-free rate via command-line flags. Defaults target 100 points and a 0% risk-free rate, but the report's figures use `--points 100 --risk-free-rate 0.02`.
- **Robust Header Parsing:** The script reads the multi-line header into a pandas MultiIndex, simplifying column selection by metric and ticker.
- **Portfolio Optimization:** Leveraging NumPy linear algebra routines, the script solves for GMV and Sharpe-optimal weights under long-only constraints.
- **Visualization:** The script saves an efficient frontier SVG to `figures/efficient_frontier.svg`, plotting individual assets, the GMV point, and the maximum-Sharpe point for reference.

Version control history confirms successful execution of the script with the aforementioned parameters, ensuring reproducibility. Analysts can extend the workflow by importing the script's helper functions into notebooks or batch processes.

## Results and Interpretation
The frontier quantifies tangible diversification benefits:

- **Global Minimum-Variance Portfolio:** Allocates 43.29% to Samsung, 51.81% to Apple, and 4.90% to NVIDIA. The blend balances Apple's moderate volatility with Samsung's partial negative correlation, landing at a 30.12% expected annual return and 21.66% volatility.
- **Maximum Sharpe Portfolio:** Increases exposure to NVIDIA (56.93%) to capitalize on its outsized historical returns, while retaining diversification through Samsung (20.81%) and Apple (22.26%). This allocation targets a 65.14% expected return with 32.46% volatility, yielding a Sharpe ratio of 1.95 against the 2% risk-free benchmark.
- **Single-Asset Benchmarks:** NVIDIA dominates in raw return but carries the highest standalone volatility (49.38%). Samsung offers lower volatility (30.76%) but a more modest return (20.11%). Apple sits between the two on both dimensions (32.40% return, 27.73% volatility).

The efficient frontier plot visually reinforces these findings: portfolios tracing the frontier lie below the line connecting the individual assets, demonstrating superior risk-adjusted performance through diversification.

## Operational Insights
From a business operations perspective, the combined documentation and tooling deliver several benefits:

- **Faster Onboarding:** Analysts new to the dataset can reference the README and this report to understand structure, assumptions, and key figures without reverse engineering scripts.
- **Scenario Analysis:** The CLI enables quick stress testing of alternative risk-free rates or target point densities, supporting workshops and stakeholder presentations.
- **Data Governance:** Explicit documentation of date ranges, metrics, and currency considerations supports data cataloguing initiatives and reduces the risk of misinterpretation in downstream models.
- **Collaboration:** The report's modular sections can be repurposed for investor decks, internal memos, or audit documentation with minimal editing.

## Recommendations and Next Steps
To build on the current foundation, we propose the following enhancements:

1. **Expand Asset Universe:** Incorporate additional tickers—such as ETFs or regional benchmarks—to assess broader diversification opportunities.
2. **Currency Normalization:** Introduce FX-adjusted returns to evaluate portfolios from a single-currency perspective, crucial for global investors managing consolidated balance sheets.
3. **Transaction Cost Modelling:** Extend the optimization to account for bid-ask spreads and turnover penalties, aligning more closely with implementable strategies.
4. **Scenario Stress Testing:** Layer macroeconomic scenarios (e.g., rate hikes, supply chain disruptions) onto the historical data to gauge robustness of the frontier allocations.
5. **Interactive Dashboard:** Translate the static outputs into a web-based dashboard that allows stakeholders to interactively adjust assumptions and visualize resulting shifts.

Each recommendation aligns with stakeholder feedback requesting deeper insight, higher fidelity modelling, and richer presentation formats.

## Appendix A: Key Metrics Table
| Ticker | Annualized Expected Return | Annualized Volatility | Average Close | Max Close (Date) | Min Close (Date) | Average Volume | Max Volume (Date) |
| ------ | -------------------------: | --------------------: | ------------: | ---------------- | ---------------- | --------------: | ----------------- |
| 005930.KS | 20.11% | 30.76% | ₩66,471.53 | ₩94,400.00 (2025-10-10) | ₩48,968.97 (2024-11-14) | 19,481,204.69 | 57,691,266 (2024-01-11) |
| AAPL | 32.40% | 27.73% | $209.52 | $258.10 (2024-12-26) | $163.82 (2024-04-19) | 56,558,297.39 | 318,679,900 (2024-09-20) |
| NVDA | 94.40% | 49.38% | $115.60 | $192.57 (2025-10-09) | $40.30 (2023-10-26) | 325,122,504.81 | 1,142,269,000 (2024-03-08) |

*Note:* Monetary units reflect each ticker's listing currency. Return and volatility figures stem from the aligned-calendar efficient frontier workflow.

## Appendix B: Script Entry Points
The following command reproduces the efficient frontier chart and supporting metrics used throughout this report:

```bash
python scripts/generate_efficient_frontier.py --points 100 --risk-free-rate 0.02 --output figures/efficient_frontier.svg
```

The script writes the resulting figure to `figures/efficient_frontier.svg` and prints portfolio statistics to stdout. Analysts can adjust `--points` to control the granularity of the frontier and `--risk-free-rate` to test alternative Sharpe ratio baselines.
