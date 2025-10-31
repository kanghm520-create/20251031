# Stock Price Dataset Analysis

## Overview
This repository contains a single CSV file, `temp.csv`, with daily price and volume data for three equities:

- Samsung Electronics Co., Ltd. (005930.KS)
- Apple Inc. (AAPL)
- NVIDIA Corporation (NVDA)

Each record captures the close, high, low, and open prices along with trading volume for all available tickers on a given trading day.

## File Structure
The CSV uses a three-line header:

1. Metric group (Close, High, Low, Open, Volume)
2. Ticker symbol (repeated for each metric)
3. `Date` label for the first column

After the header, each row represents one trading day. The first column is the ISO-formatted trade date, followed by groups of columns for each ticker in the order Close → High → Low → Open → Volume.

## Coverage
- **Date range:** 2023-10-16 to 2025-10-10
- **Total trading rows:** 516
- **Per-ticker availability:**
  - 005930.KS — 482 rows with complete price/volume data
  - AAPL — 499 rows with complete price/volume data
  - NVDA — 499 rows with complete price/volume data

## Key Statistics
All prices are in each equity's native currency. Volumes are reported in raw shares traded.

| Ticker | Avg Close | Max Close (Date) | Min Close (Date) | Avg Volume | Max Volume (Date) |
| ------ | --------: | ---------------- | ---------------- | ---------: | ----------------- |
| 005930.KS | 66,471.53 | 94,400.00 (2025-10-10) | 48,968.97 (2024-11-14) | 19,481,204.69 | 57,691,266 (2024-01-11) |
| AAPL | 209.52 | 258.10 (2024-12-26) | 163.82 (2024-04-19) | 56,558,297.39 | 318,679,900 (2024-09-20) |
| NVDA | 115.60 | 192.57 (2025-10-09) | 40.30 (2023-10-26) | 325,122,504.81 | 1,142,269,000 (2024-03-08) |

## Cross-Ticker Relationships
Pearson correlations between the daily closing prices show:
- **AAPL vs NVDA:** Strong positive correlation (0.733)
- **005930.KS vs AAPL:** Moderate negative correlation (-0.375)
- **005930.KS vs NVDA:** Mild negative correlation (-0.189)

These relationships highlight that the two U.S.-listed technology stocks have tended to move together over the sample, while Samsung's Korea-listed shares moved more independently (and slightly inversely) during the same period.

## Usage Notes
- When loading the CSV with data-analysis tools, treat the first three rows as a multi-level header for easier column selection.
- Missing dates for specific tickers correspond to non-trading days on their home exchanges; align calendars accordingly for time-series modeling.
- Volumes span several orders of magnitude across tickers, so apply scaling or normalization before comparing activity levels.
