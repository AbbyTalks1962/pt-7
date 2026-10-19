# Equity Screening Lab

A Streamlit educational stock-screening dashboard with three main tabs:

1. **Overview** — universe summary, populated KPI cards, sector distribution and top-ranked companies.
2. **Stage Analysis** — safety, growth/margin and profitability screens, elimination audit, and recommended portfolio construction.
3. **Buy / Sell Recommendations** — relative BUY/HOLD/SELL signals for the recommended portfolio.

## Data

The app downloads a prebuilt public-company dataset from the JANalytics US stock universe GitHub repository at runtime and caches it for 12 hours. No large SEC file or manually uploaded Excel file is required. An internet connection is needed for the initial download.

Source: https://github.com/JANalytics/us-stock-universe

## Styling updates

- KPI cards now show the screened population, surviving populations, final portfolio size, sectors and industries represented, and average final Z-score.
- Sector charts use a consistent distinct colour for each sector.
- Sector cells in company tables are tinted with the matching sector colour.
- The requested explanatory sentence about ranking direction and data loading has been removed from the page header.

## Important methodology notes

- The source does not provide eight annual gross-margin observations for every company. Stage 2 therefore uses available one-year revenue growth, two-year revenue CAGR and gross margin as proxies rather than presenting them as seven-year MS/MG.
- The Z-score uses public float as a market-equity proxy where available, so it is an estimate rather than the exact standard Altman Z-score.
- Missing values are not fabricated. Some metrics may be unavailable for some companies.
- BUY/HOLD/SELL signals are relative to the screened portfolio and selected thresholds; they are not price forecasts or personalized investment advice.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload `app.py`, `requirements.txt`, and `README.md` to the root of a GitHub repository, then deploy `app.py` through Streamlit Community Cloud.
