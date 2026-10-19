import io
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Equity Screening Lab", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1700px}
[data-testid="stMetric"]{background:rgba(120,140,160,.09);border:1px solid rgba(120,140,160,.28);padding:14px 16px;border-radius:12px;min-height:112px}
[data-testid="stMetricValue"]{font-weight:750}
[data-testid="stMetricLabel"]{font-weight:600}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stSidebar"]{border-right:1px solid #e8ebf0}
</style>
""", unsafe_allow_html=True)

DATA_URL = "https://raw.githubusercontent.com/JANalytics/us-stock-universe/main/us_listed_companies.csv"
SOURCE_PAGE = "https://github.com/JANalytics/us-stock-universe"

# Stable, distinct colours used for each sector across all charts and highlighted tables.
_SECTOR_PALETTE = (px.colors.qualitative.Dark24 + px.colors.qualitative.Set3 + px.colors.qualitative.Pastel1)

def sector_color_map(sectors):
    names = sorted(str(s) for s in sectors if pd.notna(s))
    return {name: _SECTOR_PALETTE[i % len(_SECTOR_PALETTE)] for i, name in enumerate(names)}

def style_sector_cells(frame, color_map):
    """Tint the Sector cells so the table uses the same sector colour key as charts."""
    if frame is None or frame.empty or "Sector" not in frame.columns:
        return frame
    def _sector_style(value):
        color = color_map.get(str(value), "#f3f4f6")
        return f"background-color: {color}; color: #172033; font-weight: 600"
    return frame.style.map(_sector_style, subset=["Sector"])

@st.cache_data(ttl=12*60*60, show_spinner=False)
def fetch_sec_dataset():
    """Fetch a compact, prebuilt SEC-derived market-wide dataset from GitHub; no SEC bulk files or Yahoo calls."""
    response = requests.get(DATA_URL, timeout=60, headers={"User-Agent": "EquityScreeningLab/1.0 educational project"})
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), low_memory=False)


def numcol(df, names, scale=1.0):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce") / scale
    return pd.Series(np.nan, index=df.index, dtype=float)


def txtcol(df, names, default="Unknown"):
    for name in names:
        if name in df.columns:
            s = df[name].fillna("").astype(str).str.strip()
            return s.replace("", default)
    return pd.Series(default, index=df.index, dtype=object)


def build_screen_dataset(raw, count=1000):
    d = raw.copy()
    d["Ticker"] = txtcol(d, ["Ticker", "Symbol"], "").str.upper()
    d["Company"] = txtcol(d, ["Issuer Name", "SEC Company Name", "Company", "Security Name"], "Unknown company")
    d["Sector"] = txtcol(d, ["Sector"], "Unclassified")
    d["Industry"] = txtcol(d, ["Industry (SIC Description)", "Industry", "SIC Description"], "Unclassified")
    # Most dollar amounts in the source's derived columns are already in millions.
    assets_m = numcol(d, ["Total Assets ($M)"])
    if assets_m.isna().all(): assets_m = numcol(d, ["Assets"]) / 1_000_000
    liabilities_m = numcol(d, ["Liabilities"]) / 1_000_000
    equity_m = numcol(d, ["Shareholders Equity ($M)"])
    if equity_m.isna().all(): equity_m = numcol(d, ["Equity"]) / 1_000_000
    current_assets_m = numcol(d, ["AssetsCurrent"]) / 1_000_000
    current_liabilities_m = numcol(d, ["LiabilitiesCurrent"]) / 1_000_000
    if current_assets_m.isna().all(): current_assets_m = numcol(d, ["Current Assets ($M)"])
    if current_liabilities_m.isna().all(): current_liabilities_m = numcol(d, ["Current Liabilities ($M)"])
    cash_m = numcol(d, ["Cash & Investments ($M)"])
    if cash_m.isna().all(): cash_m = numcol(d, ["Cash", "CashAndCashEquivalentsAtCarryingValue"]) / 1_000_000
    debt_m = numcol(d, ["Total Debt ($M)"])
    if debt_m.isna().all():
        debt_m = (numcol(d, ["LongTermDebt", "LongTermDebtNoncurrent"]).fillna(0) + numcol(d, ["CurrentDebt", "LongTermDebtCurrent"]).fillna(0)) / 1_000_000
    revenue_m = numcol(d, ["Revenue ($M)"])
    if revenue_m.isna().all(): revenue_m = numcol(d, ["Revenue"]) / 1_000_000
    net_income_m = numcol(d, ["Net Income ($M)"])
    if net_income_m.isna().all(): net_income_m = numcol(d, ["NetIncome", "NetIncomeLoss"]) / 1_000_000
    gross_profit_m = numcol(d, ["Gross Profit ($M)"])
    if gross_profit_m.isna().all(): gross_profit_m = numcol(d, ["GrossProfit"]) / 1_000_000
    ebit_m = numcol(d, ["Operating Income ($M)"])
    if ebit_m.isna().all(): ebit_m = numcol(d, ["OperatingIncome", "OperatingIncomeLoss"]) / 1_000_000
    ocf_m = numcol(d, ["Operating Cash Flow ($M)"])
    if ocf_m.isna().all(): ocf_m = numcol(d, ["OCF", "NetCashProvidedByUsedInOperatingActivities"]) / 1_000_000
    retained_m = numcol(d, ["Retained Earnings ($M)"])
    if retained_m.isna().all(): retained_m = numcol(d, ["RetainedEarnings", "RetainedEarningsAccumulatedDeficit"]) / 1_000_000
    public_float_m = numcol(d, ["Public Float ($M)"])
    if public_float_m.isna().all(): public_float_m = numcol(d, ["PublicFloat"]) / 1_000_000

    # Transparent one-period accounting-ratio calculations from real SEC-derived values.
    d["STA"] = (net_income_m - ocf_m) / assets_m.replace(0, np.nan)
    # SNOA = net operating assets / total assets; cash and debt are excluded from operating items.
    d["SNOA"] = ((assets_m - cash_m) - (liabilities_m - debt_m)) / assets_m.replace(0, np.nan)
    wc_m = current_assets_m - current_liabilities_m
    # Public float is used as an explicitly labelled market-equity proxy; it is not full market cap.
    z_parts = pd.concat([
        1.2 * wc_m / assets_m.replace(0, np.nan),
        1.4 * retained_m / assets_m.replace(0, np.nan),
        3.3 * ebit_m / assets_m.replace(0, np.nan),
        0.6 * public_float_m / liabilities_m.replace(0, np.nan),
        revenue_m / assets_m.replace(0, np.nan),
    ], axis=1)
    d["Z_Score"] = z_parts.sum(axis=1, min_count=5)
    d["GPA"] = gross_profit_m / assets_m.replace(0, np.nan)
    d["BTM"] = equity_m / public_float_m.replace(0, np.nan)
    d["CFOA"] = ocf_m / assets_m.replace(0, np.nan)
    d["RevenueGrowth"] = numcol(d, ["Revenue Growth 1Y %"]) / 100
    d["RevenueCAGR2Y"] = numcol(d, ["Revenue CAGR 2Y %"]) / 100
    d["GrossMargin"] = numcol(d, ["Gross Margin %"]) / 100
    d["MarketCap"] = public_float_m * 1_000_000
    d["Assets_M"] = assets_m
    d["Revenue_M"] = revenue_m
    d["DataSource"] = "SEC filings via JANalytics US stock universe"
    d["DataDate"] = txtcol(d, ["Revenue Period End", "Assets As Of"], "Latest filed period")
    # Stable diversified sample: rank completeness, then round-robin across sectors.
    d = d[d["Ticker"].str.len().between(1, 12) & (d["Ticker"] != "UNKNOWN")].copy()
    d = d.drop_duplicates("Ticker")
    if "Data Completeness %" in d:
        d["_completeness"] = pd.to_numeric(d["Data Completeness %"], errors="coerce").fillna(0)
    else:
        metric_cols = ["STA", "SNOA", "Z_Score", "GPA", "BTM", "CFOA", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin"]
        d["_completeness"] = d[metric_cols].notna().sum(axis=1)
    d = d.sort_values(["_completeness", "Ticker"], ascending=[False, True])
    if count < len(d):
        # Evenly draw from each sector, taking the most complete rows within each sector.
        sectors = sorted(d["Sector"].dropna().unique())
        buckets = {s: d[d["Sector"] == s].copy() for s in sectors}
        chosen = []
        while len(chosen) < count:
            added = False
            for s in sectors:
                if len(chosen) >= count: break
                if len(buckets[s]):
                    chosen.append(buckets[s].iloc[0])
                    buckets[s] = buckets[s].iloc[1:]
                    added = True
            if not added: break
        d = pd.DataFrame(chosen)
    d = d.drop(columns=["_completeness"], errors="ignore").reset_index(drop=True)
    return d


def rank_metric(s, high_good):
    x = pd.to_numeric(s, errors="coerce")
    n = len(x)
    # Missing metrics are ranked last, not treated as zero.
    ranks = x.rank(method="average", ascending=high_good is False, na_option="bottom")
    ranks = ranks.where(x.notna(), n + 1)
    return ranks


def adjusted_values(df, metrics, demean):
    out = df.copy()
    for m in metrics:
        if demean:
            means = out.groupby("Industry", dropna=False)[m].transform("mean")
            out[m + "_Adj"] = out[m] - means
        else:
            out[m + "_Adj"] = out[m]
    return out


def add_rank_block(df, metrics, high_good, weights, demean, prefix):
    out = adjusted_values(df, metrics, demean)
    rank_cols = []
    for metric in metrics:
        col = f"R_{metric}"
        out[col] = rank_metric(out[metric + "_Adj"], high_good[metric])
        rank_cols.append(col)
    wsum = sum(max(0, weights.get(m, 1.0)) for m in metrics)
    if wsum == 0: wsum = 1.0
    out[prefix + "_Score"] = sum(out[f"R_{m}"] * max(0, weights.get(m, 1.0)) for m in metrics) / wsum
    out[prefix + "_Rank"] = out[prefix + "_Score"].rank(method="min", ascending=True)
    return out


def rerank_previous(df, weights, demean_flags):
    x = add_rank_block(df, ["STA", "SNOA", "Z_Score"], {"STA": False, "SNOA": False, "Z_Score": True}, weights, demean_flags[1], "Stage1")
    # Stage 2 uses the metrics that are consistently present in the prebuilt source: 1-year revenue growth,
    # 2-year revenue CAGR and gross margin. This is clearly identified as a shorter-horizon proxy, not 7-year MS/MG.
    x = add_rank_block(x, ["RevenueGrowth", "RevenueCAGR2Y", "GrossMargin"], {"RevenueGrowth": True, "RevenueCAGR2Y": True, "GrossMargin": True}, weights, demean_flags[2], "Stage2Metrics")
    x["Cumulative12"] = x["Stage1_Score"] + x["Stage2Metrics_Score"]
    x["Cumulative12_Rank"] = x["Cumulative12"].rank(method="min", ascending=True)
    x = add_rank_block(x, ["GPA", "BTM", "CFOA"], {"GPA": True, "BTM": True, "CFOA": True}, weights, demean_flags[3], "Stage3")
    x["FinalScore"] = x["Stage1_Score"] + x["Stage2Metrics_Score"] + x["Stage3_Score"]
    x["Final_Rank"] = x["FinalScore"].rank(method="min", ascending=True)
    return x


def run_screen(df, cuts, weights, demean_flags):
    current = df.copy()
    stages, audit = {}, []
    for stage, pct in [(1, cuts[1]), (2, cuts[2]), (3, cuts[3])]:
        # Recalculate every stage's rankings for the currently surviving population before each elimination.
        ranked = rerank_previous(current, weights, demean_flags)
        if stage == 1:
            score_col = "Stage1_Score"
        elif stage == 2:
            score_col = "Cumulative12"
        else:
            score_col = "FinalScore"
        ranked = ranked.sort_values([score_col, "Ticker"], ascending=[True, True]).reset_index(drop=True)
        ranked[f"Stage{stage}_CurrentRank"] = np.arange(1, len(ranked) + 1)
        n_remove = int(np.ceil(len(ranked) * pct / 100)) if pct else 0
        # Highest (worst) cumulative scores are eliminated.
        removed = ranked.tail(n_remove)["Ticker"].tolist() if n_remove else []
        removed_set = set(removed)
        ranked["Status"] = np.where(ranked["Ticker"].isin(removed_set), f"Removed at Stage {stage}", "Survived")
        stages[stage] = ranked.copy()
        audit.extend(ranked[ranked["Ticker"].isin(removed_set)][["Ticker", "Company", "Sector", "Industry", "Status"]].to_dict("records"))
        generated_cols = ["Stage1_Score", "Stage1_Rank", "Stage2Metrics_Score", "Stage2Metrics_Rank", "Stage3_Score", "Stage3_Rank", "Cumulative12", "Cumulative12_Rank", "FinalScore", "Final_Rank", "Status", f"Stage{stage}_CurrentRank"]
        current = ranked[~ranked["Ticker"].isin(removed_set)].drop(columns=[c for c in ranked.columns if c.startswith("R_") or c.endswith("_Adj") or c in generated_cols], errors="ignore").copy()
    final = rerank_previous(current, weights, demean_flags).sort_values(["Final_Rank", "Ticker"]).reset_index(drop=True)
    return stages, final, pd.DataFrame(audit)


st.title("Equity Screening Lab")
st.caption("Accounting-based stock screening · staged company analysis · recommended portfolio · BUY / HOLD / SELL")

with st.sidebar:
    st.header("Screen settings")
    cuts = {1: st.slider("Stage 1 elimination (%)", 0, 50, 10, 5), 2: st.slider("Stage 2 elimination (%)", 0, 50, 10, 5), 3: st.slider("Stage 3 elimination (%)", 0, 50, 10, 5)}
    st.divider()
    st.subheader("Industry demeaning")
    demean_flags = {1: st.checkbox("Demean Stage 1 by industry", True), 2: st.checkbox("Demean Stage 2 by industry", False), 3: st.checkbox("Demean Stage 3 by industry", False)}
    st.divider()
    st.subheader("Ratio weights")
    weights = {}
    for m in ["STA", "SNOA", "Z_Score", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "GPA", "BTM", "CFOA"]:
        weights[m] = st.slider(f"{m} weight", 0.0, 5.0, 1.0, 0.25, key=f"w_{m}")
    st.divider()
    count = st.select_slider("Companies to screen", options=[100, 250, 500, 750, 1000, 1500, 2000], value=1000)
    if st.button("Refresh dataset", use_container_width=True):
        fetch_sec_dataset.clear()

try:
    with st.spinner("Loading the prebuilt SEC-derived company dataset…"):
        raw = fetch_sec_dataset()
        data = build_screen_dataset(raw, int(count))
except Exception as e:
    st.error("The prebuilt financial dataset could not be reached. This is a single GitHub-hosted CSV download, not a per-company Yahoo/SEC API loop.")
    st.code(str(e))
    st.markdown(f"Dataset source: [{SOURCE_PAGE}]({SOURCE_PAGE})")
    st.stop()

st.caption(f"Source: [JANalytics US stock universe]({SOURCE_PAGE}) · SEC-derived financial records and listing data · {len(raw):,} source rows")

st.subheader("Filter the company universe")
f1, f2, f3, f4 = st.columns([1, 1, 1.25, 1])
sector_values = sorted(data["Sector"].unique().tolist())
selected_sectors = f1.multiselect("Sector", sector_values, default=sector_values)
industry_values = sorted(data.loc[data["Sector"].isin(selected_sectors), "Industry"].unique().tolist()) if selected_sectors else []
selected_industries = f2.multiselect("Industry", industry_values, default=industry_values)
search = f3.text_input("Company / ticker", placeholder="Search a name or ticker")
min_completeness = f4.slider("Minimum data completeness (%)", 0, 100, 0, 10)
filtered = data[data["Sector"].isin(selected_sectors) & data["Industry"].isin(selected_industries)].copy()
if search.strip():
    filtered = filtered[filtered["Company"].str.contains(search, case=False, na=False) | filtered["Ticker"].str.contains(search, case=False, na=False)]
if "Data Completeness %" in filtered:
    filtered = filtered[pd.to_numeric(filtered["Data Completeness %"], errors="coerce").fillna(0) >= min_completeness]
SECTOR_COLORS = sector_color_map(data["Sector"].dropna().unique())
if filtered.empty:
    st.warning("No companies match these filters. Widen your filters to continue.")
    st.stop()

stages, final, audit = run_screen(filtered, cuts, weights, demean_flags)
s1, s2, s3 = stages[1], stages[2], stages[3]
removed1 = int((s1.Status == 'Removed at Stage 1').sum())
removed2 = int((s2.Status == 'Removed at Stage 2').sum())
removed3 = int((s3.Status == 'Removed at Stage 3').sum())

# Build the recommended portfolio once so the same portfolio drives the recommendation tab.
portfolio = final.copy()
if len(portfolio):
    industries = sorted(portfolio["Industry"].dropna().unique().tolist())
    default_allocation = pd.DataFrame({"Industry": industries, "Target_Weight_%": [100 / len(industries)] * len(industries)})
else:
    default_allocation = pd.DataFrame(columns=["Industry", "Target_Weight_%"])

# Three clear top-level destinations: overview, full stage analysis, and portfolio actions.
overview_tab, analysis_tab, recommendations_tab = st.tabs(["Overview", "Stage Analysis", "Buy / Sell Recommendations"])

with overview_tab:
    st.header("Screening overview")
    st.caption("A snapshot of the selected company universe and the results of your current screening settings.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Companies screened", f"{len(filtered):,}", help="Companies matching the current sector, industry and completeness filters.", border=True)
    k2.metric("After Stage 1", f"{max(0, len(s1) - removed1):,}", delta=f"−{removed1} removed" if removed1 else "No removals", help="Companies remaining after the safety screen.", border=True)
    k3.metric("After Stage 2", f"{max(0, len(s2) - removed2):,}", delta=f"−{removed2} removed" if removed2 else "No removals", help="Companies remaining after the growth and margin screen.", border=True)
    k4.metric("Recommended portfolio", f"{len(final):,}", delta=f"−{removed3} removed in Stage 3" if removed3 else "Final survivors", help="Companies that survived all three stages.", border=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Sectors represented", f"{filtered['Sector'].nunique():,}", border=True)
    m2.metric("Industries represented", f"{filtered['Industry'].nunique():,}", border=True)
    valid_z = pd.to_numeric(final.get("Z_Score", pd.Series(dtype=float)), errors="coerce").dropna()
    m3.metric("Average final Z-score", f"{valid_z.mean():.2f}" if len(valid_z) else "N/A", help="Mean of available estimated Z-scores among final survivors.", border=True)
    left, right = st.columns([1.25, 1])
    with left:
        sector_chart = filtered.groupby("Sector", as_index=False).size().sort_values("size", ascending=False)
        st.plotly_chart(px.bar(sector_chart, x="Sector", y="size", color="Sector", color_discrete_map=SECTOR_COLORS, title="Starting universe by sector", labels={"size": "Companies"}), use_container_width=True)
    with right:
        stage_summary = pd.DataFrame({"Stage": ["Starting universe", "After Stage 1", "After Stage 2", "Final portfolio"], "Companies": [len(filtered), max(0, len(s1)-removed1), max(0, len(s2)-removed2), len(final)]})
        st.plotly_chart(px.line(stage_summary, x="Stage", y="Companies", markers=True, title="Companies remaining after each stage"), use_container_width=True)
    st.subheader("Current top-ranked companies")
    top_cols = ["Final_Rank", "Ticker", "Company", "Sector", "Industry", "FinalScore", "STA", "SNOA", "Z_Score", "GPA", "BTM", "CFOA"]
    top_view = final[[c for c in top_cols if c in final]].head(20)
    st.dataframe(style_sector_cells(top_view, SECTOR_COLORS), use_container_width=True, hide_index=True)
    with st.expander("Browse the company database"):
        show_cols = ["Ticker", "Company", "Sector", "Industry", "STA", "SNOA", "Z_Score", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "GPA", "BTM", "CFOA", "DataDate"]
        st.dataframe(data[[c for c in show_cols if c in data]], use_container_width=True, hide_index=True)
        st.download_button("Download company database CSV", data.to_csv(index=False).encode(), "company_universe.csv", "text/csv", key="overview_download_database")
    st.info("**Data-method note:** the source does not provide eight annual gross-margin observations for every company. Stage 2 therefore uses available one-year revenue growth, two-year revenue CAGR and gross margin as transparent proxies, rather than presenting them as seven-year MS/MG. The Z-score uses public float as a market-equity proxy where available, so it is an estimate rather than the exact standard Altman Z-score. Missing values are not fabricated.")

with analysis_tab:
    st.header("Stage analysis & recommended portfolio")
    st.caption("Each stage ranks the surviving population again. The final portfolio is formed from companies that survive all three stages.")
    st1_tab, st2_tab, st3_tab, audit_tab = st.tabs(["Stage 1 · Safety", "Stage 2 · Growth & margins", "Stage 3 · Profitability", "Elimination audit"])
    with st1_tab:
        st.markdown("**STA and SNOA:** lower is better. **Altman Z-score:** higher is better. Companies with the worst Stage 1 composite ranks are removed.")
        cols = ["Stage1_CurrentRank", "Ticker", "Company", "Sector", "Industry", "STA", "SNOA", "Z_Score", "Stage1_Score", "Status"]
        st.dataframe(style_sector_cells(s1[[c for c in cols if c in s1]], SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download Stage 1 results", s1.to_csv(index=False).encode(), "stage1_safety.csv", "text/csv")
    with st2_tab:
        st.markdown("Stage 2 combines the available growth and margin proxies with the refreshed Stage 1 ranking. This dataset does not support a consistent seven-year MS/MG calculation for all companies.")
        cols = ["Ticker", "Company", "Sector", "Industry", "RevenueGrowth", "RevenueCAGR2Y", "GrossMargin", "Stage1_Score", "Stage2Metrics_Score", "Cumulative12", "Status"]
        st.dataframe(style_sector_cells(s2[[c for c in cols if c in s2]], SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download Stage 2 results", s2.to_csv(index=False).encode(), "stage2_growth_margins.csv", "text/csv")
    with st3_tab:
        st.markdown("**GPA, BTM and CFOA:** higher is treated as better for this screen. Confirm the preferred BTM direction against your course's value-investing methodology.")
        cols = ["Final_Rank", "Ticker", "Company", "Sector", "Industry", "GPA", "BTM", "CFOA", "Stage1_Score", "Stage2Metrics_Score", "Stage3_Score", "FinalScore", "Status"]
        st.dataframe(style_sector_cells(final[[c for c in cols if c in final]], SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download final survivors", final.to_csv(index=False).encode(), "stage3_survivors.csv", "text/csv")
    with audit_tab:
        if audit.empty:
            st.write("No companies were eliminated under the current settings.")
        else:
            st.dataframe(audit, use_container_width=True, hide_index=True)
            st.download_button("Download elimination audit", audit.to_csv(index=False).encode(), "elimination_audit.csv", "text/csv")

    st.divider()
    st.subheader("Recommended portfolio")
    st.caption("Default industry allocations are equal-weighted. Edit target weights below; each industry's allocation is divided equally among its surviving companies.")
    if len(portfolio):
        edited = st.data_editor(default_allocation, hide_index=True, use_container_width=True, key="industry_allocation", column_config={"Industry": st.column_config.TextColumn(disabled=True), "Target_Weight_%": st.column_config.NumberColumn(min_value=0, max_value=100, step=1, format="%.2f")})
        target_map = dict(zip(edited["Industry"], pd.to_numeric(edited["Target_Weight_%"], errors="coerce").fillna(0)))
        total = sum(target_map.values())
        if total <= 0:
            st.warning("Set at least one industry allocation above zero. Portfolio weights are currently zero.")
            portfolio["Industry_Target_%"] = 0.0
            portfolio["Portfolio_Weight"] = 0.0
        else:
            portfolio["Industry_Target_%"] = portfolio["Industry"].map(target_map).fillna(0) / total * 100
            counts_by_industry = portfolio.groupby("Industry")["Ticker"].transform("count")
            portfolio["Portfolio_Weight"] = (portfolio["Industry_Target_%"] / 100) / counts_by_industry
        portfolio["Portfolio_Weight_Pct"] = portfolio["Portfolio_Weight"] * 100
        st.metric("Portfolio companies", f"{len(portfolio):,}")
        st.metric("Allocated portfolio weight", f"{portfolio['Portfolio_Weight_Pct'].sum():.2f}%")
        portfolio_cols = ["Final_Rank", "Ticker", "Company", "Sector", "Industry", "Portfolio_Weight_Pct", "FinalScore", "STA", "SNOA", "Z_Score", "GPA", "BTM", "CFOA"]
        portfolio_view = portfolio[[c for c in portfolio_cols if c in portfolio]].sort_values("Final_Rank")
        st.dataframe(style_sector_cells(portfolio_view, SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download recommended portfolio", portfolio.to_csv(index=False).encode(), "recommended_portfolio.csv", "text/csv")
        allocation_chart = portfolio.groupby("Sector", as_index=False)["Portfolio_Weight_Pct"].sum().sort_values("Portfolio_Weight_Pct", ascending=False)
        st.plotly_chart(px.bar(allocation_chart, x="Sector", y="Portfolio_Weight_Pct", color="Sector", color_discrete_map=SECTOR_COLORS, title="Recommended portfolio allocation by sector", labels={"Portfolio_Weight_Pct": "Portfolio weight (%)"}), use_container_width=True)
    else:
        st.warning("No companies survived the current screening settings.")

with recommendations_tab:
    st.header("Buy / Sell recommendations")
    st.caption("Signals are based on the final rank of the recommended portfolio, not on live price targets. They are educational screening labels, not personalized investment advice.")
    if len(portfolio):
        r1, r2 = st.columns(2)
        buy_pct = r1.slider("BUY: top % of recommended portfolio", 5, 50, 20, 5, key="buy_threshold")
        sell_pct = r2.slider("SELL: bottom % of recommended portfolio", 5, 50, 20, 5, key="sell_threshold")
        portfolio = portfolio.sort_values(["Final_Rank", "Ticker"]).reset_index(drop=True)
        n = len(portfolio)
        buy_n = max(1, int(np.ceil(n * buy_pct / 100))) if n else 0
        sell_n = max(1, int(np.ceil(n * sell_pct / 100))) if n else 0
        portfolio["Signal"] = "HOLD"
        if n:
            portfolio.loc[:buy_n-1, "Signal"] = "BUY"
            if sell_n and n - sell_n >= 0:
                portfolio.loc[n-sell_n:, "Signal"] = "SELL"
            # In small universes where thresholds overlap, prioritize BUY for the best ranks.
            portfolio.loc[:buy_n-1, "Signal"] = "BUY"
        portfolio["Risk_Flag"] = np.where(portfolio["Z_Score"].lt(1.81), "Low Z-score / review", "")
        b1, b2, b3 = st.columns(3)
        b1.metric("BUY", int((portfolio.Signal == "BUY").sum()))
        b2.metric("HOLD", int((portfolio.Signal == "HOLD").sum()))
        b3.metric("SELL", int((portfolio.Signal == "SELL").sum()))
        signal_chart = portfolio["Signal"].value_counts().reindex(["BUY", "HOLD", "SELL"], fill_value=0).rename_axis("Signal").reset_index(name="Companies")
        st.plotly_chart(px.bar(signal_chart, x="Signal", y="Companies", title="Recommendation breakdown", category_orders={"Signal": ["BUY", "HOLD", "SELL"]}), use_container_width=True)
        st.subheader("Company-level recommendations")
        recommendation_view = portfolio[[c for c in ["Signal", "Final_Rank", "Ticker", "Company", "Sector", "Industry", "Portfolio_Weight_Pct", "FinalScore", "STA", "SNOA", "Z_Score", "GPA", "BTM", "CFOA", "Risk_Flag"] if c in portfolio]]
        st.dataframe(style_sector_cells(recommendation_view, SECTOR_COLORS), use_container_width=True, hide_index=True)
        st.download_button("Download BUY / HOLD / SELL recommendations", portfolio.to_csv(index=False).encode(), "buy_sell_recommendations.csv", "text/csv")
        st.caption("BUY/HOLD/SELL labels are relative to this screened universe and the chosen thresholds. A SELL label is not a claim that a company is fundamentally bad; it means it falls into the selected bottom-ranked portion of this portfolio.")
    else:
        st.warning("No companies survived the screening process, so there are no portfolio recommendations to display.")

st.caption("Source: JANalytics US stock universe. Financial fields may be blank when filings are unavailable or not comparable. This educational tool does not provide personalized investment advice.")
