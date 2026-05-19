import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import plotly.graph_objects as go



st.set_page_config(page_title="Stock Decision Support App", layout="wide")



st.title("📈 Stock Decision Support App")

st.caption("Technical + Fundamental + Forecast — Educational only, not financial advice.")



st.warning(

    "This app is for educational research only. "

    "It does not guarantee predictions or provide buy/sell financial advice."

)



TICKERS = [

    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",

    "META", "TSLA", "JPM", "V", "UNH",

    "XOM", "WMT", "MA", "PG", "HD",

    "COST", "AVGO", "LLY", "NFLX", "AMD",

    "BAC", "KO", "PEP", "CRM", "ADBE",

    "CSCO", "ORCL", "IBM", "QCOM", "DIS",

    "NKE", "MCD", "ABT", "MRK", "PFE",

    "CVX", "BA", "CAT", "GE", "GS",

    "SPY", "QQQ", "VOO", "VTI", "SCHD"

]





@st.cache_data(ttl=3600)

def load_price_data(ticker, period):

    try:

        df = yf.download(

            ticker,

            period=period,

            auto_adjust=True,

            progress=False,

            threads=False

        )



        if df is None or df.empty:

            return pd.DataFrame()



        df = df.reset_index()



        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)



        if "Date" not in df.columns or "Close" not in df.columns:

            return pd.DataFrame()



        return df



    except Exception:

        return pd.DataFrame()





@st.cache_data(ttl=3600)

def load_fundamentals(ticker):

    try:

        info = yf.Ticker(ticker).info



        return {

            "Company": info.get("longName"),

            "Sector": info.get("sector"),

            "Industry": info.get("industry"),

            "Market Cap": info.get("marketCap"),

            "P/E Ratio": info.get("trailingPE"),

            "Forward P/E": info.get("forwardPE"),

            "Profit Margin": info.get("profitMargins"),

            "Revenue Growth": info.get("revenueGrowth"),

            "Debt to Equity": info.get("debtToEquity"),

            "ROE": info.get("returnOnEquity"),

            "Beta": info.get("beta"),

        }



    except Exception:

        return {}





def add_technical_indicators(df):

    df = df.copy()



    df["Return"] = df["Close"].pct_change()

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    df["MA200"] = df["Close"].rolling(200).mean()



    df["Return_1M"] = df["Close"].pct_change(21)

    df["Return_3M"] = df["Close"].pct_change(63)

    df["Volatility"] = df["Return"].rolling(20).std() * np.sqrt(252)



    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)



    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()



    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))



    return df.dropna()





def safe_num(x, default=None):

    try:

        if x is None or pd.isna(x):

            return default

        return float(x)

    except Exception:

        return default





def score_stock(latest, fundamentals):

    technical_score = 0

    fundamental_score = 0

    reasons = []



    close = safe_num(latest["Close"], 0)

    ma20 = safe_num(latest["MA20"], 0)

    ma50 = safe_num(latest["MA50"], 0)

    ma200 = safe_num(latest["MA200"], 0)

    rsi = safe_num(latest["RSI"], 0)

    return_1m = safe_num(latest["Return_1M"], 0)

    return_3m = safe_num(latest["Return_3M"], 0)

    volatility = safe_num(latest["Volatility"], 1)



    if close > ma20:

        technical_score += 1

        reasons.append("Price is above MA20.")

    else:

        reasons.append("Price is below MA20.")



    if close > ma50:

        technical_score += 1

        reasons.append("Price is above MA50.")

    else:

        reasons.append("Price is below MA50.")



    if close > ma200:

        technical_score += 1

        reasons.append("Price is above MA200.")

    else:

        reasons.append("Price is below MA200.")



    if ma20 > ma50:

        technical_score += 1

        reasons.append("Short-term trend is stronger than medium-term trend.")

    else:

        reasons.append("Short-term trend is weaker than medium-term trend.")



    if 45 <= rsi <= 65:

        technical_score += 2

        reasons.append("RSI is in a healthy range.")

    elif 30 <= rsi < 45:

        technical_score += 1

        reasons.append("RSI is neutral but slightly weak.")

    elif rsi > 70:

        technical_score -= 1

        reasons.append("RSI may be overbought.")

    else:

        reasons.append("RSI is weak or oversold.")



    if return_1m > 0:

        technical_score += 1

        reasons.append("1-month return is positive.")

    else:

        reasons.append("1-month return is negative.")



    if return_3m > 0:

        technical_score += 1

        reasons.append("3-month return is positive.")

    else:

        reasons.append("3-month return is negative.")



    if volatility < 0.25:

        risk = "Low"

        reasons.append("Volatility is low.")

    elif volatility < 0.45:

        risk = "Medium"

        technical_score -= 1

        reasons.append("Volatility is moderate.")

    else:

        risk = "High"

        technical_score -= 2

        reasons.append("Volatility is high.")



    pe = safe_num(fundamentals.get("P/E Ratio"))

    forward_pe = safe_num(fundamentals.get("Forward P/E"))

    profit_margin = safe_num(fundamentals.get("Profit Margin"))

    revenue_growth = safe_num(fundamentals.get("Revenue Growth"))

    debt_to_equity = safe_num(fundamentals.get("Debt to Equity"))

    roe = safe_num(fundamentals.get("ROE"))

    beta = safe_num(fundamentals.get("Beta"))



    if pe is not None:

        if 0 < pe < 35:

            fundamental_score += 1

            reasons.append("P/E ratio appears reasonable.")

        elif pe >= 35:

            reasons.append("P/E ratio is high.")



    if forward_pe is not None and 0 < forward_pe < 35:

        fundamental_score += 1

        reasons.append("Forward P/E appears reasonable.")



    if profit_margin is not None:

        if profit_margin > 0.10:

            fundamental_score += 1

            reasons.append("Profit margin is strong.")

        elif profit_margin < 0:

            fundamental_score -= 1

            reasons.append("Profit margin is negative.")



    if revenue_growth is not None:

        if revenue_growth > 0.05:

            fundamental_score += 1

            reasons.append("Revenue growth is positive.")

        elif revenue_growth < 0:

            fundamental_score -= 1

            reasons.append("Revenue growth is negative.")



    if debt_to_equity is not None:

        if debt_to_equity < 150:

            fundamental_score += 1

            reasons.append("Debt-to-equity appears manageable.")

        elif debt_to_equity > 250:

            fundamental_score -= 1

            reasons.append("Debt-to-equity is high.")



    if roe is not None and roe > 0.10:

        fundamental_score += 1

        reasons.append("Return on equity is strong.")



    if beta is not None and beta > 1.5:

        reasons.append("Beta is high, meaning the stock may move more aggressively than the market.")



    total_score = technical_score + fundamental_score



    if total_score >= 11 and risk != "High":

        label = "Strong Watch"

    elif total_score >= 8:

        label = "Watch"

    elif total_score >= 5:

        label = "Neutral"

    else:

        label = "High Caution"



    return technical_score, fundamental_score, total_score, risk, label, reasons





def forecast_future(df, days=30):

    recent_df = df.copy().tail(180)



    if len(recent_df) < 30:

        return pd.DataFrame(), 0, "Not enough data"



    recent_df["Day_Number"] = np.arange(len(recent_df))



    x = recent_df["Day_Number"].values

    y = recent_df["Close"].values



    slope, intercept = np.polyfit(x, y, 1)



    future_x = np.arange(len(recent_df), len(recent_df) + days)

    future_prices = slope * future_x + intercept



    last_date = recent_df["Date"].iloc[-1]



    future_dates = pd.date_range(

        start=last_date + pd.Timedelta(days=1),

        periods=days,

        freq="D"

    )



    forecast_df = pd.DataFrame({

        "Date": future_dates,

        "Forecasted Price": future_prices

    })



    current_price = recent_df["Close"].iloc[-1]

    forecast_price = forecast_df["Forecasted Price"].iloc[-1]

    expected_return = (forecast_price / current_price) - 1



    if expected_return > 0.08:

        forecast_label = "Positive Forecast"

    elif expected_return > 0.02:

        forecast_label = "Slightly Positive Forecast"

    elif expected_return > -0.02:

        forecast_label = "Flat / Neutral Forecast"

    else:

        forecast_label = "Negative Forecast"



    return forecast_df, expected_return, forecast_label





def make_chart(df, ticker):

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA200"], mode="lines", name="MA200"))



    fig.update_layout(

        title=f"{ticker} Price Trend",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig





def make_forecast_chart(df, forecast_df, ticker):

    fig = go.Figure()



    fig.add_trace(go.Scatter(

        x=df["Date"].tail(180),

        y=df["Close"].tail(180),

        mode="lines",

        name="Historical Price"

    ))



    fig.add_trace(go.Scatter(

        x=forecast_df["Date"],

        y=forecast_df["Forecasted Price"],

        mode="lines",

        name="Forecasted Price"

    ))



    fig.update_layout(

        title=f"{ticker} Future Price Forecast",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig





st.sidebar.header("Settings")



ticker = st.sidebar.selectbox("Select stock", TICKERS)



period = st.sidebar.selectbox(

    "Historical period",

    ["1y", "2y", "5y"],

    index=1

)



forecast_days = st.sidebar.selectbox(

    "Forecast horizon",

    [7, 14, 30, 60, 90],

    index=2

)



df = load_price_data(ticker, period)

fundamentals = load_fundamentals(ticker)



if df.empty:

    st.error("Could not load stock data. Try another ticker.")

else:

    df = add_technical_indicators(df)



    if df.empty:

        st.error("Not enough historical data for this stock.")

    else:

        latest = df.iloc[-1]



        technical_score, fundamental_score, total_score, risk, label, reasons = score_stock(

            latest,

            fundamentals

        )



        forecast_df, expected_return, forecast_label = forecast_future(df, forecast_days)



        st.subheader(f"{ticker} Decision Summary")



        c1, c2, c3, c4, c5 = st.columns(5)



        c1.metric("Price", f"${latest['Close']:.2f}")

        c2.metric("RSI", f"{latest['RSI']:.1f}")

        c3.metric("Risk", risk)

        c4.metric("Total Score", total_score)

        c5.metric("Label", label)



        st.plotly_chart(make_chart(df, ticker), use_container_width=True)



        st.markdown("### Future Forecast")



        f1, f2 = st.columns(2)



        f1.metric(

            f"Expected Return in {forecast_days} Days",

            f"{expected_return:.2%}"

        )



        f2.metric("Forecast Label", forecast_label)



        if not forecast_df.empty:

            st.plotly_chart(

                make_forecast_chart(df, forecast_df, ticker),

                use_container_width=True

            )



            st.dataframe(forecast_df, use_container_width=True)



        st.markdown("### Score Breakdown")



        score_df = pd.DataFrame({

            "Category": ["Technical Score", "Fundamental Score", "Total Score"],

            "Score": [technical_score, fundamental_score, total_score]

        })



        st.dataframe(score_df, use_container_width=True)



        st.markdown("### Fundamental Data")



        fund_df = pd.DataFrame({

            "Metric": list(fundamentals.keys()),

            "Value": list(fundamentals.values())

        })



        st.dataframe(fund_df, use_container_width=True)



        st.markdown("### Explanation")



        for r in reasons:

            st.write(f"- {r}")



        st.markdown("### Recent Price Data")

        st.dataframe(df.tail(50), use_container_width=True)