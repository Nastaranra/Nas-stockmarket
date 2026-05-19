import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import plotly.graph_objects as go

from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



st.set_page_config(page_title="Advanced Stock Decision App", layout="wide")



st.title("📈 Advanced Stock Decision Support App")

st.caption("Live US Stocks | Technical + Fundamental + ML Forecast + Scanner")

st.warning("Educational research only. This is NOT financial advice and does NOT guarantee profit.")



FALLBACK = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ"]





@st.cache_data(ttl=86400)

def get_all_tickers():

    tickers = []



    try:

        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

        nasdaq_df = pd.read_csv(nasdaq_url, sep="|")

        nasdaq_df = nasdaq_df[nasdaq_df["Test Issue"] == "N"]

        tickers += nasdaq_df["Symbol"].astype(str).tolist()

    except Exception:

        pass



    try:

        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

        other_df = pd.read_csv(other_url, sep="|")

        other_df = other_df[other_df["Test Issue"] == "N"]

        tickers += other_df["ACT Symbol"].astype(str).tolist()

    except Exception:

        pass



    clean = []



    for t in tickers:

        if isinstance(t, str):

            t = t.strip().replace(".", "-")



            if (

                len(t) <= 6

                and "$" not in t

                and " " not in t

                and t.upper() != "FILE"

                and t != ""

            ):

                clean.append(t)



    clean = sorted(list(set(clean)))



    if len(clean) == 0:

        clean = FALLBACK



    return clean, pd.DataFrame({"Ticker": clean})





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





def safe_num(x, default=None):

    try:

        if x is None or pd.isna(x):

            return default

        return float(x)

    except Exception:

        return default





def add_indicators(df):

    df = df.copy()



    df["Return"] = df["Close"].pct_change()

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    df["MA200"] = df["Close"].rolling(200).mean()



    df["Return_5D"] = df["Close"].pct_change(5)

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



    df["Target_30D"] = (df["Close"].shift(-30) > df["Close"]).astype(int)



    return df.dropna()





def ml_probability(df):

    features = [

        "RSI", "MA20", "MA50", "MA200",

        "Return_5D", "Return_1M", "Return_3M", "Volatility"

    ]



    if len(df) < 260:

        return None, None



    data = df[features + ["Target_30D"]].dropna()



    if len(data) < 200:

        return None, None



    X = data[features]

    y = data["Target_30D"]



    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.25,

        shuffle=False

    )



    model = RandomForestClassifier(

        n_estimators=150,

        max_depth=5,

        random_state=42

    )



    model.fit(X_train, y_train)



    accuracy = accuracy_score(y_test, model.predict(X_test))



    latest_X = df[features].iloc[[-1]]

    probability = model.predict_proba(latest_X)[0][1]



    return probability, accuracy





def forecast_trend(df, days):

    recent = df.tail(180).copy()



    if len(recent) < 30:

        return 0, "Not enough data", None



    x = np.arange(len(recent))

    y = recent["Close"].values



    slope, intercept = np.polyfit(x, y, 1)



    forecast_price = slope * (len(recent) + days) + intercept

    current_price = recent["Close"].iloc[-1]



    expected_return = (forecast_price / current_price) - 1

    forecast_date = recent["Date"].iloc[-1] + timedelta(days=days)



    if expected_return >= 0.05:

        label = "Positive Forecast"

    elif expected_return <= -0.05:

        label = "Negative Forecast"

    else:

        label = "Neutral Forecast"



    return expected_return, label, forecast_date





def score_stock(latest, fundamentals, expected_return, probability):

    technical_score = 0

    fundamental_score = 0

    forecast_score = 0

    reasons = []



    close = safe_num(latest["Close"], 0)

    ma20 = safe_num(latest["MA20"], 0)

    ma50 = safe_num(latest["MA50"], 0)

    ma200 = safe_num(latest["MA200"], 0)

    rsi = safe_num(latest["RSI"], 0)

    ret1 = safe_num(latest["Return_1M"], 0)

    ret3 = safe_num(latest["Return_3M"], 0)

    vol = safe_num(latest["Volatility"], 1)



    if close > ma20:

        technical_score += 1

        reasons.append("Price is above MA20.")



    if close > ma50:

        technical_score += 1

        reasons.append("Price is above MA50.")



    if close > ma200:

        technical_score += 1

        reasons.append("Price is above MA200.")



    if ma20 > ma50:

        technical_score += 1

        reasons.append("MA20 is above MA50.")



    if 45 <= rsi <= 65:

        technical_score += 2

        reasons.append("RSI is in a healthy range.")

    elif 30 <= rsi < 45:

        technical_score += 1

        reasons.append("RSI is neutral but weaker.")

    elif rsi > 70:

        technical_score -= 1

        reasons.append("RSI may be overbought.")



    if ret1 > 0:

        technical_score += 1

        reasons.append("1-month return is positive.")



    if ret3 > 0:

        technical_score += 1

        reasons.append("3-month return is positive.")



    if vol < 0.25:

        risk = "Low"

    elif vol < 0.45:

        risk = "Medium"

        technical_score -= 1

    else:

        risk = "High"

        technical_score -= 2



    pe = safe_num(fundamentals.get("P/E Ratio"))

    fpe = safe_num(fundamentals.get("Forward P/E"))

    margin = safe_num(fundamentals.get("Profit Margin"))

    growth = safe_num(fundamentals.get("Revenue Growth"))

    debt = safe_num(fundamentals.get("Debt to Equity"))

    roe = safe_num(fundamentals.get("ROE"))



    if pe is not None and 0 < pe < 35:

        fundamental_score += 1

        reasons.append("P/E ratio appears reasonable.")



    if fpe is not None and 0 < fpe < 35:

        fundamental_score += 1

        reasons.append("Forward P/E appears reasonable.")



    if margin is not None and margin > 0.10:

        fundamental_score += 1

        reasons.append("Profit margin is strong.")



    if growth is not None and growth > 0.05:

        fundamental_score += 1

        reasons.append("Revenue growth is positive.")



    if debt is not None and debt < 150:

        fundamental_score += 1

        reasons.append("Debt-to-equity appears manageable.")



    if roe is not None and roe > 0.10:

        fundamental_score += 1

        reasons.append("ROE is strong.")



    if expected_return >= 0.05:

        forecast_score += 2

        reasons.append("Trend forecast is positive.")

    elif expected_return <= -0.05:

        forecast_score -= 2

        reasons.append("Trend forecast is negative.")



    if probability is not None:

        if probability >= 0.60:

            forecast_score += 2

            reasons.append("ML model shows higher probability of positive return.")

        elif probability <= 0.40:

            forecast_score -= 2

            reasons.append("ML model shows lower probability of positive return.")



    total_score = technical_score + fundamental_score + forecast_score



    if total_score >= 11 and risk != "High":

        signal = "Buy Signal"

    elif total_score <= 3 or risk == "High":

        signal = "Sell Signal / High Caution"

    else:

        signal = "Hold / Neutral"



    return technical_score, fundamental_score, forecast_score, total_score, risk, signal, reasons





def analyze_stock(ticker, period, forecast_days):

    df = load_price_data(ticker, period)



    if df.empty:

        return None



    df = add_indicators(df)



    if df.empty:

        return None



    fundamentals = load_fundamentals(ticker)

    latest = df.iloc[-1]



    expected_return, forecast_label, forecast_date = forecast_trend(df, forecast_days)

    probability, ml_accuracy = ml_probability(df)



    technical_score, fundamental_score, forecast_score, total_score, risk, signal, reasons = score_stock(

        latest,

        fundamentals,

        expected_return,

        probability

    )



    return {

        "Ticker": ticker,

        "Company": fundamentals.get("Company"),

        "Sector": fundamentals.get("Sector"),

        "Price": round(safe_num(latest["Close"], 0), 2),

        "RSI": round(safe_num(latest["RSI"], 0), 1),

        "Volatility": round(safe_num(latest["Volatility"], 0), 3),

        "Technical Score": technical_score,

        "Fundamental Score": fundamental_score,

        "Forecast Score": forecast_score,

        "Total Score": total_score,

        "Risk": risk,

        "ML Positive Probability %": round(probability * 100, 2) if probability is not None else None,

        "ML Backtest Accuracy %": round(ml_accuracy * 100, 2) if ml_accuracy is not None else None,

        "Forecast Return %": round(expected_return * 100, 2),

        "Forecast Date": forecast_date.date() if forecast_date is not None else None,

        "Forecast Label": forecast_label,

        "Final Signal": signal,

        "P/E Ratio": fundamentals.get("P/E Ratio"),

        "Profit Margin": fundamentals.get("Profit Margin"),

        "Revenue Growth": fundamentals.get("Revenue Growth"),

        "Debt to Equity": fundamentals.get("Debt to Equity"),

        "Data": df,

        "Fundamentals": fundamentals,

        "Reasons": reasons

    }





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





tickers, ticker_df = get_all_tickers()



st.sidebar.header("Settings")



period = st.sidebar.selectbox(

    "Historical period",

    ["1y", "2y", "5y"],

    index=2

)



forecast_days = st.sidebar.selectbox(

    "Forecast horizon",

    [7, 14, 30, 60, 90],

    index=2

)



scan_count = st.sidebar.selectbox(

    "How many stocks to scan?",

    [25, 50, 100, 200, 300, 500, 750, 1000],

    index=2

)



scan_count = min(scan_count, len(tickers))



st.sidebar.write(f"Available tickers loaded: {len(tickers)}")



selected_ticker = st.sidebar.selectbox(

    "Select one stock",

    tickers

)



tab1, tab2, tab3 = st.tabs([

    "Single Stock",

    "All Stocks Scanner",

    "Ticker List"

])





with tab1:

    result = analyze_stock(selected_ticker, period, forecast_days)



    if result is None:

        st.error("Could not analyze this stock.")

    else:

        st.subheader(f"{selected_ticker} Summary")



        c1, c2, c3, c4, c5 = st.columns(5)



        c1.metric("Price", f"${result['Price']}")

        c2.metric("Risk", result["Risk"])

        c3.metric("Total Score", result["Total Score"])

        c4.metric("ML Probability", f"{result['ML Positive Probability %']}%")

        c5.metric("Signal", result["Final Signal"])



        st.info(

            f"Educational model signal until {result['Forecast Date']}: "

            f"{result['Final Signal']}"

        )



        st.plotly_chart(

            make_chart(result["Data"], selected_ticker),

            use_container_width=True

        )



        st.markdown("### Why this signal?")

        for reason in result["Reasons"]:

            st.write(f"- {reason}")



        st.markdown("### Details")



        detail_df = pd.DataFrame({

            "Metric": [

                "Technical Score",

                "Fundamental Score",

                "Forecast Score",

                "ML Positive Probability %",

                "ML Backtest Accuracy %",

                "Forecast Return %",

                "Forecast Date",

                "Forecast Label"

            ],

            "Value": [

                result["Technical Score"],

                result["Fundamental Score"],

                result["Forecast Score"],

                result["ML Positive Probability %"],

                result["ML Backtest Accuracy %"],

                result["Forecast Return %"],

                result["Forecast Date"],

                result["Forecast Label"]

            ]

        })



        st.dataframe(detail_df, use_container_width=True)



        st.markdown("### Fundamentals")



        st.dataframe(

            pd.DataFrame({

                "Metric": list(result["Fundamentals"].keys()),

                "Value": list(result["Fundamentals"].values())

            }),

            use_container_width=True

        )





with tab2:

    st.subheader("All Stocks Scanner")



    st.write(f"Scanning first {scan_count} stocks from live US stock list.")



    rows = []



    progress = st.progress(0)



    for i, ticker in enumerate(tickers[:scan_count]):

        result = analyze_stock(ticker, period, forecast_days)



        if result is not None:

            row = result.copy()

            row.pop("Data", None)

            row.pop("Fundamentals", None)

            row.pop("Reasons", None)

            rows.append(row)



        progress.progress((i + 1) / scan_count)



    if rows:

        scanner_df = pd.DataFrame(rows)

        scanner_df = scanner_df.sort_values("Total Score", ascending=False)



        st.dataframe(scanner_df, use_container_width=True)



        st.download_button(

            "Download Results",

            scanner_df.to_csv(index=False).encode("utf-8"),

            "advanced_stock_scanner_results.csv",

            "text/csv"

        )



        st.markdown("### Signal Summary")



        signal_summary = scanner_df["Final Signal"].value_counts().reset_index()

        signal_summary.columns = ["Signal", "Count"]



        st.dataframe(signal_summary, use_container_width=True)



    else:

        st.warning("No results found.")





with tab3:

    st.subheader("Live Ticker List")



    st.write(f"Total tickers loaded: {len(ticker_df)}")



    st.dataframe(ticker_df, use_container_width=True)

