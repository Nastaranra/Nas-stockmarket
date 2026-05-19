import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor



st.set_page_config(page_title="Stable Stock Decision App", layout="wide")



st.title("📈 Stable Stock Decision Support App")

st.caption("Technical + Fundamental + ML Forecast | Educational only, not financial advice.")





FALLBACK = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ"]





@st.cache_data(ttl=86400)

def get_all_tickers():

    tickers = []



    try:

        df1 = pd.read_csv(

            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",

            sep="|"

        )

        df1 = df1[df1["Test Issue"] == "N"]

        tickers += df1["Symbol"].astype(str).tolist()

    except Exception:

        pass



    try:

        df2 = pd.read_csv(

            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",

            sep="|"

        )

        df2 = df2[df2["Test Issue"] == "N"]

        tickers += df2["ACT Symbol"].astype(str).tolist()

    except Exception:

        pass



    clean = []

    for t in tickers:

        t = str(t).strip().replace(".", "-")

        if len(t) <= 6 and "$" not in t and " " not in t and t.upper() != "FILE":

            clean.append(t)



    clean = sorted(list(set(clean)))



    if not clean:

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





def safe_num(x, default=0):

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



    return df.dropna()





def ml_forecast_chart(df, days):

    recent = df.copy()



    if len(recent) < 260:

        return None, pd.DataFrame(), 0, "Not enough data"



    data = recent.copy()



    data["Lag_1"] = data["Close"].shift(1)

    data["Lag_5"] = data["Close"].shift(5)

    data["Lag_10"] = data["Close"].shift(10)

    data["Lag_20"] = data["Close"].shift(20)

    data["MA20_Model"] = data["Close"].rolling(20).mean()

    data["MA50_Model"] = data["Close"].rolling(50).mean()

    data["Return_5D_Model"] = data["Close"].pct_change(5)

    data["Return_20D_Model"] = data["Close"].pct_change(20)

    data["Volatility_Model"] = data["Close"].pct_change().rolling(20).std()

    data["Target"] = data["Close"].shift(-1)



    data = data.dropna()



    features = [

        "Lag_1",

        "Lag_5",

        "Lag_10",

        "Lag_20",

        "MA20_Model",

        "MA50_Model",

        "Return_5D_Model",

        "Return_20D_Model",

        "Volatility_Model"

    ]



    if len(data) < 150:

        return None, pd.DataFrame(), 0, "Not enough data"



    X = data[features]

    y = data["Target"]



    model = RandomForestRegressor(

        n_estimators=250,

        max_depth=8,

        min_samples_leaf=5,

        random_state=42,

        n_jobs=-1

    )



    model.fit(X, y)



    forecast_prices = []

    temp = recent[["Date", "Close"]].copy()



    for _ in range(days):

        temp_model = temp.copy()



        temp_model["Lag_1"] = temp_model["Close"].shift(1)

        temp_model["Lag_5"] = temp_model["Close"].shift(5)

        temp_model["Lag_10"] = temp_model["Close"].shift(10)

        temp_model["Lag_20"] = temp_model["Close"].shift(20)

        temp_model["MA20_Model"] = temp_model["Close"].rolling(20).mean()

        temp_model["MA50_Model"] = temp_model["Close"].rolling(50).mean()

        temp_model["Return_5D_Model"] = temp_model["Close"].pct_change(5)

        temp_model["Return_20D_Model"] = temp_model["Close"].pct_change(20)

        temp_model["Volatility_Model"] = temp_model["Close"].pct_change().rolling(20).std()



        latest_features = temp_model[features].dropna().iloc[[-1]]

        next_price = model.predict(latest_features)[0]



        next_date = temp["Date"].iloc[-1] + pd.Timedelta(days=1)



        new_row = pd.DataFrame({

            "Date": [next_date],

            "Close": [next_price]

        })



        temp = pd.concat([temp, new_row], ignore_index=True)

        forecast_prices.append(next_price)



    future_dates = pd.date_range(

        start=recent["Date"].iloc[-1] + pd.Timedelta(days=1),

        periods=days,

        freq="D"

    )



    forecast_df = pd.DataFrame({

        "Date": future_dates,

        "Forecasted Price": forecast_prices

    })



    current_price = recent["Close"].iloc[-1]

    final_price = forecast_df["Forecasted Price"].iloc[-1]

    expected_return = (final_price / current_price) - 1



    if expected_return >= 0.08:

        forecast_label = "Strong Positive Forecast"

    elif expected_return >= 0.03:

        forecast_label = "Positive Forecast"

    elif expected_return <= -0.08:

        forecast_label = "Strong Negative Forecast"

    elif expected_return <= -0.03:

        forecast_label = "Negative Forecast"

    else:

        forecast_label = "Neutral Forecast"



    fig = go.Figure()



    fig.add_trace(go.Scatter(

        x=recent["Date"].tail(260),

        y=recent["Close"].tail(260),

        mode="lines",

        name="Historical Price"

    ))



    fig.add_trace(go.Scatter(

        x=forecast_df["Date"],

        y=forecast_df["Forecasted Price"],

        mode="lines",

        name="ML Forecast"

    ))



    fig.update_layout(

        title="Historical Price + ML Forecast",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig, forecast_df, expected_return, forecast_label





def score_stock(latest, fundamentals, expected_return):

    technical_score = 0

    fundamental_score = 0

    reasons = []



    close = safe_num(latest["Close"])

    ma20 = safe_num(latest["MA20"])

    ma50 = safe_num(latest["MA50"])

    ma200 = safe_num(latest["MA200"])

    rsi = safe_num(latest["RSI"])

    ret1 = safe_num(latest["Return_1M"])

    ret3 = safe_num(latest["Return_3M"])

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



    pe = safe_num(fundamentals.get("P/E Ratio"), None)

    fpe = safe_num(fundamentals.get("Forward P/E"), None)

    margin = safe_num(fundamentals.get("Profit Margin"), None)

    growth = safe_num(fundamentals.get("Revenue Growth"), None)

    debt = safe_num(fundamentals.get("Debt to Equity"), None)

    roe = safe_num(fundamentals.get("ROE"), None)



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



    forecast_score = 0

    if expected_return >= 0.05:

        forecast_score = 2

        reasons.append("ML forecast trend is positive.")

    elif expected_return <= -0.05:

        forecast_score = -2

        reasons.append("ML forecast trend is negative.")



    total_score = technical_score + fundamental_score + forecast_score



    if total_score >= 10 and risk != "High":

        signal = "Buy Signal"

    elif total_score <= 3 or risk == "High":

        signal = "Sell Signal / High Caution"

    else:

        signal = "Hold / Neutral"



    return technical_score, fundamental_score, forecast_score, total_score, risk, signal, reasons





def scanner_score(ticker, period, forecast_days):

    df = load_price_data(ticker, period)



    if df.empty:

        return None



    df = add_indicators(df)



    if df.empty:

        return None



    latest = df.iloc[-1]



    close = safe_num(latest["Close"])

    ma20 = safe_num(latest["MA20"])

    ma50 = safe_num(latest["MA50"])

    ma200 = safe_num(latest["MA200"])

    rsi = safe_num(latest["RSI"])

    vol = safe_num(latest["Volatility"], 1)

    ret1 = safe_num(latest["Return_1M"])

    ret3 = safe_num(latest["Return_3M"])



    score = 0



    if close > ma20:

        score += 1

    if close > ma50:

        score += 1

    if close > ma200:

        score += 1

    if ma20 > ma50:

        score += 1

    if 45 <= rsi <= 65:

        score += 2

    elif 30 <= rsi < 45:

        score += 1

    elif rsi > 70:

        score -= 1

    if ret1 > 0:

        score += 1

    if ret3 > 0:

        score += 1



    if vol > 0.45:

        risk = "High"

        score -= 2

    elif vol > 0.25:

        risk = "Medium"

        score -= 1

    else:

        risk = "Low"



    if score >= 7 and risk != "High":

        signal = "Buy Signal"

    elif score <= 2 or risk == "High":

        signal = "Sell Signal / High Caution"

    else:

        signal = "Hold / Neutral"



    return {

        "Ticker": ticker,

        "Price": round(close, 2),

        "RSI": round(rsi, 1),

        "Volatility": round(vol, 3),

        "Score": score,

        "Risk": risk,

        "Signal": signal

    }





def make_price_chart(df, ticker):

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

    [30, 60, 90, 120, 180],

    index=2

)



scan_count = st.sidebar.selectbox(

    "How many stocks to scan?",

    [25, 50, 100, 200, 300, 500, 750, 1000],

    index=2

)



scan_count = min(scan_count, len(tickers))



st.sidebar.write(f"Available tickers loaded: {len(tickers)}")



selected_ticker = st.sidebar.selectbox("Select one stock", tickers)



tab1, tab2, tab3 = st.tabs(["Single Stock", "Stable Scanner", "Ticker List"])





with tab1:

    df = load_price_data(selected_ticker, period)



    if df.empty:

        st.error("Could not load data.")

    else:

        df = add_indicators(df)



        if df.empty:

            st.error("Not enough data.")

        else:

            fundamentals = load_fundamentals(selected_ticker)

            latest = df.iloc[-1]



            forecast_fig, forecast_df, expected_return, forecast_label = ml_forecast_chart(

                df,

                forecast_days

            )



            if forecast_fig is None:

                st.error("Not enough data for ML forecast.")

            else:

                technical_score, fundamental_score, forecast_score, total_score, risk, signal, reasons = score_stock(

                    latest,

                    fundamentals,

                    expected_return

                )



                st.subheader(f"{selected_ticker} Summary")



                c1, c2, c3, c4, c5 = st.columns(5)



                c1.metric("Price", f"${latest['Close']:.2f}")

                c2.metric("Risk", risk)

                c3.metric("Total Score", total_score)

                c4.metric("ML Forecast Return", f"{expected_return:.2%}")

                c5.metric("Signal", signal)



                st.info(

                    f"Educational model signal for next {forecast_days} days: {signal}. "

                    f"Forecast label: {forecast_label}."

                )



                st.plotly_chart(

                    make_price_chart(df, selected_ticker),

                    use_container_width=True

                )



                st.markdown("### ML Forecast Chart")

                st.plotly_chart(forecast_fig, use_container_width=True)



                st.markdown("### Forecasted Prices")

                st.dataframe(forecast_df, use_container_width=True)



                st.markdown("### Why this signal?")

                for r in reasons:

                    st.write(f"- {r}")



                st.markdown("### Fundamentals")

                st.dataframe(

                    pd.DataFrame({

                        "Metric": list(fundamentals.keys()),

                        "Value": list(fundamentals.values())

                    }),

                    use_container_width=True

                )





with tab2:

    st.subheader("Stable Scanner")

    st.write(f"Scanning first {scan_count} stocks. This version is lighter to reduce crashes.")



    rows = []

    progress = st.progress(0)



    for i, ticker in enumerate(tickers[:scan_count]):

        result = scanner_score(ticker, period, forecast_days)



        if result is not None:

            rows.append(result)



        progress.progress((i + 1) / scan_count)



    if rows:

        scanner_df = pd.DataFrame(rows).sort_values("Score", ascending=False)

        st.dataframe(scanner_df, use_container_width=True)



        st.download_button(

            "Download Scanner Results",

            scanner_df.to_csv(index=False).encode("utf-8"),

            "stable_stock_scanner_results.csv",

            "text/csv"

        )

    else:

        st.warning("No scanner results found.")





with tab3:

    st.subheader("Ticker List")

    st.write(f"Total tickers loaded: {len(ticker_df)}")

    st.dataframe(ticker_df, use_container_width=True)

