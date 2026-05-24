import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import requests

import plotly.graph_objects as go

from datetime import datetime, timedelta

from io import StringIO



st.set_page_config(page_title="AI Trading Signal App", layout="wide")



st.title("📈 AI Trading Signal App")

st.caption("Live Finnhub Price + Technical + News + Trading Plan")

st.warning("Educational only. Not financial advice. No signal is guaranteed.")



TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "SPY", "QQQ"]



if st.button("Clear cache / Refresh data"):

    st.cache_data.clear()

    st.rerun()





def safe_num(x, default=0):

    try:

        if x is None or pd.isna(x):

            return default

        return float(x)

    except Exception:

        return default





@st.cache_data(ttl=30)

def get_live_quote(ticker):

    api_key = st.secrets.get("FINNHUB_API_KEY", "")



    if not api_key:

        return None, "No Finnhub API Key"



    try:

        url = "https://finnhub.io/api/v1/quote"

        params = {"symbol": ticker, "token": api_key}

        r = requests.get(url, params=params, timeout=8)



        if r.status_code != 200:

            return None, "Finnhub quote error"



        data = r.json()

        price = data.get("c")



        if price is None or float(price) <= 0:

            return None, "No live quote"



        return float(price), "Live/Finnhub"



    except Exception:

        return None, "Finnhub quote failed"





@st.cache_data(ttl=300)

def load_price_data(ticker, period="1y"):

    ticker = str(ticker).upper().strip()



    try:

        df = yf.download(

            ticker,

            period=period,

            interval="1d",

            auto_adjust=True,

            progress=False,

            threads=False,

            timeout=10

        )



        if df is not None and not df.empty:

            df = df.reset_index()



            if isinstance(df.columns, pd.MultiIndex):

                df.columns = df.columns.get_level_values(0)



            if "Date" in df.columns and "Close" in df.columns:

                df = df.dropna(subset=["Close"])

                df["Data_Source"] = "Historical/YFinance"

                return df



    except Exception:

        pass



    try:

        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"

        r = requests.get(url, timeout=8)



        if r.status_code == 200 and r.text.strip():

            df = pd.read_csv(StringIO(r.text))



            if not df.empty and "Close" in df.columns:

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

                df = df.dropna(subset=["Date", "Close"])

                df = df.sort_values("Date")



                if period == "5y":

                    df = df.tail(1260)

                else:

                    df = df.tail(252)



                df["Data_Source"] = "Historical/Stooq"

                return df



    except Exception:

        pass



    return pd.DataFrame()





@st.cache_data(ttl=1800)

def load_news_sentiment(ticker):

    api_key = st.secrets.get("FINNHUB_API_KEY", "")



    if not api_key:

        return pd.DataFrame(), 0, "No API Key"



    try:

        today = datetime.today().date()

        start = today - timedelta(days=7)



        url = "https://finnhub.io/api/v1/company-news"

        params = {

            "symbol": ticker,

            "from": str(start),

            "to": str(today),

            "token": api_key

        }



        r = requests.get(url, params=params, timeout=8)



        if r.status_code != 200:

            return pd.DataFrame(), 0, "News Error"



        news = r.json()



        if not news:

            return pd.DataFrame(), 0, "No Recent News"



        positive_words = ["beat", "growth", "strong", "upgrade", "surge", "profit", "bullish", "gain", "rally", "ai"]

        negative_words = ["miss", "drop", "fall", "lawsuit", "weak", "downgrade", "loss", "bearish", "risk", "warning"]



        rows = []

        total_score = 0



        for item in news[:20]:

            headline = str(item.get("headline", ""))

            summary = str(item.get("summary", ""))

            text = (headline + " " + summary).lower()



            pos = sum(1 for w in positive_words if w in text)

            neg = sum(1 for w in negative_words if w in text)

            score = pos - neg

            total_score += score



            sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"



            rows.append({

                "Date": datetime.fromtimestamp(item.get("datetime")).strftime("%Y-%m-%d") if item.get("datetime") else None,

                "Headline": headline,

                "Sentiment": sentiment,

                "Source": item.get("source"),

                "URL": item.get("url")

            })



        avg_score = total_score / max(len(rows), 1)



        if avg_score > 0.25:

            label = "Positive News"

        elif avg_score < -0.25:

            label = "Negative News"

        else:

            label = "Neutral News"



        return pd.DataFrame(rows), avg_score, label



    except Exception:

        return pd.DataFrame(), 0, "News Error"





def add_indicators(df, live_price=None):

    df = df.copy()



    if df.empty or "Close" not in df.columns:

        return pd.DataFrame()



    if live_price is not None and len(df) > 0:

        df.loc[df.index[-1], "Close"] = live_price

        if "High" in df.columns:

            df.loc[df.index[-1], "High"] = max(df.loc[df.index[-1], "High"], live_price)

        if "Low" in df.columns:

            df.loc[df.index[-1], "Low"] = min(df.loc[df.index[-1], "Low"], live_price)



    df["Return"] = df["Close"].pct_change()

    df["MA9"] = df["Close"].rolling(9, min_periods=5).mean()

    df["MA20"] = df["Close"].rolling(20, min_periods=10).mean()

    df["MA50"] = df["Close"].rolling(50, min_periods=20).mean()



    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)



    avg_gain = gain.rolling(14, min_periods=7).mean()

    avg_loss = loss.rolling(14, min_periods=7).mean()



    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))



    exp1 = df["Close"].ewm(span=12, adjust=False).mean()

    exp2 = df["Close"].ewm(span=26, adjust=False).mean()



    df["MACD"] = exp1 - exp2

    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()



    if "High" in df.columns and "Low" in df.columns:

        tr1 = df["High"] - df["Low"]

        tr2 = (df["High"] - df["Close"].shift()).abs()

        tr3 = (df["Low"] - df["Close"].shift()).abs()

        df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["ATR"] = df["TR"].rolling(14, min_periods=7).mean()

    else:

        df["ATR"] = df["Close"].rolling(14, min_periods=7).std()



    df["Support"] = df["Close"].rolling(30, min_periods=10).min()

    df["Resistance"] = df["Close"].rolling(30, min_periods=10).max()



    df = df.replace([np.inf, -np.inf], np.nan)



    needed = ["Close", "MA9", "MA20", "MA50", "RSI", "MACD", "MACD_Signal", "ATR", "Support", "Resistance"]

    return df.dropna(subset=needed)





def make_signal(latest, news_score=0):

    score = 0

    reasons = []



    if latest["Close"] > latest["MA9"]:

        score += 1

        reasons.append("Price is above MA9.")



    if latest["Close"] > latest["MA20"]:

        score += 1

        reasons.append("Price is above MA20.")



    if latest["MA9"] > latest["MA20"]:

        score += 1

        reasons.append("Short-term trend is positive.")



    if latest["MA20"] > latest["MA50"]:

        score += 1

        reasons.append("Medium-term trend is positive.")



    if latest["MACD"] > latest["MACD_Signal"]:

        score += 1

        reasons.append("MACD is bullish.")



    if 45 <= latest["RSI"] <= 68:

        score += 1

        reasons.append("RSI is healthy.")

    elif latest["RSI"] > 72:

        score -= 1

        reasons.append("RSI is overbought.")

    elif latest["RSI"] < 32:

        score -= 1

        reasons.append("RSI is weak or oversold.")



    if news_score > 0.25:

        score += 1

        reasons.append("Recent news is positive.")

    elif news_score < -0.25:

        score -= 1

        reasons.append("Recent news is negative.")



    if score >= 6:

        signal = "Strong Buy"

    elif score >= 4:

        signal = "Buy Signal"

    elif score >= 3:

        signal = "Hold / Wait"

    else:

        signal = "Sell / High Caution"



    confidence = int(min(85, max(35, 45 + score * 7)))



    return signal, score, confidence, reasons





def estimate_future_price(df, days):

    recent = df.tail(252).copy()



    if len(recent) < 30:

        return pd.DataFrame(), 0



    returns = recent["Close"].pct_change().dropna()



    if returns.empty:

        return pd.DataFrame(), 0



    avg_return = returns.mean()

    volatility = returns.std()

    last_price = recent["Close"].iloc[-1]



    base_price = last_price

    bull_price = last_price

    bear_price = last_price



    for _ in range(days):

        base_price *= (1 + avg_return)

        bull_price *= (1 + avg_return + volatility * 0.25)

        bear_price *= (1 + avg_return - volatility * 0.25)



    expected_return = (base_price / last_price) - 1



    return pd.DataFrame({

        "Forecast Horizon": [f"{days} days"],

        "Current Price": [round(last_price, 2)],

        "Base Estimated Price": [round(base_price, 2)],

        "Bull Case Price": [round(bull_price, 2)],

        "Bear Case Price": [round(bear_price, 2)],

        "Estimated Return": [f"{expected_return:.2%}"]

    }), expected_return





def trade_plan(latest, signal, confidence, expected_return, horizon_days):

    close = safe_num(latest["Close"])

    atr = safe_num(latest["ATR"], close * 0.02)

    support = safe_num(latest["Support"], close - atr)

    resistance = safe_num(latest["Resistance"], close + atr)



    if signal in ["Strong Buy", "Buy Signal"]:

        action = "BUY SETUP"

        buy_low = max(support, close - 0.7 * atr)

        buy_high = min(close + 0.25 * atr, close * 1.015)

        target = max(resistance, close + 1.7 * atr)

        stop_loss = buy_low - atr

    elif signal == "Hold / Wait":

        action = "WAIT / HOLD SETUP"

        buy_low = close - 0.6 * atr

        buy_high = close + 0.2 * atr

        target = close + 1.1 * atr

        stop_loss = close - atr

    else:

        action = "AVOID / SELL SETUP"

        buy_low = np.nan

        buy_high = np.nan

        target = close - atr

        stop_loss = close + atr



    hold = "1-5 days" if horizon_days <= 5 else "5-14 days" if horizon_days <= 14 else "2-8 weeks"



    return pd.DataFrame({

        "Action": [action],

        "Buy Zone Low": [round(buy_low, 2) if not pd.isna(buy_low) else None],

        "Buy Zone High": [round(buy_high, 2) if not pd.isna(buy_high) else None],

        "Target": [round(target, 2)],

        "Stop Loss": [round(stop_loss, 2)],

        "Expected Hold": [hold],

        "Confidence": [f"{confidence}%"],

        "Estimated Return": [f"{expected_return:.2%}"]

    })





def make_chart(df, ticker):

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA9"], mode="lines", name="MA9"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))



    fig.update_layout(

        title=f"{ticker} Price Chart",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig





def analyze_ticker(ticker, period, short_days):

    live_price, quote_source = get_live_quote(ticker)

    df = load_price_data(ticker, period)



    if df.empty:

        return None



    df = add_indicators(df, live_price)



    if df.empty:

        return None



    news_df, news_score, news_label = load_news_sentiment(ticker)

    forecast_df, expected_return = estimate_future_price(df, short_days)



    latest = df.iloc[-1]

    signal, score, confidence, reasons = make_signal(latest, news_score)

    plan = trade_plan(latest, signal, confidence, expected_return, short_days)



    source = quote_source if live_price is not None else df["Data_Source"].iloc[-1]



    return {

        "df": df,

        "news_df": news_df,

        "news_score": news_score,

        "news_label": news_label,

        "forecast_df": forecast_df,

        "expected_return": expected_return,

        "latest": latest,

        "signal": signal,

        "score": score,

        "confidence": confidence,

        "reasons": reasons,

        "plan": plan,

        "source": source

    }





st.sidebar.header("Settings")



mode = st.sidebar.selectbox("Trading mode", ["Swing / Multi-day", "Historical / Long-term"], index=0)



period = "1y" if mode == "Swing / Multi-day" else "5y"



short_days = st.sidebar.selectbox("Short-term horizon", [1, 3, 5, 10, 14], index=2)

scan_count = st.sidebar.selectbox("How many stocks to scan?", [5, 10], index=1)



selected_ticker = st.sidebar.selectbox("Select one stock", TICKERS, index=0)



tab1, tab2, tab3 = st.tabs(["Single Stock", "Scanner", "Ticker List"])



with tab1:

    result = analyze_ticker(selected_ticker, period, short_days)



    if result is None:

        st.error("Could not load data.")

    else:

        latest = result["latest"]



        st.subheader(f"{selected_ticker} Trading Plan")

        st.info(f"Data source for current price: {result['source']}")



        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Current Price", f"${latest['Close']:.2f}")

        c2.metric("Signal", result["signal"])

        c3.metric("Score", result["score"])

        c4.metric("Confidence", f"{result['confidence']}%")



        st.markdown("### Trading Plan")

        st.dataframe(result["plan"], use_container_width=True)



        st.markdown("### Price Chart")

        st.plotly_chart(make_chart(result["df"], selected_ticker), use_container_width=True)



        st.markdown("### Estimated Future Price")

        st.dataframe(result["forecast_df"], use_container_width=True)



        st.markdown("### Why this signal?")

        for r in result["reasons"]:

            st.write(f"- {r}")



        st.markdown("### Latest Indicators")

        indicators = pd.DataFrame({

            "Metric": ["Close", "MA9", "MA20", "MA50", "RSI", "MACD", "MACD Signal", "ATR", "Support", "Resistance"],

            "Value": [

                round(latest["Close"], 2),

                round(latest["MA9"], 2),

                round(latest["MA20"], 2),

                round(latest["MA50"], 2),

                round(latest["RSI"], 2),

                round(latest["MACD"], 2),

                round(latest["MACD_Signal"], 2),

                round(latest["ATR"], 2),

                round(latest["Support"], 2),

                round(latest["Resistance"], 2)

            ]

        })

        st.dataframe(indicators, use_container_width=True)



        st.markdown("### Recent News")

        if result["news_df"].empty:

            st.warning("No news loaded. Add FINNHUB_API_KEY in Streamlit secrets.")

        else:

            st.dataframe(result["news_df"], use_container_width=True)



with tab2:

    rows = []



    for ticker in TICKERS[:scan_count]:

        r = analyze_ticker(ticker, period, short_days)



        if r is None:

            continue



        rows.append({

            "Ticker": ticker,

            "Price": round(r["latest"]["Close"], 2),

            "Signal": r["signal"],

            "Score": r["score"],

            "Confidence": r["confidence"],

            "News": r["news_label"],

            "Expected Return %": round(r["expected_return"] * 100, 2),

            "Data Source": r["source"]

        })



    if rows:

        scanner_df = pd.DataFrame(rows).sort_values(["Score", "Confidence"], ascending=False)

        st.dataframe(scanner_df, use_container_width=True)



        st.download_button(

            "Download Scanner Results",

            scanner_df.to_csv(index=False).encode("utf-8"),

            "stock_scanner_results.csv",

            "text/csv"

        )

    else:

        st.warning("No scanner results found.")



with tab3:

    st.subheader("Ticker List")

    st.dataframe(pd.DataFrame({"Ticker": TICKERS}), use_container_width=True)



