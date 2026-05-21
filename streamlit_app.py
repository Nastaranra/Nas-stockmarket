import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import requests

import plotly.graph_objects as go

from datetime import datetime, timedelta



st.set_page_config(page_title="AI Stock Decision Support", layout="wide")



st.title("📈 AI Stock Decision Support App")

st.caption("Technical + Fundamental + News Sentiment | Educational only, not financial advice.")

st.warning("This is NOT financial advice. Signals are educational estimates only.")



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





@st.cache_data(ttl=900)

def load_price_data(ticker, period, interval):

    try:

        df = yf.download(

            ticker,

            period=period,

            interval=interval,

            auto_adjust=True,

            progress=False,

            threads=False

        )



        if df is None or df.empty:

            return pd.DataFrame()



        df = df.reset_index()



        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)



        if "Close" not in df.columns:

            return pd.DataFrame()



        if "Datetime" in df.columns:

            df = df.rename(columns={"Datetime": "Date"})



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





@st.cache_data(ttl=1800)

def load_news_sentiment(ticker):

    try:

        api_key = st.secrets.get("FINNHUB_API_KEY", "")



        if api_key == "":

            return pd.DataFrame(), 0, "No API Key"



        today = datetime.today().date()

        start = today - timedelta(days=7)



        url = "https://finnhub.io/api/v1/company-news"



        params = {

            "symbol": ticker,

            "from": str(start),

            "to": str(today),

            "token": api_key

        }



        response = requests.get(url, params=params, timeout=10)



        if response.status_code != 200:

            return pd.DataFrame(), 0, "News Error"



        news = response.json()



        if not news:

            return pd.DataFrame(), 0, "No Recent News"



        rows = []



        positive_words = [

            "beat", "beats", "growth", "strong", "upgrade", "surge", "record",

            "profit", "higher", "bullish", "positive", "gain", "rally",

            "outperform", "raises", "increase"

        ]



        negative_words = [

            "miss", "misses", "drop", "fall", "lawsuit", "weak", "downgrade",

            "loss", "lower", "bearish", "negative", "decline", "cut",

            "warning", "risk", "investigation"

        ]



        sentiment_score = 0



        for item in news[:20]:

            headline = str(item.get("headline", ""))

            summary = str(item.get("summary", ""))

            text = (headline + " " + summary).lower()



            pos = sum(1 for w in positive_words if w in text)

            neg = sum(1 for w in negative_words if w in text)



            score = pos - neg

            sentiment_score += score



            if score > 0:

                sentiment = "Positive"

            elif score < 0:

                sentiment = "Negative"

            else:

                sentiment = "Neutral"



            rows.append({

                "Date": datetime.fromtimestamp(item.get("datetime")).strftime("%Y-%m-%d") if item.get("datetime") else None,

                "Headline": headline,

                "Sentiment": sentiment,

                "Source": item.get("source"),

                "URL": item.get("url")

            })



        avg_score = sentiment_score / max(len(rows), 1)



        if avg_score > 0.25:

            label = "Positive News Sentiment"

        elif avg_score < -0.25:

            label = "Negative News Sentiment"

        else:

            label = "Neutral News Sentiment"



        return pd.DataFrame(rows), avg_score, label



    except Exception:

        return pd.DataFrame(), 0, "News Error"





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

    df["MA9"] = df["Close"].rolling(9).mean()

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    df["MA200"] = df["Close"].rolling(200).mean()



    df["Return_5"] = df["Close"].pct_change(5)

    df["Return_20"] = df["Close"].pct_change(20)

    df["Volatility"] = df["Return"].rolling(20).std() * np.sqrt(252)



    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)



    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()



    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))



    exp1 = df["Close"].ewm(span=12, adjust=False).mean()

    exp2 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = exp1 - exp2

    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()



    if "Volume" in df.columns:

        df["Volume_MA20"] = df["Volume"].rolling(20).mean()

        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    else:

        df["Volume_Ratio"] = 1



    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3 if "High" in df.columns and "Low" in df.columns else df["Close"]

    df["VWAP"] = (typical_price * df.get("Volume", 1)).cumsum() / df.get("Volume", 1).cumsum()



    return df.dropna()





def estimate_future_price(df, days):

    recent = df.tail(252).copy()



    if len(recent) < 120:

        return pd.DataFrame(), 0, "Not enough data"



    daily_returns = recent["Close"].pct_change().dropna()



    avg_return = daily_returns.mean()

    volatility = daily_returns.std()



    last_price = recent["Close"].iloc[-1]



    base_price = last_price

    bull_price = last_price

    bear_price = last_price



    for _ in range(days):

        base_price = base_price * (1 + avg_return)

        bull_price = bull_price * (1 + avg_return + volatility * 0.35)

        bear_price = bear_price * (1 + avg_return - volatility * 0.35)



    expected_return = (base_price / last_price) - 1



    if expected_return >= 0.08:

        forecast_label = "Strong Positive Estimate"

    elif expected_return >= 0.03:

        forecast_label = "Positive Estimate"

    elif expected_return <= -0.08:

        forecast_label = "Strong Negative Estimate"

    elif expected_return <= -0.03:

        forecast_label = "Negative Estimate"

    else:

        forecast_label = "Neutral Estimate"



    result = pd.DataFrame({

        "Forecast Horizon": [f"{days} days"],

        "Current Price": [round(last_price, 2)],

        "Base Estimated Price": [round(base_price, 2)],

        "Bull Case Price": [round(bull_price, 2)],

        "Bear Case Price": [round(bear_price, 2)],

        "Estimated Return": [f"{expected_return:.2%}"],

        "Forecast Label": [forecast_label]

    })



    return result, expected_return, forecast_label





def make_price_chart(df, ticker):

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA9"], mode="lines", name="MA9"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))



    fig.update_layout(

        title=f"{ticker} Historical Price Chart",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig





def score_stock(latest, fundamentals, expected_return, news_score):

    technical_score = 0

    fundamental_score = 0

    forecast_score = 0

    news_signal_score = 0

    reasons = []



    close = safe_num(latest["Close"])

    ma9 = safe_num(latest["MA9"])

    ma20 = safe_num(latest["MA20"])

    ma50 = safe_num(latest["MA50"])

    rsi = safe_num(latest["RSI"])

    macd = safe_num(latest["MACD"])

    macd_signal = safe_num(latest["MACD_Signal"])

    vwap = safe_num(latest["VWAP"])

    volume_ratio = safe_num(latest["Volume_Ratio"], 1)

    vol = safe_num(latest["Volatility"], 1)



    if close > ma9:

        technical_score += 1

        reasons.append("Price is above MA9.")

    if close > ma20:

        technical_score += 1

        reasons.append("Price is above MA20.")

    if ma9 > ma20:

        technical_score += 1

        reasons.append("Short-term trend is positive.")

    if ma20 > ma50:

        technical_score += 1

        reasons.append("Medium-term trend is positive.")

    if close > vwap:

        technical_score += 1

        reasons.append("Price is above VWAP.")

    if macd > macd_signal:

        technical_score += 1

        reasons.append("MACD is bullish.")

    if 45 <= rsi <= 65:

        technical_score += 1

        reasons.append("RSI is healthy.")

    elif rsi > 70:

        technical_score -= 1

        reasons.append("RSI may be overbought.")

    elif rsi < 30:

        technical_score -= 1

        reasons.append("RSI is weak/oversold.")



    if volume_ratio > 1.5:

        technical_score += 1

        reasons.append("Volume is above average.")



    if vol < 0.30:

        risk = "Low"

    elif vol < 0.55:

        risk = "Medium"

    else:

        risk = "High"

        technical_score -= 2



    pe = safe_num(fundamentals.get("P/E Ratio"), None)

    fpe = safe_num(fundamentals.get("Forward P/E"), None)

    margin = safe_num(fundamentals.get("Profit Margin"), None)

    growth = safe_num(fundamentals.get("Revenue Growth"), None)

    debt = safe_num(fundamentals.get("Debt to Equity"), None)

    roe = safe_num(fundamentals.get("ROE"), None)



    if pe is not None and 0 < pe < 40:

        fundamental_score += 1

        reasons.append("P/E ratio is acceptable.")

    if fpe is not None and 0 < fpe < 40:

        fundamental_score += 1

        reasons.append("Forward P/E is acceptable.")

    if margin is not None and margin > 0.10:

        fundamental_score += 1

        reasons.append("Profit margin is strong.")

    if growth is not None and growth > 0.05:

        fundamental_score += 1

        reasons.append("Revenue growth is positive.")

    if debt is not None and debt < 200:

        fundamental_score += 1

        reasons.append("Debt-to-equity is manageable.")

    if roe is not None and roe > 0.10:

        fundamental_score += 1

        reasons.append("ROE is strong.")



    if expected_return >= 0.05:

        forecast_score += 2

        reasons.append("Estimated future return is positive.")

    elif expected_return <= -0.05:

        forecast_score -= 2

        reasons.append("Estimated future return is negative.")



    if news_score > 0.25:

        news_signal_score += 2

        reasons.append("Recent news sentiment is positive.")

    elif news_score < -0.25:

        news_signal_score -= 2

        reasons.append("Recent news sentiment is negative.")

    else:

        reasons.append("Recent news sentiment is neutral or unavailable.")



    total_score = technical_score + fundamental_score + forecast_score + news_signal_score



    if total_score >= 11 and risk != "High":

        signal = "Buy Signal"

    elif total_score <= 3 or risk == "High":

        signal = "Sell Signal / High Caution"

    else:

        signal = "Hold / Wait"



    return {

        "technical_score": technical_score,

        "fundamental_score": fundamental_score,

        "forecast_score": forecast_score,

        "news_score_component": news_signal_score,

        "total_score": total_score,

        "risk": risk,

        "signal": signal,

        "reasons": reasons

    }





def scanner_score(ticker, period, interval):

    df = load_price_data(ticker, period, interval)



    if df.empty:

        return None



    df = add_indicators(df)



    if df.empty:

        return None



    latest = df.iloc[-1]



    close = safe_num(latest["Close"])

    ma9 = safe_num(latest["MA9"])

    ma20 = safe_num(latest["MA20"])

    ma50 = safe_num(latest["MA50"])

    rsi = safe_num(latest["RSI"])

    macd = safe_num(latest["MACD"])

    macd_signal = safe_num(latest["MACD_Signal"])

    vwap = safe_num(latest["VWAP"])

    vol = safe_num(latest["Volatility"], 1)



    score = 0



    if close > ma9:

        score += 1

    if close > ma20:

        score += 1

    if ma9 > ma20:

        score += 1

    if ma20 > ma50:

        score += 1

    if close > vwap:

        score += 1

    if macd > macd_signal:

        score += 1

    if 45 <= rsi <= 65:

        score += 1

    elif rsi > 70 or rsi < 30:

        score -= 1



    if vol > 0.55:

        risk = "High"

        score -= 2

    elif vol > 0.30:

        risk = "Medium"

    else:

        risk = "Low"



    if score >= 6 and risk != "High":

        signal = "Buy Signal"

    elif score <= 1 or risk == "High":

        signal = "Sell Signal / High Caution"

    else:

        signal = "Hold / Wait"



    return {

        "Ticker": ticker,

        "Price": round(close, 2),

        "RSI": round(rsi, 1),

        "Score": score,

        "Risk": risk,

        "Signal": signal

    }





tickers, ticker_df = get_all_tickers()



st.sidebar.header("Settings")



mode = st.sidebar.selectbox(

    "Trading mode",

    ["Intraday / Short-term", "Swing / Multi-day"],

    index=0

)



if mode == "Intraday / Short-term":

    period = "5d"

    interval = "5m"

else:

    period = "1y"

    interval = "1d"



forecast_days = st.sidebar.selectbox(

    "Forecast horizon",

    [1, 3, 5, 10, 30, 60, 90],

    index=2

)



scan_count = st.sidebar.selectbox(

    "How many stocks to scan?",

    [25, 50, 100, 200, 300, 500],

    index=1

)



scan_count = min(scan_count, len(tickers))



st.sidebar.write(f"Available tickers loaded: {len(tickers)}")



selected_ticker = st.sidebar.selectbox("Select one stock", tickers)



tab1, tab2, tab3 = st.tabs(["Single Stock", "Scanner", "Ticker List"])





with tab1:

    df = load_price_data(selected_ticker, period, interval)



    if df.empty:

        st.error("Could not load data.")

    else:

        df = add_indicators(df)



        if df.empty:

            st.error("Not enough data.")

        else:

            fundamentals = load_fundamentals(selected_ticker)

            news_df, news_score, news_label = load_news_sentiment(selected_ticker)

            estimate_df, expected_return, forecast_label = estimate_future_price(df, forecast_days)



            latest = df.iloc[-1]



            scores = score_stock(

                latest,

                fundamentals,

                expected_return,

                news_score

            )



            st.subheader(f"{selected_ticker} Summary")



            c1, c2, c3, c4, c5 = st.columns(5)



            c1.metric("Current Price", f"${latest['Close']:.2f}")

            c2.metric("Risk", scores["risk"])

            c3.metric("Total Score", scores["total_score"])

            c4.metric("News", news_label)

            c5.metric("Signal", scores["signal"])



            st.info(

                f"Educational signal: {scores['signal']} | "

                f"Mode: {mode} | Forecast label: {forecast_label}"

            )



            st.markdown("### Historical Price Chart")

            st.plotly_chart(make_price_chart(df, selected_ticker), use_container_width=True)



            st.markdown("### Estimated Future Price")

            st.dataframe(estimate_df, use_container_width=True)



            st.markdown("### Score Breakdown")

            score_df = pd.DataFrame({

                "Category": ["Technical", "Fundamental", "Forecast", "News", "Total"],

                "Score": [

                    scores["technical_score"],

                    scores["fundamental_score"],

                    scores["forecast_score"],

                    scores["news_score_component"],

                    scores["total_score"]

                ]

            })

            st.dataframe(score_df, use_container_width=True)



            st.markdown("### Why this signal?")

            for reason in scores["reasons"]:

                st.write(f"- {reason}")



            st.markdown("### Fundamentals")

            st.dataframe(

                pd.DataFrame({

                    "Metric": list(fundamentals.keys()),

                    "Value": list(fundamentals.values())

                }),

                use_container_width=True

            )



            st.markdown("### Recent News")

            if news_df.empty:

                st.warning("No news loaded. Add FINNHUB_API_KEY in Streamlit secrets.")

            else:

                st.dataframe(news_df, use_container_width=True)





with tab2:

    st.subheader("Scanner")

    st.write(f"Scanning first {scan_count} stocks. Scanner uses technical signals only for speed.")



    rows = []

    progress = st.progress(0)



    for i, ticker in enumerate(tickers[:scan_count]):

        result = scanner_score(ticker, period, interval)



        if result is not None:

            rows.append(result)



        progress.progress((i + 1) / scan_count)



    if rows:

        scanner_df = pd.DataFrame(rows).sort_values("Score", ascending=False)

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

    st.write(f"Total tickers loaded: {len(ticker_df)}")

    st.dataframe(ticker_df, use_container_width=True)