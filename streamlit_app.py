import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import requests

import plotly.graph_objects as go

from datetime import datetime, timedelta



st.set_page_config(page_title="AI Trading Signal App", layout="wide")



st.title("📈 AI Trading Signal App")

st.caption("Technical + Fundamental + News + Market Direction + Backtest")

st.warning("Educational only. Not financial advice. No signal is guaranteed.")



FALLBACK = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "SPY", "QQQ"]





@st.cache_data(ttl=86400)

def get_all_tickers():

    tickers = []

    try:

        df1 = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|")

        df1 = df1[df1["Test Issue"] == "N"]

        tickers += df1["Symbol"].astype(str).tolist()

    except Exception:

        pass



    try:

        df2 = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|")

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

    return clean if clean else FALLBACK, pd.DataFrame({"Ticker": clean if clean else FALLBACK})





@st.cache_data(ttl=600)

def load_price_data(ticker, period, interval):

    try:

        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)



        if df is None or df.empty:

            return pd.DataFrame()



        df = df.reset_index()



        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)



        if "Datetime" in df.columns:

            df = df.rename(columns={"Datetime": "Date"})



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





@st.cache_data(ttl=1800)

def load_news_sentiment(ticker):

    try:

        api_key = st.secrets.get("FINNHUB_API_KEY", "")



        if api_key == "":

            return pd.DataFrame(), 0, "No API Key"



        today = datetime.today().date()

        start = today - timedelta(days=7)



        url = "https://finnhub.io/api/v1/company-news"

        params = {"symbol": ticker, "from": str(start), "to": str(today), "token": api_key}



        r = requests.get(url, params=params, timeout=10)



        if r.status_code != 200:

            return pd.DataFrame(), 0, "News Error"



        news = r.json()



        if not news:

            return pd.DataFrame(), 0, "No Recent News"



        positive_words = [

            "beat", "growth", "strong", "upgrade", "surge", "record", "profit",

            "higher", "bullish", "positive", "gain", "rally", "outperform",

            "raises", "increase", "ai", "partnership", "approval"

        ]



        negative_words = [

            "miss", "drop", "fall", "lawsuit", "weak", "downgrade", "loss",

            "lower", "bearish", "negative", "decline", "cut", "warning",

            "risk", "investigation", "delay", "probe"

        ]



        rows = []

        total = 0



        for item in news[:20]:

            headline = str(item.get("headline", ""))

            summary = str(item.get("summary", ""))

            text = (headline + " " + summary).lower()



            pos = sum(1 for w in positive_words if w in text)

            neg = sum(1 for w in negative_words if w in text)



            score = pos - neg

            total += score



            sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"



            rows.append({

                "Date": datetime.fromtimestamp(item.get("datetime")).strftime("%Y-%m-%d") if item.get("datetime") else None,

                "Headline": headline,

                "Sentiment": sentiment,

                "Source": item.get("source"),

                "URL": item.get("url")

            })



        avg = total / max(len(rows), 1)



        if avg > 0.25:

            label = "Positive News"

        elif avg < -0.25:

            label = "Negative News"

        else:

            label = "Neutral News"



        return pd.DataFrame(rows), avg, label



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



    if "High" in df.columns and "Low" in df.columns:

        tr1 = df["High"] - df["Low"]

        tr2 = (df["High"] - df["Close"].shift()).abs()

        tr3 = (df["Low"] - df["Close"].shift()).abs()

        df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["ATR"] = df["TR"].rolling(14).mean()

    else:

        df["ATR"] = df["Close"].rolling(14).std()



    if "Volume" in df.columns:

        df["Volume_MA20"] = df["Volume"].rolling(20).mean()

        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]



        if "High" in df.columns and "Low" in df.columns:

            typical = (df["High"] + df["Low"] + df["Close"]) / 3

        else:

            typical = df["Close"]



        volume_sum = df["Volume"].replace(0, np.nan).cumsum()

        df["VWAP"] = (typical * df["Volume"]).cumsum() / volume_sum

    else:

        df["Volume_Ratio"] = 1

        df["VWAP"] = df["Close"]



    df["Support"] = df["Close"].rolling(30).min()

    df["Resistance"] = df["Close"].rolling(30).max()



    return df.dropna()





@st.cache_data(ttl=900)

def get_market_direction():

    rows = []



    for ticker in ["SPY", "QQQ"]:

        df = load_price_data(ticker, "6mo", "1d")



        if df.empty:

            continue



        df = add_indicators(df)



        if df.empty:

            continue



        latest = df.iloc[-1]



        score = 0



        if latest["Close"] > latest["MA20"]:

            score += 1

        if latest["Close"] > latest["MA50"]:

            score += 1

        if latest["MA20"] > latest["MA50"]:

            score += 1

        if latest["MACD"] > latest["MACD_Signal"]:

            score += 1



        rows.append({"Ticker": ticker, "Market Score": score})



    if not rows:

        return 0, "Unknown Market"



    market_score = sum(r["Market Score"] for r in rows)



    if market_score >= 6:

        label = "Bullish Market"

    elif market_score >= 3:

        label = "Mixed Market"

    else:

        label = "Bearish Market"



    return market_score, label





def estimate_future_price(df, days):

    recent = df.tail(252).copy()



    if len(recent) < 60:

        return pd.DataFrame(), 0, "Not enough data"



    returns = recent["Close"].pct_change().dropna()

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



    if expected_return >= 0.08:

        label = "Strong Positive Estimate"

    elif expected_return >= 0.03:

        label = "Positive Estimate"

    elif expected_return <= -0.08:

        label = "Strong Negative Estimate"

    elif expected_return <= -0.03:

        label = "Negative Estimate"

    else:

        label = "Neutral Estimate"



    out = pd.DataFrame({

        "Forecast Horizon": [f"{days} days"],

        "Current Price": [round(last_price, 2)],

        "Base Estimated Price": [round(base_price, 2)],

        "Bull Case Price": [round(bull_price, 2)],

        "Bear Case Price": [round(bear_price, 2)],

        "Estimated Return": [f"{expected_return:.2%}"],

        "Forecast Label": [label]

    })



    return out, expected_return, label





def make_price_chart(df, ticker, title_suffix=""):

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA9"], mode="lines", name="MA9"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP"))



    fig.update_layout(

        title=f"{ticker} Price Chart {title_suffix}",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig





def technical_score_only(latest):

    score = 0

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

    vol = safe_num(latest["Volatility



# ================================

# PART 2/2

# Add below your previous code

# ================================



def get_market_direction():



    spy = load_price_data("SPY", "6mo", "1d")

    qqq = load_price_data("QQQ", "6mo", "1d")



    if spy.empty or qqq.empty:

        return "Neutral"



    spy = add_indicators(spy)

    qqq = add_indicators(qqq)



    spy_latest = spy.iloc[-1]

    qqq_latest = qqq.iloc[-1]



    score = 0



    if spy_latest["Close"] > spy_latest["MA50"]:

        score += 1



    if qqq_latest["Close"] > qqq_latest["MA50"]:

        score += 1



    if spy_latest["MACD"] > spy_latest["MACD_Signal"]:

        score += 1



    if qqq_latest["MACD"] > qqq_latest["MACD_Signal"]:

        score += 1



    if score >= 3:

        return "Bullish"

    elif score <= 1:

        return "Bearish"

    else:

        return "Neutral"





def calculate_trade_levels(latest):



    close = safe_num(latest["Close"])

    atr = safe_num(latest["ATR"], close * 0.02)



    support = safe_num(latest["Support"], close - atr)

    resistance = safe_num(latest["Resistance"], close + atr)



    buy_low = max(support, close - 0.60 * atr)

    buy_high = close + 0.25 * atr



    target1 = close + 1.5 * atr

    target2 = close + 3.0 * atr



    stop_loss = close - 1.0 * atr



    return {

        "buy_low": round(buy_low, 2),

        "buy_high": round(buy_high, 2),

        "target1": round(target1, 2),

        "target2": round(target2, 2),

        "stop_loss": round(stop_loss, 2)

    }





def classify_signal(

    total_score,

    market_direction,

    risk,

    news_score

):



    if (

        total_score >= 14

        and market_direction == "Bullish"

        and risk != "High"

        and news_score >= 0

    ):

        return "Strong Buy"



    if (

        total_score >= 10

        and risk != "High"

    ):

        return "Buy on Dip"



    if (

        total_score >= 7

    ):

        return "Hold / Wait"



    if risk == "High":

        return "Avoid / High Risk"



    return "Sell / Avoid"





def realistic_confidence(

    total_score,

    risk,

    market_direction,

    news_score

):



    conf = 50



    conf += total_score * 2



    if market_direction == "Bullish":

        conf += 8



    if market_direction == "Bearish":

        conf -= 10



    if news_score > 0.25:

        conf += 5



    if news_score < -0.25:

        conf -= 5



    if risk == "High":

        conf -= 15



    if risk == "Low":

        conf += 5



    conf = max(10, min(85, conf))



    return int(conf)





def estimate_hold_days(signal):



    if signal == "Strong Buy":

        return "5-20 days"



    if signal == "Buy on Dip":

        return "3-14 days"



    if signal == "Hold / Wait":

        return "Wait for better setup"



    return "Avoid"





def make_summary_table(

    ticker,

    latest,

    signal,

    confidence,

    market_direction,

    trade_levels,

    expected_return,

    risk

):



    return pd.DataFrame({

        "Ticker": [ticker],

        "Current Price": [round(latest["Close"], 2)],

        "Market Direction": [market_direction],

        "Signal": [signal],

        "Confidence": [f"{confidence}%"],

        "Risk": [risk],



        "BUY ZONE": [

            f"{trade_levels['buy_low']} - {trade_levels['buy_high']}"

        ],



        "TARGET 1": [trade_levels["target1"]],

        "TARGET 2": [trade_levels["target2"]],

        "STOP LOSS": [trade_levels["stop_loss"]],



        "Expected Return": [

            f"{expected_return:.2%}"

        ],



        "Expected Hold": [

            estimate_hold_days(signal)

        ]

    })





# ====================================

# MAIN APP SECTION

# ====================================



tickers, ticker_df = get_all_tickers()



st.sidebar.header("Settings")



mode = st.sidebar.selectbox(

    "Trading Mode",

    [

        "Intraday / Short-term",

        "Swing / Multi-day",

        "Historical / Long-term"

    ]

)



if mode == "Intraday / Short-term":

    period = "5d"

    interval = "5m"



elif mode == "Swing / Multi-day":

    period = "1y"

    interval = "1d"



else:

    period = "5y"

    interval = "1d"



selected_ticker = st.sidebar.selectbox(

    "Select Stock",

    tickers

)



forecast_days = st.sidebar.selectbox(

    "Forecast Horizon",

    [5, 10, 30, 60, 90],

    index=1

)



scan_count = st.sidebar.selectbox(

    "Scanner Count",

    [25, 50, 100, 200, 300, 500],

    index=2

)



tab1, tab2, tab3 = st.tabs([

    "Single Stock",

    "Scanner",

    "Ticker List"

])



# ====================================

# TAB 1

# ====================================



with tab1:



    market_direction = get_market_direction()



    st.info(

        f"Overall Market Direction: {market_direction}"

    )



    df = load_price_data(

        selected_ticker,

        period,

        interval

    )



    if df.empty:

        st.error("Could not load data.")



    else:



        df = add_indicators(df)



        if df.empty:

            st.error("Not enough data.")



        else:



            latest = df.iloc[-1]



            fundamentals = load_fundamentals(

                selected_ticker

            )



            news_df, news_score, news_label = load_news_sentiment(

                selected_ticker

            )



            estimate_df, expected_return, forecast_label = estimate_future_price(

                df,

                forecast_days

            )



            scores = score_stock(

                latest,

                fundamentals,

                expected_return,

                news_score

            )



            signal = classify_signal(

                scores["total_score"],

                market_direction,

                scores["risk"],

                news_score

            )



            confidence = realistic_confidence(

                scores["total_score"],

                scores["risk"],

                market_direction,

                news_score

            )



            trade_levels = calculate_trade_levels(

                latest

            )



            summary_df = make_summary_table(

                selected_ticker,

                latest,

                signal,

                confidence,

                market_direction,

                trade_levels,

                expected_return,

                scores["risk"]

            )



            st.subheader(

                f"{selected_ticker} AI Trading Summary"

            )



            c1, c2, c3, c4, c5 = st.columns(5)



            c1.metric(

                "Current Price",

                f"${latest['Close']:.2f}"

            )



            c2.metric(

                "Signal",

                signal

            )



            c3.metric(

                "Confidence",

                f"{confidence}%"

            )



            c4.metric(

                "Risk",

                scores["risk"]

            )



            c5.metric(

                "Market",

                market_direction

            )



            st.markdown("## Trading Plan")



            st.dataframe(

                summary_df,

                use_container_width=True

            )



            st.markdown("## Historical Chart")



            st.plotly_chart(

                make_price_chart(

                    df,

                    selected_ticker

                ),

                use_container_width=True

            )



            st.markdown("## Forecast Summary")



            st.dataframe(

                estimate_df,

                use_container_width=True

            )



            st.markdown("## Signal Explanations")



            for r in scores["reasons"]:

                st.write(f"- {r}")



            st.markdown("## Fundamentals")



            st.dataframe(

                pd.DataFrame({

                    "Metric": list(fundamentals.keys()),

                    "Value": list(fundamentals.values())

                }),

                use_container_width=True

            )



            st.markdown("## Recent News")



            if news_df.empty:



                st.warning(

                    "No news found or no FINNHUB_API_KEY."

                )



            else:



                st.dataframe(

                    news_df,

                    use_container_width=True

                )



# ====================================

# TAB 2

# ====================================



with tab2:



    st.subheader("Scanner")



    rows = []



    progress = st.progress(0)



    for i, ticker in enumerate(

        tickers[:scan_count]

    ):



        try:



            df = load_price_data(

                ticker,

                period,

                interval

            )



            if df.empty:

                continue



            df = add_indicators(df)



            if df.empty:

                continue



            latest = df.iloc[-1]



            fundamentals = load_fundamentals(

                ticker

            )



            _, news_score, _ = load_news_sentiment(

                ticker

            )



            _, expected_return, _ = estimate_future_price(

                df,

                forecast_days

            )



            scores = score_stock(

                latest,

                fundamentals,

                expected_return,

                news_score

            )



            signal = classify_signal(

                scores["total_score"],

                get_market_direction(),

                scores["risk"],

                news_score

            )



            confidence = realistic_confidence(

                scores["total_score"],

                scores["risk"],

                get_market_direction(),

                news_score

            )



            rows.append({

                "Ticker": ticker,

                "Price": round(latest["Close"], 2),

                "Signal": signal,

                "Confidence": confidence,

                "Risk": scores["risk"],

                "Expected Return": round(

                    expected_return * 100,

                    2

                )

            })



        except Exception:

            pass



        progress.progress(

            (i + 1) / scan_count

        )



    if rows:



        scan_df = pd.DataFrame(rows)



        order = {

            "Strong Buy": 1,

            "Buy on Dip": 2,

            "Hold / Wait": 3,

            "Avoid / High Risk": 4,

            "Sell / Avoid": 5

        }



        scan_df["sort"] = scan_df["Signal"].map(order)



        scan_df = scan_df.sort_values(

            ["sort", "Confidence"],

            ascending=[True, False]

        )



        scan_df = scan_df.drop(

            columns=["sort"]

        )



        st.dataframe(

            scan_df,

            use_container_width=True

        )



        st.download_button(

            "Download Scanner Results",

            scan_df.to_csv(index=False).encode("utf-8"),

            "scanner_results.csv",

            "text/csv"

        )



# ====================================

# TAB 3

# ====================================



with tab3:



    st.subheader("All Loaded Tickers")



    st.dataframe(

        ticker_df,

        use_container_width=True

    )

