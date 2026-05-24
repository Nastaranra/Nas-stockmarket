import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import requests

import plotly.graph_objects as go

from datetime import datetime, timedelta

from streamlit_autorefresh import st_autorefresh



st.set_page_config(page_title="AI Trading Signal App", layout="wide")



st.title("📈 AI Trading Signal App")

st.caption("Live Price + Locked Trading Plan + Technical + Fundamental + News + Market Direction")

st.warning("Educational only. Not financial advice. No signal is guaranteed.")



st_autorefresh(interval=180 * 1000, key="live_refresh")



FALLBACK = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "SPY", "QQQ"]





def safe_num(x, default=0):

    try:

        if x is None or pd.isna(x):

            return default

        return float(x)

    except Exception:

        return default





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

    if not clean:

        clean = FALLBACK



    return clean, pd.DataFrame({"Ticker": clean})





@st.cache_data(ttl=30)

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



        positive_words = [

            "beat", "growth", "strong", "upgrade", "surge", "record",

            "profit", "higher", "bullish", "positive", "gain", "rally",

            "outperform", "raises", "increase", "ai", "partnership"

        ]



        negative_words = [

            "miss", "drop", "fall", "lawsuit", "weak", "downgrade",

            "loss", "lower", "bearish", "negative", "decline", "cut",

            "warning", "risk", "investigation", "delay"

        ]



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

            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3

        else:

            typical_price = df["Close"]



        volume_sum = df["Volume"].replace(0, np.nan).cumsum()

        df["VWAP"] = (typical_price * df["Volume"]).cumsum() / volume_sum

    else:

        df["Volume_Ratio"] = 1

        df["VWAP"] = df["Close"]



    df["Support"] = df["Close"].rolling(30).min()

    df["Resistance"] = df["Close"].rolling(30).max()



    df = df.replace([np.inf, -np.inf], np.nan)

    return df.dropna()





@st.cache_data(ttl=300)

def get_market_direction():

    try:

        spy = yf.download("SPY", period="6mo", interval="1d", auto_adjust=True, progress=False, threads=False)

        qqq = yf.download("QQQ", period="6mo", interval="1d", auto_adjust=True, progress=False, threads=False)



        if spy is None or spy.empty or qqq is None or qqq.empty:

            return 0, "Unknown Market"



        spy = spy.reset_index()

        qqq = qqq.reset_index()



        if isinstance(spy.columns, pd.MultiIndex):

            spy.columns = spy.columns.get_level_values(0)

        if isinstance(qqq.columns, pd.MultiIndex):

            qqq.columns = qqq.columns.get_level_values(0)



        spy["MA20"] = spy["Close"].rolling(20).mean()

        spy["MA50"] = spy["Close"].rolling(50).mean()

        qqq["MA20"] = qqq["Close"].rolling(20).mean()

        qqq["MA50"] = qqq["Close"].rolling(50).mean()



        spy = spy.dropna()

        qqq = qqq.dropna()



        if spy.empty or qqq.empty:

            return 0, "Unknown Market"



        spy_latest = spy.iloc[-1]

        qqq_latest = qqq.iloc[-1]



        score = 0



        if safe_num(spy_latest["Close"]) > safe_num(spy_latest["MA20"]):

            score += 1

        if safe_num(spy_latest["MA20"]) > safe_num(spy_latest["MA50"]):

            score += 1

        if safe_num(qqq_latest["Close"]) > safe_num(qqq_latest["MA20"]):

            score += 1

        if safe_num(qqq_latest["MA20"]) > safe_num(qqq_latest["MA50"]):

            score += 1



        if score >= 3:

            return score, "Bullish Market"

        elif score <= 1:

            return score, "Bearish Market"

        else:

            return score, "Sideways Market"



    except Exception:

        return 0, "Unknown Market"





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

    vol = safe_num(latest["Volatility"], 1)



    if close > ma9:

        score += 1

        reasons.append("Price is above MA9.")

    if close > ma20:

        score += 1

        reasons.append("Price is above MA20.")

    if ma9 > ma20:

        score += 1

        reasons.append("Short-term trend is positive.")

    if ma20 > ma50:

        score += 1

        reasons.append("Medium-term trend is positive.")

    if close > vwap:

        score += 1

        reasons.append("Price is above VWAP.")

    if macd > macd_signal:

        score += 1

        reasons.append("MACD is bullish.")



    if 45 <= rsi <= 68:

        score += 1

        reasons.append("RSI is in a healthy range.")

    elif rsi > 72:

        score -= 1

        reasons.append("RSI is overbought.")

    elif rsi < 32:

        score -= 1

        reasons.append("RSI is weak or oversold.")



    if volume_ratio > 1.4:

        score += 1

        reasons.append("Volume is above average.")



    if vol > 0.65:

        risk = "High"

        score -= 2

        reasons.append("Volatility is high.")

    elif vol > 0.35:

        risk = "Medium"

        reasons.append("Volatility is moderate.")

    else:

        risk = "Low"

        reasons.append("Volatility is low.")



    return score, risk, reasons





def fundamental_score_only(fundamentals):

    score = 0

    reasons = []



    pe = safe_num(fundamentals.get("P/E Ratio"), None)

    fpe = safe_num(fundamentals.get("Forward P/E"), None)

    margin = safe_num(fundamentals.get("Profit Margin"), None)

    growth = safe_num(fundamentals.get("Revenue Growth"), None)

    debt = safe_num(fundamentals.get("Debt to Equity"), None)

    roe = safe_num(fundamentals.get("ROE"), None)



    if pe is not None and 0 < pe < 45:

        score += 1

        reasons.append("P/E ratio is acceptable.")

    if fpe is not None and 0 < fpe < 45:

        score += 1

        reasons.append("Forward P/E is acceptable.")

    if margin is not None and margin > 0.10:

        score += 1

        reasons.append("Profit margin is strong.")

    if growth is not None and growth > 0.05:

        score += 1

        reasons.append("Revenue growth is positive.")

    if debt is not None and debt < 220:

        score += 1

        reasons.append("Debt-to-equity is manageable.")

    if roe is not None and roe > 0.10:

        score += 1

        reasons.append("ROE is strong.")



    return score, reasons





def signal_type(total_score, risk, market_label, news_score, expected_return):

    if risk == "High":

        return "Avoid / High Risk"



    if news_score < -0.25:

        return "Avoid / Negative News"



    if market_label == "Bearish Market" and total_score < 13:

        return "Wait for Market Confirmation"



    if total_score >= 14 and expected_return > 0:

        return "Strong Buy"

    elif total_score >= 11 and expected_return > 0:

        return "Buy Signal"

    elif total_score >= 9:

        return "Buy on Dip"

    elif total_score >= 6:

        return "Hold / Wait"

    else:

        return "Sell / High Caution"





def score_stock(latest, fundamentals, expected_return, news_score, market_score, market_label):

    tech_score, risk, tech_reasons = technical_score_only(latest)

    fund_score, fund_reasons = fundamental_score_only(fundamentals)



    forecast_score = 0

    forecast_reasons = []



    if expected_return >= 0.05:

        forecast_score += 2

        forecast_reasons.append("Estimated future return is positive.")

    elif expected_return <= -0.05:

        forecast_score -= 2

        forecast_reasons.append("Estimated future return is negative.")

    else:

        forecast_reasons.append("Estimated future return is neutral.")



    news_component = 0

    news_reasons = []



    if news_score > 0.25:

        news_component += 2

        news_reasons.append("Recent news sentiment is positive.")

    elif news_score < -0.25:

        news_component -= 2

        news_reasons.append("Recent news sentiment is negative.")

    else:

        news_reasons.append("Recent news sentiment is neutral or unavailable.")



    market_component = 0

    market_reasons = []



    if market_label == "Bullish Market":

        market_component += 2

        market_reasons.append("Overall market trend is bullish.")

    elif market_label == "Bearish Market":

        market_component -= 2

        market_reasons.append("Overall market trend is bearish.")

    elif market_label == "Sideways Market":

        market_reasons.append("Overall market trend is sideways.")

    else:

        market_reasons.append("Market direction is unavailable.")



    total_score = tech_score + fund_score + forecast_score + news_component + market_component

    signal = signal_type(total_score, risk, market_label, news_score, expected_return)



    return {

        "technical_score": tech_score,

        "fundamental_score": fund_score,

        "forecast_score": forecast_score,

        "news_score_component": news_component,

        "market_score_component": market_component,

        "total_score": total_score,

        "risk": risk,

        "signal": signal,

        "reasons": tech_reasons + fund_reasons + forecast_reasons + news_reasons + market_reasons

    }





def confidence_score(scores, expected_return, news_score):

    base = 45



    base += scores["technical_score"] * 2.2

    base += scores["fundamental_score"] * 1.5

    base += scores["forecast_score"] * 2.5

    base += scores["news_score_component"] * 2

    base += scores["market_score_component"] * 2



    if expected_return > 0.05:

        base += 3

    elif expected_return < -0.05:

        base -= 4



    if scores["risk"] == "High":

        base -= 18

    elif scores["risk"] == "Low":

        base += 4



    if news_score < -0.25:

        base -= 8



    return int(max(35, min(85, base)))





def build_locked_plan(latest, signal, confidence, expected_return, horizon_days, risk):

    close = safe_num(latest["Close"])

    atr = safe_num(latest["ATR"], close * 0.02)

    support = safe_num(latest["Support"], close - atr)

    resistance = safe_num(latest["Resistance"], close + atr)



    if signal in ["Strong Buy", "Buy Signal"]:

        buy_low = max(support, close - 0.7 * atr)

        buy_high = min(close + 0.25 * atr, close * 1.015)

        target = max(resistance, close + 1.7 * atr)

        stop_loss = buy_low - 1.05 * atr

        action = "BUY SETUP"



    elif signal == "Buy on Dip":

        buy_low = max(support, close - 1.2 * atr)

        buy_high = close - 0.3 * atr

        target = close + 1.5 * atr

        stop_loss = buy_low - 1.0 * atr

        action = "BUY ON DIP"



    elif "Sell" in signal or "Avoid" in signal:

        buy_low = None

        buy_high = None

        target = min(support, close - 1.4 * atr)

        stop_loss = close + 1.0 * atr

        action = "AVOID / SELL SETUP"



    else:

        buy_low = close - 0.6 * atr

        buy_high = close + 0.2 * atr

        target = close + 1.1 * atr

        stop_loss = close - 1.0 * atr

        action = "WAIT / HOLD SETUP"



    expected_return_pct = expected_return * 100



    if buy_high is not None:

        trade_return_pct = ((target - buy_high) / buy_high) * 100

        downside_risk_pct = ((buy_high - stop_loss) / buy_high) * 100

    else:

        trade_return_pct = expected_return_pct

        downside_risk_pct = None



    if expected_return_pct < 0:

        final_decision = "WAIT / DO NOT BUY"

        reason = "Estimated return is negative."

    elif expected_return_pct < 1:

        final_decision = "WAIT"

        reason = "Expected return is too small."

    elif risk == "High":

        final_decision = "AVOID"

        reason = "Risk is high."

    elif "Avoid" in signal or "Sell" in signal:

        final_decision = "AVOID"

        reason = "Main signal is avoid/sell."

    elif confidence < 70:

        final_decision = "WAIT"

        reason = "Confidence is not strong enough."

    elif trade_return_pct < 2:

        final_decision = "WATCH ONLY"

        reason = "Trade return from buy zone to target is too small."

    elif downside_risk_pct is not None and downside_risk_pct > 5:

        final_decision = "WAIT"

        reason = "Downside risk is too high."

    elif signal in ["Strong Buy", "Buy Signal", "Buy on Dip"]:

        final_decision = "BUY ON DIP"

        reason = "Plan is valid, but entry should only happen inside the buy zone."

    else:

        final_decision = "WAIT"

        reason = "Setup is not strong enough."



    if horizon_days <= 5:

        hold = "1-5 days"

    elif horizon_days <= 14:

        hold = "5-14 days"

    elif horizon_days <= 60:

        hold = "2-8 weeks"

    else:

        hold = "2-6 months"



    return {

        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "Entry Price At Plan": round(close, 2),

        "Action": action,

        "Signal": signal,

        "Final Decision": final_decision,

        "Decision Reason": reason,

        "Buy Zone Low": round(buy_low, 2) if buy_low is not None else None,

        "Buy Zone High": round(buy_high, 2) if buy_high is not None else None,

        "Target": round(target, 2),

        "Stop Loss": round(stop_loss, 2),

        "Expected Hold": hold,

        "Confidence": confidence,

        "Estimated Return %": round(expected_return_pct, 2),

        "Trade Return From Buy Zone %": round(trade_return_pct, 2),

        "Downside Risk %": round(downside_risk_pct, 2) if downside_risk_pct is not None else None,

        "Risk": risk

    }





def live_plan_status(live_price, plan):

    final_decision = plan.get("Final Decision")

    buy_low = plan.get("Buy Zone Low")

    buy_high = plan.get("Buy Zone High")

    target = plan.get("Target")

    stop_loss = plan.get("Stop Loss")



    if final_decision in ["AVOID", "WAIT / DO NOT BUY"]:

        return final_decision, plan.get("Decision Reason")



    if buy_low is None or buy_high is None:

        return "NO ENTRY ZONE", "This plan does not have a valid buy zone."



    if live_price >= target:

        return "TARGET HIT", "Price reached or passed the target."



    if live_price <= stop_loss:

        return "STOP LOSS HIT", "Price reached or passed the stop loss."



    if buy_low <= live_price <= buy_high:

        return "ENTRY ZONE HIT", "Price is inside the locked buy zone."



    if live_price > buy_high:

        return "WAIT / DO NOT CHASE", "Price is above the locked buy zone. Do not chase."



    if live_price < buy_low:

        return "WAIT / BELOW ZONE", "Price is below the buy zone. Wait for confirmation."



    return "WAIT", "No action right now."





def make_price_chart(df, ticker, title_suffix="", plan=None):

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA9"], mode="lines", name="MA9"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["VWAP"], mode="lines", name="VWAP"))



    if plan is not None:

        if plan.get("Buy Zone Low") is not None:

            fig.add_hline(y=plan["Buy Zone Low"], line_dash="dot", annotation_text="Buy Zone Low")

        if plan.get("Buy Zone High") is not None:

            fig.add_hline(y=plan["Buy Zone High"], line_dash="dot", annotation_text="Buy Zone High")

        if plan.get("Target") is not None:

            fig.add_hline(y=plan["Target"], line_dash="dash", annotation_text="Target")

        if plan.get("Stop Loss") is not None:

            fig.add_hline(y=plan["Stop Loss"], line_dash="dash", annotation_text="Stop Loss")



    fig.update_layout(

        title=f"{ticker} Price Chart {title_suffix}",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig





def scanner_score(ticker, period, interval, market_score, market_label):

    df = load_price_data(ticker, period, interval)



    if df.empty:

        return None



    df = add_indicators(df)



    if df.empty:

        return None



    latest = df.iloc[-1]

    fundamentals = load_fundamentals(ticker)

    _, news_score, news_label = load_news_sentiment(ticker)

    _, expected_return, _ = estimate_future_price(df, 5)



    scores = score_stock(

        latest,

        fundamentals,

        expected_return,

        news_score,

        market_score,

        market_label

    )



    confidence = confidence_score(scores, expected_return, news_score)



    temp_plan = build_locked_plan(

        latest,

        scores["signal"],

        confidence,

        expected_return,

        5,

        scores["risk"]

    )



    return {

        "Ticker": ticker,

        "Price": round(safe_num(latest["Close"]), 2),

        "Signal": scores["signal"],

        "Final Decision": temp_plan["Final Decision"],

        "Decision Reason": temp_plan["Decision Reason"],

        "Confidence": confidence,

        "Risk": scores["risk"],

        "News": news_label,

        "Expected Return %": round(expected_return * 100, 2),

        "Total Score": scores["total_score"]

    }





tickers, ticker_df = get_all_tickers()

market_score, market_label = get_market_direction()



st.sidebar.header("Settings")



mode = st.sidebar.selectbox(

    "Trading mode",

    ["Intraday / Short-term", "Swing / Multi-day", "Historical / Long-term"],

    index=0

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



short_term_days = st.sidebar.selectbox("Short-term horizon", [1, 3, 5, 10, 14], index=2)

long_term_days = st.sidebar.selectbox("Long-term horizon", [30, 60, 90, 120, 180], index=1)



scan_count = st.sidebar.selectbox("How many stocks to scan?", [25, 50, 100, 200, 300, 500], index=1)

scan_count = min(scan_count, len(tickers))



st.sidebar.write(f"Available tickers loaded: {len(tickers)}")

st.sidebar.write(f"Market: {market_label}")

st.sidebar.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")



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



            short_estimate_df, short_expected_return, short_forecast_label = estimate_future_price(df, short_term_days)

            long_estimate_df, long_expected_return, long_forecast_label = estimate_future_price(df, long_term_days)



            latest = df.iloc[-1]

            live_price = safe_num(latest["Close"])



            short_scores = score_stock(latest, fundamentals, short_expected_return, news_score, market_score, market_label)

            long_scores = score_stock(latest, fundamentals, long_expected_return, news_score, market_score, market_label)



            short_confidence = confidence_score(short_scores, short_expected_return, news_score)

            long_confidence = confidence_score(long_scores, long_expected_return, news_score)



            plan_key_short = f"locked_short_plan_{selected_ticker}_{mode}_{short_term_days}"

            plan_key_long = f"locked_long_plan_{selected_ticker}_{mode}_{long_term_days}"



            st.subheader(f"{selected_ticker} Trading Plan")



            col_a, col_b, col_c = st.columns(3)



            with col_a:

                generate_short = st.button("Generate / Replace Short-Term Locked Plan")

            with col_b:

                generate_long = st.button("Generate / Replace Long-Term Locked Plan")

            with col_c:

                reset_plans = st.button("Reset Locked Plans")



            if reset_plans:

                if plan_key_short in st.session_state:

                    del st.session_state[plan_key_short]

                if plan_key_long in st.session_state:

                    del st.session_state[plan_key_long]

                st.success("Locked plans reset.")



            if generate_short or plan_key_short not in st.session_state:

                st.session_state[plan_key_short] = build_locked_plan(

                    latest,

                    short_scores["signal"],

                    short_confidence,

                    short_expected_return,

                    short_term_days,

                    short_scores["risk"]

                )



            if generate_long or plan_key_long not in st.session_state:

                st.session_state[plan_key_long] = build_locked_plan(

                    latest,

                    long_scores["signal"],

                    long_confidence,

                    long_expected_return,

                    long_term_days,

                    long_scores["risk"]

                )



            short_plan = st.session_state[plan_key_short]

            long_plan = st.session_state[plan_key_long]



            short_live_status, short_live_reason = live_plan_status(live_price, short_plan)

            long_live_status, long_live_reason = live_plan_status(live_price, long_plan)



            c1, c2, c3, c4, c5 = st.columns(5)



            c1.metric("Live / Latest Price", f"${live_price:.2f}")

            c2.metric("Short Signal", short_plan["Signal"])

            c3.metric("Short Locked Decision", short_plan["Final Decision"])

            c4.metric("Live Status", short_live_status)

            c5.metric("Market", market_label)



            st.markdown("### Short-Term Locked Plan")

            short_plan_display = pd.DataFrame([short_plan])

            short_plan_display["Live Price"] = round(live_price, 2)

            short_plan_display["Live Status"] = short_live_status

            short_plan_display["Live Status Reason"] = short_live_reason

            st.dataframe(short_plan_display, use_container_width=True)



            if short_live_status == "ENTRY ZONE HIT":

                st.success(short_live_reason)

            elif short_live_status in ["WAIT / DO NOT CHASE", "WAIT", "WAIT / BELOW ZONE"]:

                st.warning(short_live_reason)

            elif short_live_status in ["TARGET HIT"]:

                st.success(short_live_reason)

            elif short_live_status in ["STOP LOSS HIT", "AVOID", "WAIT / DO NOT BUY"]:

                st.error(short_live_reason)



            st.markdown("### Long-Term Locked Plan")

            long_plan_display = pd.DataFrame([long_plan])

            long_plan_display["Live Price"] = round(live_price, 2)

            long_plan_display["Live Status"] = long_live_status

            long_plan_display["Live Status Reason"] = long_live_reason

            st.dataframe(long_plan_display, use_container_width=True)



            st.info(

                f"Important: Live price updates every 60 seconds, but locked Buy Zone / Target / Stop Loss "

                f"do not change unless you click Generate / Replace Plan."

            )



            st.markdown("### Price Chart with Locked Plan Lines")

            st.plotly_chart(make_price_chart(df, selected_ticker, f"({mode})", short_plan), use_container_width=True)



            hist_df = load_price_data(selected_ticker, "5y", "1d")



            if not hist_df.empty:

                hist_df = add_indicators(hist_df)



                if not hist_df.empty:

                    st.markdown("### 5-Year Historical Chart")

                    st.plotly_chart(make_price_chart(hist_df, selected_ticker, "(5-Year Historical)", long_plan), use_container_width=True)



            st.markdown("### Estimated Future Prices")

            combined_estimate_df = pd.concat([short_estimate_df, long_estimate_df], ignore_index=True)

            st.dataframe(combined_estimate_df, use_container_width=True)



            st.markdown("### Score Breakdown")

            score_df = pd.DataFrame({

                "Category": [

                    "Short Technical", "Short Fundamental", "Short Forecast", "Short News", "Short Market", "Short Total",

                    "Long Technical", "Long Fundamental", "Long Forecast", "Long News", "Long Market", "Long Total"

                ],

                "Score": [

                    short_scores["technical_score"],

                    short_scores["fundamental_score"],

                    short_scores["forecast_score"],

                    short_scores["news_score_component"],

                    short_scores["market_score_component"],

                    short_scores["total_score"],

                    long_scores["technical_score"],

                    long_scores["fundamental_score"],

                    long_scores["forecast_score"],

                    long_scores["news_score_component"],

                    long_scores["market_score_component"],

                    long_scores["total_score"]

                ]

            })

            st.dataframe(score_df, use_container_width=True)



            st.markdown("### Why Short-Term Signal?")

            for r in short_scores["reasons"]:

                st.write(f"- {r}")



            st.markdown("### Why Long-Term Signal?")

            for r in long_scores["reasons"]:

                st.write(f"- {r}")



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

    st.write(f"Scanning first {scan_count} stocks. Market condition: {market_label}")



    rows = []

    progress = st.progress(0)



    for i, ticker in enumerate(tickers[:scan_count]):

        result = scanner_score(ticker, period, interval, market_score, market_label)



        if result is not None:

            rows.append(result)



        progress.progress((i + 1) / scan_count)



    if rows:

        scanner_df = pd.DataFrame(rows)



        order = {

            "BUY ON DIP": 1,

            "WATCH ONLY": 2,

            "WAIT": 3,

            "WAIT / DO NOT BUY": 4,

            "AVOID": 5

        }



        scanner_df["Sort"] = scanner_df["Final Decision"].map(order).fillna(9)



        scanner_df = scanner_df.sort_values(

            ["Sort", "Confidence", "Total Score"],

            ascending=[True, False, False]

        ).drop(columns=["Sort"])



        st.dataframe(scanner_df, use_container_width=True)



        st.download_button(

            "Download Scanner Results",

            scanner_df.to_csv(index=False).encode("utf-8"),

            "improved_stock_scanner_results.csv",

            "text/csv"

        )

    else:

        st.warning("No scanner results found.")





with tab3:

    st.subheader("Ticker List")

    st.write(f"Total tickers loaded: {len(ticker_df)}")

    st.dataframe(ticker_df, use_container_width=True)

