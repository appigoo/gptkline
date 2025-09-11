import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Optional candlestick (if installed)
try:
    import mplfinance as mpf
    HAS_MPF = True
except Exception:
    HAS_MPF = False

st.set_page_config(page_title="自動日報表 (K線 + 技術解讀)", layout="wide")

# ----------------- helper functions -----------------
@st.cache_data(ttl=60*30)
def download_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    """下載資料並做基本清理"""
    df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
    if df.empty:
        return df
    # 優先使用 Adjusted Close 做指標計算（若有）
    if 'Adj Close' in df.columns:
        df['Close_for_calc'] = df['Adj Close']
    else:
        df['Close_for_calc'] = df['Close']
    df.dropna(inplace=True)
    return df

def compute_indicators(df: pd.DataFrame):
    c = df['Close_for_calc']
    df['EMA10'] = c.ewm(span=10, adjust=False).mean()
    df['EMA30'] = c.ewm(span=30, adjust=False).mean()
    df['EMA40'] = c.ewm(span=40, adjust=False).mean()

    # MACD
    exp12 = c.ewm(span=12, adjust=False).mean()
    exp26 = c.ewm(span=26, adjust=False).mean()
    df['DIF'] = exp12 - exp26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['DIF'] - df['DEA']

    # 成交量平均
    df['VOL_MA20'] = df['Volume'].rolling(window=20).mean()
    return df

def classify_candle(o, h, l, c):
    body = abs(c - o)
    full = h - l + 1e-9
    upper = h - max(c, o)
    lower = min(c, o) - l
    body_ratio = body / full if full > 0 else 0

    if body_ratio < 0.15:
        typ = "十字/小實體（盤整）"
    else:
        typ = "陽線（紅K）" if c > o else "陰線（綠K）"

    if lower > body * 2 and lower > upper:
        typ += "，長下影（可能支撐或吸籌）"
    elif upper > body * 2 and upper > lower:
        typ += "，長上影（可能壓力或獲利了結）"
    elif body_ratio > 0.7:
        typ += "，實體長（趨勢強烈）"

    return typ

def single_candle_interpret(row, ema10, ema30, vol_ma20):
    o, h, l, c, v = row['Open'], row['High'], row['Low'], row['Close_for_calc'], row['Volume']
    typ = classify_candle(o, h, l, c)

    if c > ema10:
        pos = "收在 EMA10 之上（短線偏多）"
    elif c > ema30:
        pos = "跌破 EMA10 但守住 EMA30（短線整理）"
    else:
        pos = "跌破 EMA30（短期偏弱）"

    if np.isnan(vol_ma20) or vol_ma20 == 0:
        vol_note = "成交量資料不足"
    else:
        ratio = v / vol_ma20
        if ratio > 1.5:
            vol_note = f"量大（{ratio:.2f}x 近20日均量）"
        elif ratio < 0.7:
            vol_note = f"量縮（{ratio:.2f}x 近20日均量）"
        else:
            vol_note = f"量正常（{ratio:.2f}x 近20日均量）"
    return f"{typ}；{pos}；{vol_note}"

def overall_trend_text(df):
    last = df.iloc[-1]
    ema10 = float(last['EMA10'])
    ema30 = float(last['EMA30'])
    ema40 = float(last['EMA40'])

    slope10 = float(ema10 - df['EMA10'].iloc[-5]) if len(df) >= 6 else float(ema10 - df['EMA10'].iloc[0])
    slope30 = float(ema30 - df['EMA30'].iloc[-5]) if len(df) >= 6 else float(ema30 - df['EMA30'].iloc[0])

    if ema10 > ema30 > ema40 and slope10 > 0 and slope30 > 0:
        return "中期趨勢仍偏多（均線多頭排列且向上）"
    elif ema10 < ema30 < ema40 and slope10 < 0:
        return "中期偏空（均線空頭排列）"
    else:
        return "趨勢分歧或整理（需關注均線與量能）"

def macd_status(df):
    dif = float(df['DIF'].iloc[-1])
    dea = float(df['DEA'].iloc[-1])
    hist = float(df['MACD_hist'].iloc[-1])
    hist_trend = "上升" if (len(df)>=3 and df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-3]) else "下降或收斂"
    cross = ""
    if dif > dea and df['DIF'].iloc[-2] <= df['DEA'].iloc[-2]:
        cross = "（近期出現 MACD 黃金交叉）"
    elif dif < dea and df['DIF'].iloc[-2] >= df['DEA'].iloc[-2]:
        cross = "（近期出現 MACD 死亡交叉）"
    return f"DIF={dif:.3f}, DEA={dea:.3f}, MACD_hist={hist:.3f}；柱狀態趨勢：{hist_trend} {cross}"

def important_levels(df):
    last = df.iloc[-1]
    return {
        "EMA10": float(last['EMA10']),
        "EMA30": float(last['EMA30']),
        "EMA40": float(last['EMA40']),
        "recent_high_20": float(df['High'].rolling(window=20, min_periods=1).max().iloc[-1]),
        "recent_low_20": float(df['Low'].rolling(window=20, min_periods=1).min().iloc[-1])
    }

def generate_detailed_report(df, ticker):
    last5 = df.tail(5)
    last = df.iloc[-1]

    quick = overall_trend_text(df)
    macd = macd_status(df)
    vol_now = int(last['Volume'])
    vol_ma20 = last['VOL_MA20']
    vol_note = "量不足" if np.isnan(vol_ma20) else f"當日量 {vol_now:.0f}，20日均量 {vol_ma20:.0f}（比率 {vol_now/vol_ma20:.2f}x）"

    rows = []
    for idx, row in last5.iterrows():
        rows.append({
            "日期": idx.strftime("%Y-%m-%d"),
            "開": round(row['Open'],2),
            "高": round(row['High'],2),
            "低": round(row['Low'],2),
            "收": round(row['Close_for_calc'],2),
            "成交量": int(row['Volume']),
            "單根解讀": single_candle_interpret(row, row['EMA10'], row['EMA30'], row['VOL_MA20'])
        })
    per_candle_df = pd.DataFrame(rows)

    lv = important_levels(df)
    levels_text = (
        f"EMA10={lv['EMA10']:.3f}，EMA30={lv['EMA30']:.3f}，EMA40={lv['EMA40']:.3f}。\n"
        f"最近20日高點 {lv['recent_high_20']:.2f}，低點 {lv['recent_low_20']:.2f}。"
    )

    price_pos = ("收在 EMA10 之上" if last['Close_for_calc'] > lv['EMA10']
                 else "收在 EMA10 與 EMA30 之間" if last['Close_for_calc'] > lv['EMA30']
                 else "已跌破 EMA30")

    macd_hint = "動能仍正但柱體縮小（需留意動能是否繼續衰竭）" if df['MACD_hist'].iloc[-1] > 0 and len(df)>=3 and df['MACD_hist'].iloc[-1] < df['MACD_hist'].iloc[-3] else \
               ("動能擴張（上攻續有機會）" if len(df)>=3 and df['MACD_hist'].iloc[-1] > df['MACD_hist'].iloc[-3] else "動能偏弱或收斂")

    composite = (
        f"目前價格 {price_pos}；{macd_hint}。異動量能：{vol_note}。\n"
        "綜合來看：\n"
        "- 中期趨勢：" + quick + "\n"
        f"- MACD 與動能狀態：{macd}\n"
        f"- 重要價位：{levels_text}\n"
    )

    advice = [
        "持有者：若你屬保守型，建議把停損放在 EMA30（或略下方）；若偏積極，可保留核心倉並在價格回測 EMA10 或 EMA30 並確認止跌後分批加碼。",
        "新多單：目前不建議追高。較安全做法為等回測到 EMA10（觀察是否量縮並帶下影）或等價格突破且站穩近期高點並放量後再追。",
        "短線/做空者：當日若出現放量跌破 EMA30 且 MACD 柱轉負，可考慮短線做空（嚴設停損，目標先看 EMA40 / 最近低點）。",
        "風險控管：控倉、分批與嚴格停損是關鍵；留意成交量是否放大配合方向，以免被套在高位。"
    ]
    advice_text = "\n".join([f"- {a}" for a in advice])

    return {
        "ticker": ticker,
        "quick_summary": quick,
        "indicators_text": macd,
        "volume_note": vol_note,
        "levels_text": levels_text,
        "composite_text": composite,
        "advice_text": advice_text,
        "per_candle_df": per_candle_df
    }

# ----------------- Streamlit UI -----------------
st.title("📈 自動日報表：最後 5 日K + 詳細綜合解讀")
with st.sidebar:
    st.header("設定")
    ticker = st.text_input("股票代號（例：TSLA、AAPL、0700.HK）", value="TSLA")
    period = st.selectbox("歷史資料區間", options=["1mo","3mo","6mo","1y","2y"], index=2)
    interval = "1d"
    show_candlestick = st.checkbox("顯示 K 線圖（需要 mplfinance）", value=False)
    run_button = st.button("生成報表")

if run_button:
    with st.spinner("下載資料並計算指標..."):
        df = download_data(ticker, period=period, interval=interval)
        if df.empty:
            st.error("找不到資料或下載失敗，請確認股票代號或網路連線。")
        else:
            df = compute_indicators(df)
            if len(df) < 20:
                st.warning("資料筆數較少（<20），部分指標可能不足或不精準。")

            report = generate_detailed_report(df, ticker)

            col1, col2 = st.columns([2,3])

            with col1:
                st.subheader("走勢圖與均線")
                if show_candlestick and HAS_MPF:
                    mpf_df = df[['Open','High','Low','Close_for_calc','Volume']].copy()
                    mpf_df.rename(columns={'Close_for_calc':'Close'}, inplace=True)
                    addplots = [
                        mpf.make_addplot(df['EMA10'], color='orange'),
                        mpf.make_addplot(df['EMA30'], color='red'),
                        mpf.make_addplot(df['EMA40'], color='blue')
                    ]
                    fig, axlist = mpf.plot(
                        mpf_df, type='candle', style='charles',
                        addplot=addplots, volume=True, returnfig=True, figsize=(10,6)
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), gridspec_kw={'height_ratios':[3,1]})
                    ax1.plot(df.index, df['Close_for_calc'], label='收盤價', linewidth=1)
                    ax1.plot(df.index, df['EMA10'], label='EMA10', linewidth=1)
                    ax1.plot(df.index, df['EMA30'], label='EMA30', linewidth=1)
                    ax1.plot(df.index, df['EMA40'], label='EMA40', linewidth=1)
                    ax1.legend(loc='upper left')
                    ax1.set_title(f"{ticker} 收盤價與EMA")
                    ax2.bar(df.index, df['Volume'])
                    ax2.set_title("成交量")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                csv_bytes = report['per_candle_df'].to_csv(index=False).encode('utf-8')
                st.download_button("下載最近5根K之表格（CSV）", data=csv_bytes, file_name=f"{ticker}_last5.csv", mime="text/csv")

            with col2:
                st.subheader("🔎 詳細綜合解讀")
                st.markdown(f"**快速結論：** {report['quick_summary']}")
                st.markdown(f"**指標摘要：** {report['indicators_text']}")
                st.markdown(f"**量能觀察：** {report['volume_note']}")
                st.markdown(f"**重要價位：** {report['levels_text']}")
                st.markdown("**綜合說明：**")
                st.write(report['composite_text'])
                st.markdown("**具體建議（持有 / 新多 / 短線）**")
                st.write(report['advice_text'])

            st.subheader("最近 5 根 K 線逐根解讀")
            st.dataframe(report['per_candle_df'])

            st.info("提示：程式使用 'Adj Close'（若有）做指標計算。請記得將停損與倉位依照你的風險承受度調整。")
            st.success("報表產生完成 ✅")

# Footer
st.markdown("---")
st.caption("Powered by yfinance + Streamlit")
