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

# ...（中間函數維持不變，和你原始代碼一樣）...

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

            # Layout: 左：圖，右：綜合解讀
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
