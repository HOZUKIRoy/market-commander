import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# ページ設定
st.set_page_config(layout="wide", page_title="Market Commander Alpha")
st.title("🏹 超長期成績最大化・運用司令室")

# --- 定数設定 ---
WORLD_CAPE_PCT = 0.89  
MACRO_FORCED_DEFENSE = True 
TICKER_MAP = {"N225": "1321.T", "TPX": "1306.T", "JREIT": "1343.T", "GROW": "2516.T", "JDEF": "1399.T", "JVLU": "1593.T", "JQ": "2636.T"}
SHORT_HAND_TICKERS = ["SPY", "QQQ", "NOBL", "FDD", "VWO", "GLD", "SLV", "TLT", "N225", "TPX", "GROW", "JDEF", "VT", "VTV", "MTUM", "QUAL", "JVLU", "JQ", "FEZ", "VNQI", "VYM", "SCHD"]
HIGH_BETA = ["SPY", "QQQ", "VTV", "MTUM", "FEZ", "VWO", "N225", "TPX", "GROW", "JVLU", "VNQ", "JREIT", "VNQI", "QUAL", "VT", "NOBL"]
PROFIT_MARGINS = {"FDD": 0.017, "JQ": 0.206, "N225": 0.20}

# --- ロジック関数 ---
def calculate_tsi_energy(series):
    diff = series.diff()
    ema_v = diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    a_ema_v = diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi = (ema_v / a_ema_v)
    energy = "OK" if tsi.iloc[-1] > tsi.ewm(span=7).mean().iloc[-1] else "DEAD"
    return energy

def get_cm_history(series):
    def calc_rg(off):
        idx = -1 - off
        try:
            return ((series.iloc[idx]/series.iloc[idx-63]-1) + (series.iloc[idx]/series.iloc[idx-126]-1) + (series.iloc[idx]/series.iloc[idx-252]-1))/3
        except: return 0.0
    return calc_rg(0), calc_rg(21), calc_rg(63)

# --- データ取得 ---
@st.cache_data(ttl=3600)
def load_data():
    tickers = [TICKER_MAP.get(t, t) for t in SHORT_HAND_TICKERS]
    data = yf.download(tickers, start=datetime.now()-timedelta(days=730), progress=False, auto_adjust=True)['Close']
    return data.rename(columns={v: k for k, v in TICKER_MAP.items()}).ffill()

df_price = load_data()

# --- 計算実行 ---
results = []
for t in SHORT_HAND_TICKERS:
    if t not in df_price.columns: continue
    p = df_price[t]
    energy = calculate_tsi_energy(p)
    rg_now, rg_1m, rg_3m = get_cm_history(p)
    ma200 = p.rolling(200).mean().iloc[-1]
    ret = p.pct_change().dropna()
    sd_all = ret.tail(252).std() * np.sqrt(252)
    sd_d = max(ret[ret<0].tail(252).std() * np.sqrt(252), 1e-6)
    v_drag = (sd_all ** 2) / 2
    net_rg = rg_now - v_drag
    geta = (PROFIT_MARGINS.get(t, 0) * 0.20315) * max(0, net_rg)
    score = net_rg + geta
    bias_200 = ((p.iloc[-1] / ma200) - 1) * 100
    m1_s, m3_s, ma_s = ("U" if rg_now > rg_1m else "D"), ("U" if rg_now > rg_3m else "D"), ("U" if p.iloc[-1] > ma200 else "D")

    # 判定
    if score < 0: d_judge = "🚨 SELL(Score<0)"
    elif ma_s == "D": d_judge = "⚠️ EXIT(MA200 Down)"
    elif energy == "DEAD" and (m1_s == "D" or m3_s == "D"): d_judge = "📉 REDUCE(50%)"
    else: d_judge = "✅ KEEP"

    results.append({"Ticker": t, "Judge": d_judge, "Score": score, "NetRG": net_rg, "CM": rg_now, "Eng/1/3": f"{energy}/{m1_s}/{m3_s}", "Bias200": bias_200, "SD_D": sd_d})

res_df = pd.DataFrame(results).set_index("Ticker")

# --- ブラウザ画面表示 ---
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📋 執行判定リスト")
    st.dataframe(res_df[["Judge", "Score", "Eng/1/3", "Bias200"]].sort_values("Score", ascending=False), height=450)
with col2:
    st.subheader("🛣️ 200MA乖離率 (場所の判定)")
    fig_bias = px.bar(res_df.sort_values("Bias200"), x="Bias200", y=res_df.sort_values("Bias200").index, orientation='h', color="Bias200", color_continuous_scale="RdBu", range_color=[-15, 15])
    st.plotly_chart(fig_bias, use_container_width=True)

st.subheader("🎯 戦略マップ (縦:実力 / 横:リスク / 色:期待値)")
fig_map = px.scatter(res_df.reset_index(), x="SD_D", y="NetRG", color="Score", text="Ticker", size=[20]*len(res_df), color_continuous_scale="RdYlGn", hover_data=["Eng/1/3", "Judge", "Bias200"])
fig_map.update_traces(textposition='top center')
st.plotly_chart(fig_map, use_container_width=True)