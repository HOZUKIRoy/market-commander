import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==========================================================
# 1. 規約設定（鋼の原則）
# ==========================================================
st.set_page_config(layout="wide", page_title="Universal Asset Commander")
st.title("🚀 超長期成績最大化：統合司令部 (MA-Deviation Ver.)")

with st.sidebar:
    st.header("🌍 環境認識")
    cape_pct = st.slider("ワールドCAPE％タイル", 0.0, 1.0, 0.45)
    st.divider()
    if cape_pct < 0.50:
        st.success("推奨フェーズ: CLR (逆張り/平均回帰)")
    else:
        st.warning("推奨フェーズ: CM (順張り/トレンド)")

TICKER_MAP = {"N225": "1321.T", "TPX": "1306.T", "JREIT": "1343.T", "GROW": "2516.T", "JDEF": "1399.T", "JVLU": "1593.T", "JQ": "2636.T"}
TICKERS = ["SPY", "QQQ", "NOBL", "FDD", "VWO", "N225", "TPX", "GROW", "JDEF", "VT", "VTV", "MTUM", "QUAL", "JVLU", "JQ", "FEZ", "VNQI", "SCHD", "VYM", "JREIT", "GLD", "SLV", "TLT"]
PROFIT_MARGINS = {"N225": 0.20, "TPX": 0.10, "FDD": 0.017, "JQ": 0.206, "JREIT": 0.05, "TLT": 0.04}

# ==========================================================
# 2. 関数・データ処理
# ==========================================================
def get_energy_status(prices):
    diff = prices.diff()
    ema_v = diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    a_ema_v = diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi = ema_v / a_ema_v
    signal = tsi.ewm(span=7).mean()
    return "OK" if tsi.iloc[-1] > signal.iloc[-1] else "DEAD"

@st.cache_data(ttl=3600)
def load_data(ticker_list):
    end = datetime.now()
    start = end - timedelta(days=365*6)
    symbols = [TICKER_MAP.get(t, t) for t in ticker_list]
    data = yf.download(symbols, start=start, progress=False, auto_adjust=True)['Close']
    return data.rename(columns={v: k for k, v in TICKER_MAP.items()}).ffill()

data = load_data(TICKERS)
clr_results, cm_results = [], []

for t in TICKERS:
    if t not in data.columns: continue
    p = data[t]
    c_ref = p.iloc[-252]
    p_avg = (p.iloc[-252*3] + p.iloc[-252*4] + p.iloc[-252*5]) / 3
    clr_val = (c_ref / p_avg) - 1
    c_m1 = p.iloc[-21]
    c_m4 = p.iloc[-21-63]
    cm_val = (c_m1 / c_m4) - 1
    ret = p.pct_change().dropna()
    v_drag = ((ret.tail(252*3).std() * np.sqrt(252))**2) / 2
    energy_s = get_energy_status(p)
    ma200_val = p.rolling(200).mean().iloc[-1]
    
    score_clr = (clr_val - v_drag) + (PROFIT_MARGINS.get(t, 0) * 0.20315) * max(0, clr_val - v_drag)
    clr_results.append({"Ticker": t, "Judge": "🚀 FULL" if score_clr > 0.05 and energy_s == "OK" else "⏳ WAIT", "Score": score_clr, "Energy": energy_s})

    score_cm = cm_val - v_drag
    judge_cm = "🔥 FULL" if p.iloc[-1] > ma200_val and energy_s == "OK" and score_cm > 0 else "🚨 EXIT"
    cm_results.append({"Ticker": t, "Judge": judge_cm, "Speed": score_cm, "Energy": energy_s, "MA200": "Above" if p.iloc[-1] > ma200_val else "Below"})

# ==========================================================
# 3. グラフ・表示セクション
# ==========================================================
df_clr = pd.DataFrame(clr_results).set_index("Ticker").sort_values("Score", ascending=False)
df_cm = pd.DataFrame(cm_results).set_index("Ticker").sort_values("Speed", ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 CLR (Value)")
    st.dataframe(df_clr, height=350)
with col2:
    st.subheader("📈 CM (Momentum)")
    st.dataframe(df_cm, height=350)

st.divider()
mode = st.radio("分析モード:", ["CLR基準 (1年前=0%)", "CM基準 (1ヶ月前=0%)"], horizontal=True)
selected = st.multiselect("銘柄選択:", TICKERS, default=df_clr.index[:3].tolist())

if selected:
    # サブプロット作成 (上: 相対価格, 下: MA200乖離率)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    anchor_days = 252 if "CLR" in mode else 21
    start_view = data.index[0] if "CLR" in mode else data.index[-378]

    for t in selected:
        p_series = data[t]
        ref = p_series.iloc[-anchor_days]
        # 1. 相対価格チャート
        rel = (p_series / ref - 1) * 100
        fig.add_trace(go.Scatter(x=p_series.index, y=rel, name=f"{t} Rel", line=dict(width=2)), row=1, col=1)
        
        # 2. MA200乖離率チャート
        ma200 = p_series.rolling(200).mean()
        deviation = (p_series / ma200 - 1) * 100
        fig.add_trace(go.Scatter(x=p_series.index, y=deviation, name=f"{t} MA-Dev", line=dict(dash='dot')), row=2, col=1)

    fig.add_hline(y=0, line_color="yellow", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_color="white", line_dash="dash", row=2, col=1)
    fig.update_xaxes(range=[start_view, data.index[-1]])
    fig.update_layout(template="plotly_dark", height=800, hovermode="x unified", title="相対価格(上) と 200MA乖離率(下)")
    st.plotly_chart(fig, use_container_width=True)

st.caption("※下のグラフがマイナス圏で深ければ深いほど、地盤から乖離した『逆張りの期待値』が高いことを示します。")
