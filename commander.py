import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================================
# 1. 規約設定
# ==========================================================
st.set_page_config(layout="wide", page_title="Universal Asset Commander")
st.title("🚀 超長期成績最大化：統合司令部 (Full Indicators Ver.)")

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

# --- 共通指標計算 ---
rets = data.pct_change()
# Downside Deviation (σ-down) の計算
def get_sigma_down(series):
    downside_rets = series[series < 0]
    return downside_rets.std() * np.sqrt(252)

# VTの基準値取得
vt_p = data["VT"]
vt_cm_raw = (vt_p.iloc[-21] / vt_p.iloc[-21-63]) - 1
vt_sigma_all = rets["VT"].std() * np.sqrt(252)
vt_cm_norm = vt_cm_raw / vt_sigma_all if vt_sigma_all != 0 else 0

clr_results, cm_results = [], []

for t in TICKERS:
    if t not in data.columns: continue
    p = data[t]
    
    # 指標計算
    c_ref = p.iloc[-252]
    p_avg = (p.iloc[-252*3] + p.iloc[-252*4] + p.iloc[-252*5]) / 3
    clr_val = (c_ref / p_avg) - 1
    c_m1 = p.iloc[-21]
    c_m4 = p.iloc[-21-63]
    cm_val = (c_m1 / c_m4) - 1
    
    # 200MA乖離
    ma200 = p.rolling(200).mean().iloc[-1]
    dev_ma = (p.iloc[-1] / ma200 - 1) * 100
    
    # リスク調整
    v_drag = (rets[t].tail(252*3).std() * np.sqrt(252))**2 / 2
    s_down = get_sigma_down(rets[t].tail(252*3))
    energy_s = get_energy_status(p)
    
    # CLRスコア
    score_clr = (clr_val - v_drag) + (PROFIT_MARGINS.get(t, 0) * 0.20315) * max(0, clr_val - v_drag)
    clr_results.append({
        "Ticker": t, "Judge": "🚀 FULL" if score_clr > 0.05 and energy_s == "OK" else "⏳ WAIT",
        "Score": round(score_clr, 4), "MA-Dev%": round(dev_ma, 2), "Energy": energy_s
    })

    # CMスコア正規化
    cm_norm_down = (cm_val - v_drag) / s_down if s_down != 0 else 0
    cm_results.append({
        "Ticker": t, "Judge": "🔥 FULL" if p.iloc[-1] > ma200 and energy_s == "OK" and cm_val > 0 else "🚨 EXIT",
        "Speed": round(cm_val, 4), "σ-down": round(cm_norm_down, 2), "MA-Dev%": round(dev_ma, 2), "Energy": energy_s
    })

# ==========================================================
# 3. 表示セクション
# ==========================================================
st.sidebar.metric("🌍 VT CM/σ-all", round(vt_cm_norm, 3))

df_clr = pd.DataFrame(clr_results).set_index("Ticker").sort_values("Score", ascending=False)
df_cm = pd.DataFrame(cm_results).set_index("Ticker").sort_values("Speed", ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 CLR (Value/Anchor)")
    st.dataframe(df_clr)
with col2:
    st.subheader("📈 CM (Momentum/σ-down)")
    st.dataframe(df_cm)

# --- チャート (判断ポイント復元) ---
st.divider()
mode = st.radio("表示:", ["CLR基準 (1年前=0% / 6y)", "CM基準 (1ヶ月前=0% / 1.5y)"], horizontal=True)
selected = st.multiselect("銘柄選択:", TICKERS, default=["SPY", "VT", "GLD"])

if selected:
    fig = go.Figure()
    anchor_days = 252 if "CLR" in mode else 21
    start_view = data.index[0] if "CLR" in mode else data.index[-378]

    for t in selected:
        p_series = data[t]
        ref = p_series.iloc[-anchor_days]
        rel = (p_series / ref - 1) * 100
        fig.add_trace(go.Scatter(x=p_series.index, y=rel, name=t))

        # 参照点 (判断の根拠)
        lookbacks = [252*3, 252*4, 252*5] if "CLR" in mode else [21*3, 21*6, 21*12]
        sym = 'x' if "CLR" in mode else 'diamond'
        for lb in lookbacks:
            try:
                v = (p_series.iloc[-lb] / ref - 1) * 100
                fig.add_trace(go.Scatter(x=[p_series.index[-lb]], y=[v], mode='markers+text',
                                         text=[f"{v:.1f}%"], textposition="top center",
                                         marker=dict(size=8, symbol=sym), showlegend=False))
            except: pass
        
        # 現在値
        now_v = rel.iloc[-1]
        fig.add_trace(go.Scatter(x=[p_series.index[-1]], y=[now_v], mode='markers+text',
                                 text=[f"NOW:{now_v:.1f}%"], textposition="middle right",
                                 marker=dict(size=10, color='white'), showlegend=False))

    fig.add_vline(x=data.index[-anchor_days], line_dash="dash", line_color="yellow")
    fig.add_hline(y=0, line_color="yellow")
    fig.update_xaxes(range=[start_view, data.index[-1]])
    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"VT CM/σ-all: {vt_cm_norm:.3f} | CM/σ-downは負のボラティリティに対するリターン効率。MA-Dev%がマイナスかつScoreプラスがCLRの好機。")
