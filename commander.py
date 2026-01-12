import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================================
# 1. 規約設定（鋼の原則：主観を排除）
# ==========================================================
st.set_page_config(layout="wide", page_title="Universal Asset Commander")
st.title("🚀 超長期成績最大化：統合司令部 (Multi-Anchor Ver.)")

# サイドバー：環境認識
with st.sidebar:
    st.header("🌍 環境認識")
    cape_pct = st.slider("ワールドCAPE％タイル", 0.0, 1.0, 0.45)
    st.divider()
    if cape_pct < 0.50:
        st.success("推奨フェーズ: CLR (逆張り/平均回帰)")
        phase_mode = "CLR"
    else:
        st.warning("推奨フェーズ: CM (順張り/トレンド)")
        phase_mode = "CM"

# 全資産銘柄リスト
TICKER_MAP = {
    "N225": "1321.T", "TPX": "1306.T", "JREIT": "1343.T", 
    "GROW": "2516.T", "JDEF": "1399.T", "JVLU": "1593.T", "JQ": "2636.T"
}
TICKERS = [
    "SPY", "QQQ", "NOBL", "FDD", "VWO", "N225", "TPX", "GROW", 
    "JDEF", "VT", "VTV", "MTUM", "QUAL", "JVLU", "JQ", "FEZ", 
    "VNQI", "SCHD", "VYM", "JREIT", "GLD", "SLV", "TLT"
]

PROFIT_MARGINS = {
    "N225": 0.20, "TPX": 0.10, "FDD": 0.017, "JQ": 0.206, 
    "JREIT": 0.05, "TLT": 0.04
}

# ==========================================================
# 2. 関数：TSI Energy
# ==========================================================
def get_energy_status(prices):
    diff = prices.diff()
    ema_v = diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    a_ema_v = diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi = ema_v / a_ema_v
    signal = tsi.ewm(span=7).mean()
    return "OK" if tsi.iloc[-1] > signal.iloc[-1] else "DEAD"

# ==========================================================
# 3. データ処理
# ==========================================================
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
    
    # CLR (1年前アンカー)
    try:
        c_ref = p.iloc[-252]
        p_avg = (p.iloc[-252*3] + p.iloc[-252*4] + p.iloc[-252*5]) / 3
        clr_val = (c_ref / p_avg) - 1
    except: clr_val = 0
    
    # CM (1ヶ月前アンカー)
    try:
        c_m1 = p.iloc[-21]
        c_m4 = p.iloc[-21-63]
        cm_val = (c_m1 / c_m4) - 1
    except: cm_val = 0

    ret = p.pct_change().dropna()
    v_drag = ((ret.tail(252*3).std() * np.sqrt(252))**2) / 2
    energy_s = get_energy_status(p)
    ma200 = p.rolling(200).mean().iloc[-1]
    
    # 判定
    margin = PROFIT_MARGINS.get(t, 0)
    net_rg = clr_val - v_drag
    score_clr = net_rg + (margin * 0.20315) * max(0, net_rg)
    judge_clr = "🚀 FULL" if score_clr > 0.05 and energy_s == "OK" else "⏳ WAIT"
    clr_results.append({"Ticker": t, "Judge": judge_clr, "Score": score_clr, "Energy": energy_s})

    score_cm = cm_val - v_drag
    is_trend = p.iloc[-1] > ma200
    judge_cm = "🔥 FULL" if is_trend and energy_s == "OK" and score_cm > 0 else "🚨 EXIT"
    cm_results.append({"Ticker": t, "Judge": judge_cm, "Speed": score_cm, "Energy": energy_s, "MA200": "Above" if is_trend else "Below"})

# ==========================================================
# 4. 表示：テーブル
# ==========================================================
df_clr = pd.DataFrame(clr_results).set_index("Ticker").sort_values("Score", ascending=False)
df_cm = pd.DataFrame(cm_results).set_index("Ticker").sort_values("Speed", ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 CLR (1y Anchor)")
    st.dataframe(df_clr, height=400)
with col2:
    st.subheader("📈 CM (1m Anchor)")
    st.dataframe(df_cm, height=400)

# ==========================================================
# 5. 表示：3点プロット相対チャート
# ==========================================================
st.divider()
st.subheader("📊 アンカー別・相対パフォーマンス検証")
mode = st.radio("表示モード:", ["CLR基準 (1年前=0%)", "CM基準 (1ヶ月前=0%)"], horizontal=True)
anchor_days = 252 if "CLR" in mode else 21

selected = st.multiselect("銘柄選択:", TICKERS, default=df_clr.index[:3].tolist())

if selected:
    fig = go.Figure()
    for t in selected:
        p_series = data[t]
        ref = p_series.iloc[-anchor_days]
        rel = (p_series / ref - 1) * 100
        fig.add_trace(go.Scatter(x=p_series.index, y=rel, name=t, line=dict(width=2)))

        # 参照点プロット
        lookbacks = [252*3, 252*4, 252*5] if "CLR" in mode else [21*3, 21*6, 21*12]
        symbol = 'x' if "CLR" in mode else 'diamond'
        for lb in lookbacks:
            try:
                val = (p_series.iloc[-lb] / ref - 1) * 100
                fig.add_trace(go.Scatter(x=[p_series.index[-lb]], y=[val], mode='markers+text',
                                         text=[f"{val:.1f}%"], textposition="top center",
                                         marker=dict(size=8, symbol=symbol), showlegend=False))
            except: pass
        
        # 現在値ラベル
        now_val = rel.iloc[-1]
        fig.add_trace(go.Scatter(x=[p_series.index[-1]], y=[now_val], mode='markers+text',
                                 text=[f"NOW:{now_val:.1f}%"], textposition="middle right",
                                 marker=dict(size=10, color='white'), showlegend=False))

    fig.add_vline(x=data.index[-anchor_days], line_dash="dash", line_color="yellow")
    fig.add_hline(y=0, line_color="yellow", line_width=1)
    fig.update_layout(template="plotly_dark", height=600, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

st.caption("※黄色点線が起点。CLRは過去の『x』より現在が低いほどバネが強い。CMは過去の『◆』から右肩上がりならトレンド継続。")
