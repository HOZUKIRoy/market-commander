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
st.title("🚀 超長期成績最大化：全資産統合司令部")

# サイドバー：環境認識（CAPEによるフェーズ決定）
with st.sidebar:
    st.header("🌍 環境認識")
    cape_pct = st.slider("ワールドCAPE％タイル", 0.0, 1.0, 0.45)
    st.divider()
    if cape_pct < 0.50:
        st.success("推奨フェーズ: CLR (逆張り/平均回帰)")
        phase_goal = "安値（エッジ）の最大化"
    else:
        st.warning("推奨フェーズ: CM (順張り/トレンド)")
        phase_goal = "勢いへの便乗と逃げ足の速さ"
    st.info(f"目的: {phase_goal}")

# 全資産銘柄リスト（株式、REIT、コモディティ、債券）
TICKER_MAP = {
    "N225": "1321.T", "TPX": "1306.T", "JREIT": "1343.T", 
    "GROW": "2516.T", "JDEF": "1399.T", "JVLU": "1593.T", "JQ": "2636.T"
}
TICKERS = [
    "SPY", "QQQ", "NOBL", "FDD", "VWO", "N225", "TPX", "GROW", 
    "JDEF", "VT", "VTV", "MTUM", "QUAL", "JVLU", "JQ", "FEZ", 
    "VNQI", "SCHD", "VYM", "JREIT", "GLD", "SLV", "TLT"
]

# 期待値補正（利回り・利益率）
PROFIT_MARGINS = {
    "N225": 0.20, "TPX": 0.10, "FDD": 0.017, "JQ": 0.206, 
    "JREIT": 0.05, "TLT": 0.04  # 債券は利回りを計上、金銀は0
}

# ==========================================================
# 2. 関数：TSI Energy (最速トリガー)
# ==========================================================
def get_energy_status(prices):
    diff = prices.diff()
    # 2段平滑化による売り枯れ・反転の検知
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
clr_results = []
cm_results = []

for t in TICKERS:
    if t not in data.columns: continue
    p = data[t]
    
    # --- CLR (1年前基準) ---
    try:
        c_ref = p.iloc[-252]
        p_avg = (p.iloc[-252*3] + p.iloc[-252*4] + p.iloc[-252*5]) / 3
        clr_val = (c_ref / p_avg) - 1
    except: clr_val = 0
    
    # --- CM (1ヶ月前基準) ---
    try:
        c_m1 = p.iloc[-21]
        c_m4 = p.iloc[-21-63]
        cm_val = (c_m1 / c_m4) - 1
    except: cm_val = 0

    # 共通計算
    ret = p.pct_change().dropna()
    v_drag = ((ret.tail(252*3).std() * np.sqrt(252))**2) / 2
    energy_s = get_energy_status(p)
    ma200 = p.rolling(200).mean().iloc[-1]
    p_now = p.iloc[-1]

    # CLR判定 (0 or 100)
    margin = PROFIT_MARGINS.get(t, 0)
    net_rg = clr_val - v_drag
    score_clr = net_rg + (margin * 0.20315) * max(0, net_rg)
    judge_clr = "🚀 FULL" if score_clr > 0.05 and energy_s == "OK" else "⏳ WAIT"
    clr_results.append({"Ticker": t, "Judge": judge_clr, "Score": score_clr, "Energy": energy_s, "V-Drag": v_drag})

    # CM判定 (0 or 100)
    score_cm = cm_val - v_drag
    is_trend = p_now > ma200
    judge_cm = "🔥 FULL" if is_trend and energy_s == "OK" and score_cm > 0 else "🚨 EXIT"
    cm_results.append({"Ticker": t, "Judge": judge_cm, "Speed": score_cm, "Energy": energy_s, "MA200": "Above" if is_trend else "Below"})

# ==========================================================
# 4. 表示セクション
# ==========================================================
df_clr = pd.DataFrame(clr_results).set_index("Ticker").sort_values("Score", ascending=False)
df_cm = pd.DataFrame(cm_results).set_index("Ticker").sort_values("Speed", ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 CLR (Value/Reversal)")
    if cape_pct < 0.50: st.caption("✅ CURRENT RECOMMENDED")
    st.dataframe(df_clr, height=450)

with col2:
    st.subheader("📈 CM (Momentum/Trend)")
    if cape_pct >= 0.50: st.caption("✅ CURRENT RECOMMENDED")
    st.dataframe(df_cm, height=450)

# ==========================================================
# 5. 視覚分析：0基準相対チャート
# ==========================================================
st.divider()
st.subheader("📊 0基準・相対パフォーマンス比較")
mode = st.radio("基準（アンカー）:", ["CLR (1年前)", "CM (1ヶ月前)"], horizontal=True)
anchor_days = 252 if "CLR" in mode else 21

selected = st.multiselect("銘柄選択:", TICKERS, default=df_clr.index[:5].tolist())

if selected:
    fig = go.Figure()
    for t in selected:
        if t not in data.columns: continue
        p_series = data[t]
        ref = p_series.iloc[-anchor_days]
        rel = (p_series / ref - 1) * 100
        fig.add_trace(go.Scatter(x=p_series.index, y=rel, name=t))
        # 200MA相対表示（点線）
        ma_rel = (p_series.rolling(200).mean() / ref - 1) * 100
        fig.add_trace(go.Scatter(x=p_series.index, y=ma_rel, name=f"{t}(200MA)", 
                                 line=dict(dash='dot', width=1), visible='legendonly'))

    fig.add_hline(y=0, line_color="white", line_width=2)
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.write("※規約：CAPE 50%を境界に、推奨される表の『FULL』銘柄へ資金を100%割り当てる。妥協（半分）はしない。")
