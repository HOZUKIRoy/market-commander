import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ==========================================================
# 1. 規約設定（鋼の原則）
# ==========================================================
st.set_page_config(layout="wide", page_title="Master Strategy Terminal")
st.title("🚀 超長期成績最大化：統合司令部")

# サイドバー：環境認識
with st.sidebar:
    st.header("🌍 環境認識")
    cape_pct = st.slider("ワールドCAPE％タイル", 0.0, 1.0, 0.45) # ここでフェーズが決まる
    st.divider()
    if cape_pct < 0.50:
        st.success("推奨フェーズ: CLR (逆張り/平均回帰)")
        st.info("理由: 割安圏では『安さ』というエッジがトレンドを上回るため。")
    else:
        st.warning("推奨フェーズ: CM (順張り/トレンド)")
        st.info("理由: 割高圏では『勢い』のみがリスクを上回るため。")

# 銘柄リスト
TICKER_MAP = {"N225": "1321.T", "TPX": "1306.T", "JREIT": "1343.T", "GROW": "2516.T", "JDEF": "1399.T", "JVLU": "1593.T", "JQ": "2636.T"}
TICKERS = ["SPY", "QQQ", "NOBL", "FDD", "VWO", "N225", "TPX", "GROW", "JDEF", "VT", "VTV", "MTUM", "QUAL", "JVLU", "JQ", "FEZ", "VNQI", "SCHD", "VYM", "JREIT"]
PROFIT_MARGINS = {"N225": 0.20, "TPX": 0.10, "FDD": 0.017, "JQ": 0.206, "JREIT": 0.05}

# ==========================================================
# 2. 関数：TSI Energy (最速トリガー)
# ==========================================================
def get_energy_status(prices):
    diff = prices.diff()
    ema_v = diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    a_ema_v = diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi = ema_v / a_ema_v
    signal = tsi.ewm(span=7).mean()
    return "OK" if tsi.iloc[-1] > signal.iloc[-1] else "DEAD", tsi.iloc[-1]

# ==========================================================
# 3. データ取得 & 演算
# ==========================================================
@st.cache_data(ttl=3600)
def load_data(ticker_list):
    end = datetime.now()
    start = end - timedelta(days=365*6)
    data = yf.download([TICKER_MAP.get(t, t) for t in ticker_list], start=start, progress=False, auto_adjust=True)['Close']
    return data.rename(columns={v: k for k, v in TICKER_MAP.items()}).ffill()

data = load_data(TICKERS)

clr_results = []
cm_results = []

for t in TICKERS:
    if t not in data.columns: continue
    p = data[t]
    p_now = p.iloc[-1]
    
    # --- CLRロジック (1年前アンカー) ---
    try:
        c_ref = p.iloc[-252]
        p3y, p4y, p5y = p.iloc[-252*3], p.iloc[-252*4], p.iloc[-252*5]
        clr_val = ((c_ref/p3y-1) + (c_ref/p4y-1) + (c_ref/p5y-1)) / 3
    except: clr_val = 0
    
    # --- CMロジック (1ヶ月前アンカー) ---
    try:
        c_m1 = p.iloc[-21] # 1ヶ月前
        c_m4 = p.iloc[-21-63] # 4ヶ月前
        cm_val = (c_m1 / c_m4) - 1 # 直近1ヶ月を除いた3ヶ月の勢い
    except: cm_val = 0

    # 共通計算：ボラドラ & Energy
    ret = p.pct_change().dropna()
    v_drag = ( (ret.tail(252*3).std() * np.sqrt(252))**2 ) / 2
    energy_s, tsi_val = get_energy_status(p)
    ma200 = p.rolling(200).mean().iloc[-1]

    # --- 判定：CLR (CAPE < 50% 時) ---
    score_clr = clr_val - v_drag + (PROFIT_MARGINS.get(t, 0) * 0.20315) * max(0, clr_val - v_drag)
    judge_clr = "🚀 FULL" if score_clr > 0.05 and energy_s == "OK" else "⏳ WAIT"
    clr_results.append({"Ticker": t, "Judge": judge_clr, "Score": score_clr, "Energy": energy_s, "CLR": clr_val, "V-Drag": v_drag})

    # --- 判定：CM (CAPE > 50% 時) ---
    score_cm = cm_val - v_drag
    is_trend = p_now > ma200
    judge_cm = "🔥 FULL" if is_trend and energy_s == "OK" and score_cm > 0 else "🚨 EXIT"
    cm_results.append({"Ticker": t, "Judge": judge_cm, "Speed": score_cm, "Energy": energy_s, "AboveMA200": "Yes" if is_trend else "No"})

# ==========================================================
# 4. ブラウザ表示（並列パネル）
# ==========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 CLRセクション (1年前基準)")
    if cape_pct < 0.50: st.caption("✅ 現在の推奨戦略: 逆張りエッジ最大化モード")
    st.dataframe(pd.DataFrame(clr_results).set_index("Ticker").sort_values("Score", ascending=False), height=600)

with col2:
    st.subheader("📈 CMセクション (1ヶ月前基準)")
    if cape_pct >= 0.50: st.caption("✅ 現在の推奨戦略: 順張りトレンド追随モード")
    st.dataframe(pd.DataFrame(cm_results).set_index("Ticker").sort_values("Speed", ascending=False), height=600)

st.divider()
st.write("※超長期成績規約：半分投入は禁止。Score > 0 かつ Energy OK の場合のみフルコミットする。")

# --- 5. 視覚的検証：0基準相対チャート（修正版） ---
st.divider()
st.subheader("📊 タイムトラベル分析（0基準相対比較）")

# 表示モードの選択
mode = st.radio("表示基準（アンカー）を選択してください:", 
                ["CLR基準 (1年前を0%)", "CM基準 (1ヶ月前を0%)"], horizontal=True)

anchor_val = 252 if "CLR" in mode else 21

# 判定結果から表示用のデータフレームを一時作成
df_for_plot = pd.DataFrame(clr_results).set_index("Ticker")

# グラフ化する銘柄の選択（Scoreが高い順にデフォルト表示）
# 変数名を df_for_plot に統一して NameError を回避
default_selected = df_for_plot.sort_values("Score", ascending=False).index[:5].tolist()
selected_tickers = st.multiselect("銘柄を選択:", TICKERS, default=default_selected)

if selected_tickers:
    import plotly.graph_objects as go
    fig = go.Figure()
    for t in selected_tickers:
        if t not in data.columns: continue
        p = data[t]
        ref_price = p.iloc[-anchor_val]
        # 騰落率の計算
        rel_p = (p / ref_price - 1) * 100
        
        fig.add_trace(go.Scatter(x=p.index, y=rel_p, name=t, hovertemplate='%{y:.2f}%'))

        # 200MAも相対化して表示
        ma200_rel = (p.rolling(200).mean() / ref_price - 1) * 100
        fig.add_trace(go.Scatter(x=p.index, y=ma200_rel, name=f"{t}(200MA)", 
                                 line=dict(dash='dot', width=1), visible='legendonly'))

    # 基準線（0%）
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=2)
    
    fig.update_layout(
        title=f"【{mode}】 期待値とトレンドの可視化",
        yaxis_title="騰落率 (%)",
        xaxis_title="日付",
        hovermode="x unified",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("※200MAは凡例をクリックすると表示されます。価格が0%（基準線）より下にあり、かつ200MAより大きく乖離しているほどCLRの期待値は高まります。")
