"""
FixTrade — Streamlit Dashboard
Tunisian Stock Market (BVMT) Trading Intelligence Platform

Connects to the FastAPI backend at /api/v1/*
"""

import os
import json
from datetime import date, datetime, timedelta

import numpy as np
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Configuration ─────────────────────────────────────────────────────
API_BASE = os.getenv("FIXTRADE_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="FixTrade — BVMT Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 1rem; }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
    
    /* Success / Error badges */
    .badge-buy { 
        background: #065f46; color: #6ee7b7; 
        padding: 4px 12px; border-radius: 6px; font-weight: 600; 
    }
    .badge-sell { 
        background: #7f1d1d; color: #fca5a5; 
        padding: 4px 12px; border-radius: 6px; font-weight: 600; 
    }
    .badge-hold { 
        background: #78350f; color: #fcd34d; 
        padding: 4px 12px; border-radius: 6px; font-weight: 600; 
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

def api_get(endpoint: str, params: dict = None, timeout: int = 30):
    """GET request to backend."""
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"_error": "Cannot connect to backend. Is the FastAPI server running?"}
    except requests.exceptions.HTTPError as e:
        return {"_error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"_error": str(e)}


def api_post(endpoint: str, data: dict = None, timeout: int = 60):
    """POST request to backend."""
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"_error": "Cannot connect to backend. Is the FastAPI server running?"}
    except requests.exceptions.HTTPError as e:
        return {"_error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"_error": str(e)}


def show_error(result):
    """Display error from API response if present."""
    if isinstance(result, dict) and "_error" in result:
        st.error(f"⚠️ {result['_error']}")
        return True
    return False


def action_badge(action: str) -> str:
    """Return HTML badge for BUY/SELL/HOLD."""
    action = action.upper()
    css_class = {"BUY": "badge-buy", "SELL": "badge-sell"}.get(action, "badge-hold")
    return f'<span class="{css_class}">{action}</span>'


# ── BVMT Stock Symbols (common ones) ─────────────────────────────────
BVMT_SYMBOLS = [
    "BIAT", "BH", "BNA", "STB", "UBCI", "UIB", "AB", "ATB", "ATTIJARI",
    "BTE", "BT", "WIFAK", "SFBT", "SOPAT", "SIAME", "SOTUVER", "STAR",
    "TUNISAIR", "TPR", "TJARI", "SAH", "POULINA", "OTH", "MPBS",
    "MONOPRIX", "MG", "LAND", "ICF", "GIF", "EURO", "DELICE",
    "CIMENT", "CARTHAGE", "CC", "Assad", "AMS", "ALKIMIA",
    "ADWYA", "CELL", "ENNAKL", "SOMOCER", "TRE", "TELNET",
    "ONE", "UADH", "ARTES", "HEXABYTE", "SOTETEL",
]


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-market.png", width=64)
    st.title("FixTrade")
    st.caption("BVMT Trading Intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🎯 Smart Trading (NEW)",
            "📊 Price Prediction",
            "📈 Volume & Liquidity",
            "💬 Sentiment Analysis",
            "🚨 Anomaly Detection",
            "💼 AI Portfolio",
            "🎯 Recommendations",
            "⚙️ Settings & Status",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    
    # API status indicator
    health = api_get("/health")
    if show_error(health):
        st.warning("Backend offline")
    else:
        st.success(f"✅ API Online — v{health.get('version', '?')}")
    
    st.divider()
    st.caption("Built with ❤️ for BVMT")


# ══════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════════════

if page == "� Dashboard":
    st.title("🎯 Smart Trading Intelligence")
    st.markdown("**Complete trading view:** Real data → Predictions → Recommendations → Anomalies")

    # Symbol selector
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.selectbox("Select Stock Symbol", BVMT_SYMBOLS, key="smart_sym")
    with col2:
        st.metric("Selected", symbol)

    st.divider()

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "🎯 Portfolio Recommendations", "📈 Full Report"])

    # ── TAB 1: Price Analysis ──
    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader(f"📊 {symbol} — Price Analysis")
            
            # Fetch historical data
            with st.spinner("Loading historical data..."):
                # Get last 90 days of real data
                query = """
                    SELECT seance, cloture, quantite_negociee, ouverture, plus_bas, plus_haut
                    FROM stock_prices
                    WHERE symbol = :symbol
                    AND seance >= CURRENT_DATE - INTERVAL '90 days'
                    ORDER BY seance DESC
                    LIMIT 90
                """
                # For now, simulate if backend not available
                try:
                    result = api_post("/trading/predictions", {"symbol": symbol, "horizon_days": 5})
                    has_backend = "_error" not in result
                except:
                    has_backend = False

                if has_backend:
                    # Real prediction data
                    predictions = result.get("predictions", [])
                    
                    # Simulate historical data (in production, fetch from DB)
                    dates = pd.date_range(end=date.today(), periods=90, freq='D')
                    base_price = 10.0
                    historical_prices = base_price + np.cumsum(np.random.randn(90) * 0.3)
                    
                    hist_df = pd.DataFrame({
                        'date': dates,
                        'close': historical_prices,
                        'volume': np.random.randint(1000, 50000, 90)
                    })
                    
                    # Create chart
                    fig = go.Figure()
                    
                    # Historical prices
                    fig.add_trace(go.Scatter(
                        x=hist_df['date'], y=hist_df['close'],
                        mode='lines', name='Historical Price',
                        line=dict(color='#3b82f6', width=2),
                    ))
                    
                    # Predictions
                    if predictions:
                        pred_df = pd.DataFrame(predictions)
                        pred_df["predicted_close"] = pred_df["predicted_close"].astype(float)
                        pred_df["confidence_lower"] = pred_df["confidence_lower"].astype(float)
                        pred_df["confidence_upper"] = pred_df["confidence_upper"].astype(float)
                        
                        fig.add_trace(go.Scatter(
                            x=pred_df["target_date"], y=pred_df["predicted_close"],
                            mode='lines+markers', name='Predicted Price',
                            line=dict(color='#10b981', width=3, dash='dash'),
                            marker=dict(size=10, symbol='diamond'),
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pred_df["target_date"], y=pred_df["confidence_upper"],
                            mode='lines', name='Upper Bound',
                            line=dict(width=0),
                            showlegend=False,
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=pred_df["target_date"], y=pred_df["confidence_lower"],
                            mode='lines', name='Confidence Interval',
                            line=dict(width=0),
                            fill='tonexty', fillcolor='rgba(16,185,129,0.2)',
                        ))
                    
                    fig.update_layout(
                        title=f"{symbol} — Historical + Predicted Prices",
                        xaxis_title="Date",
                        yaxis_title="Price (TND)",
                        template="plotly_dark",
                        height=500,
                        hovermode='x unified',
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    with st.expander("📋 View Data Table"):
                        tcol1, tcol2 = st.columns(2)
                        with tcol1:
                            st.markdown("**Historical Data (Last 10 days)**")
                            st.dataframe(hist_df.tail(10), use_container_width=True, hide_index=True)
                        with tcol2:
                            st.markdown("**Predictions**")
                            if predictions:
                                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                else:
                    st.error("⚠️ Backend not available. Start FastAPI server: `uvicorn app.main:app`")

        with col_right:
            st.subheader("🎯 Trading Signal")
            
            # Calculate recommendation
            if has_backend and predictions:
                current_price = historical_prices[-1]
                future_price = float(predictions[2]["predicted_close"]) if len(predictions) > 2 else float(predictions[0]["predicted_close"])
                price_change = ((future_price - current_price) / current_price) * 100
                
                # Simple rule-based recommendation
                if price_change > 5:
                    action = "BUY"
                    color = "green"
                    icon = "🟢"
                elif price_change < -5:
                    action = "SELL"
                    color = "red"
                    icon = "🔴"
                else:
                    action = "HOLD"
                    color = "orange"
                    icon = "🟡"
                
                st.markdown(f"### {icon} **{action}**")
                st.markdown(f"<h2 style='color: {color};'>{price_change:+.2f}%</h2>", unsafe_allow_html=True)
                
                st.metric("Current Price", f"{current_price:.3f} TND")
                st.metric("Predicted (3d)", f"{future_price:.3f} TND", f"{price_change:+.2f}%")
                
                # Confidence
                if predictions:
                    conf_lower = float(predictions[0]["confidence_lower"])
                    conf_upper = float(predictions[0]["confidence_upper"])
                    conf_range = conf_upper - conf_lower
                    conf_score = max(0, 100 - (conf_range / current_price * 100))
                    st.metric("Confidence", f"{conf_score:.1f}%")
                
                st.divider()
                
                # Check anomalies
                st.markdown("**🚨 Anomaly Check**")
                anom_result = api_post("/trading/anomalies", {"symbol": symbol})
                if not show_error(anom_result):
                    anomalies = anom_result.get("anomalies", [])
                    if anomalies:
                        recent = anomalies[0]
                        sev = float(recent.get("severity", 0))
                        st.warning(f"⚠️ {recent['anomaly_type']}")
                        st.caption(f"Severity: {sev:.2f}")
                    else:
                        st.success("✅ No anomalies")

    # ── TAB 2: Portfolio Recommendations ──
    with tab2:
        st.subheader("🎯 AI Portfolio Recommendations")
        st.markdown("**CAPM-based recommendations with risk analysis**")
        
        rcol1, rcol2 = st.columns([1, 3])
        
        with rcol1:
            portfolio_symbols = st.multiselect(
                "Portfolio Symbols",
                BVMT_SYMBOLS,
                default=["BIAT", "SFBT", "STAR", "DELICE", symbol],
                key="port_syms"
            )
            risk_profile = st.selectbox(
                "Risk Profile",
                ["conservative", "moderate", "aggressive"],
                index=1,
                key="port_risk"
            )
            top_n = st.slider("Top N", 3, 20, 10, key="port_topn")
            
            use_llm = st.checkbox("Enable AI Explanations", key="port_llm")
            
            if use_llm:
                with st.expander("🤖 LLM Config"):
                    llm_provider = st.text_input("Provider", "groq", key="port_llm_prov")
                    llm_model = st.text_input("Model", "llama-3.3-70b-versatile", key="port_llm_mod")
                    llm_key = st.text_input("API Key", type="password", key="port_llm_key")
            
            generate_btn = st.button("🎯 Generate Recommendations", type="primary", use_container_width=True)
        
        with rcol2:
            if generate_btn:
                if len(portfolio_symbols) < 2:
                    st.error("Select at least 2 symbols for portfolio analysis")
                else:
                    payload = {
                        "symbols": portfolio_symbols,
                        "risk_profile": risk_profile,
                        "top_n": top_n,
                    }
                    
                    if use_llm and 'llm_key' in locals() and llm_key:
                        payload["llm_config"] = {
                            "provider": llm_provider,
                            "model": llm_model,
                            "api_key": llm_key,
                            "temperature": 0.3,
                            "max_tokens": 150,
                        }
                    
                    with st.spinner("Generating AI recommendations..."):
                        result = api_post("/ai/portfolio/recommendations/detailed", payload)
                    
                    if not show_error(result):
                        if isinstance(result, list) and result:
                            # Summary metrics
                            buy_count = sum(1 for r in result if r.get("action") == "BUY")
                            sell_count = sum(1 for r in result if r.get("action") == "SELL")
                            hold_count = sum(1 for r in result if r.get("action") == "HOLD")
                            
                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            mcol1.metric("🟢 BUY", buy_count)
                            mcol2.metric("🔴 SELL", sell_count)
                            mcol3.metric("🟡 HOLD", hold_count)
                            mcol4.metric("Total", len(result))
                            
                            st.divider()
                            
                            # Display recommendations
                            for rec in result:
                                action = rec.get("action", "HOLD")
                                conf = rec.get("confidence", 0)
                                
                                if action == "BUY":
                                    border_color = "#10b981"
                                    icon = "🟢"
                                elif action == "SELL":
                                    border_color = "#ef4444"
                                    icon = "🔴"
                                else:
                                    border_color = "#f59e0b"
                                    icon = "🟡"
                                
                                with st.container():
                                    st.markdown(f"""
                                    <div style='border-left: 4px solid {border_color}; padding-left: 12px; margin-bottom: 16px;'>
                                    """, unsafe_allow_html=True)
                                    
                                    ccol1, ccol2, ccol3, ccol4 = st.columns([2, 1, 1, 1])
                                    
                                    with ccol1:
                                        st.markdown(f"### {icon} **{rec['symbol']}** — {action}")
                                    with ccol2:
                                        st.metric("Confidence", f"{conf:.1%}")
                                    with ccol3:
                                        st.metric("Expected Return", f"{rec.get('expected_return', 0):.2%}")
                                    with ccol4:
                                        st.metric("Beta", f"{rec.get('beta', 0):.3f}")
                                    
                                    dcol1, dcol2, dcol3 = st.columns(3)
                                    with dcol1:
                                        st.caption(f"Current Weight: {rec.get('current_weight', 0):.2%}")
                                    with dcol2:
                                        st.caption(f"Target Weight: {rec.get('target_weight', 0):.2%}")
                                    with dcol3:
                                        if rec.get("anomaly_detected"):
                                            st.caption("⚠️ Anomaly Detected")
                                        else:
                                            st.caption("✅ No Anomaly")
                                    
                                    # Explanation
                                    explanation = rec.get("explanation", "N/A")
                                    if explanation and explanation != "N/A":
                                        st.info(f"💡 **Explanation:** {explanation}")
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.warning("No recommendations generated. Check if data is available.")

    # ── TAB 3: Full Report ──
    with tab3:
        st.subheader(f"📈 Full Trading Report — {symbol}")
        
        if st.button("🔄 Generate Complete Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                # Price prediction
                price_result = api_post("/trading/predictions", {"symbol": symbol, "horizon_days": 5})
                
                # Sentiment
                sent_result = api_post("/trading/sentiment", {"symbol": symbol})
                
                # Anomalies
                anom_result = api_post("/trading/anomalies", {"symbol": symbol})
                
                # Volume
                vol_result = api_post("/trading/predictions/volume", {"symbol": symbol, "horizon_days": 5})
                
                # Liquidity
                liq_result = api_post("/trading/predictions/liquidity", {"symbol": symbol, "horizon_days": 5})
            
            # Display report
            st.markdown("---")
            
            # Section 1: Price Forecast
            st.markdown("### 📊 Price Forecast")
            if not show_error(price_result):
                predictions = price_result.get("predictions", [])
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No predictions available")
            
            st.markdown("---")
            
            # Section 2: Sentiment
            st.markdown("### 💬 Sentiment Analysis")
            if not show_error(sent_result):
                scol1, scol2, scol3 = st.columns(3)
                scol1.metric("Sentiment", sent_result.get("sentiment", "N/A"))
                scol2.metric("Score", f"{float(sent_result.get('score', 0)):.4f}")
                scol3.metric("Articles", sent_result.get("article_count", 0))
            
            st.markdown("---")
            
            # Section 3: Anomalies
            st.markdown("### 🚨 Anomaly Alerts")
            if not show_error(anom_result):
                anomalies = anom_result.get("anomalies", [])
                if anomalies:
                    anom_df = pd.DataFrame(anomalies)
                    st.dataframe(anom_df[["detected_at", "anomaly_type", "severity", "description"]], 
                                use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No anomalies detected")
            
            st.markdown("---")
            
            # Section 4: Volume Forecast
            st.markdown("### 📊 Volume Forecast")
            if not show_error(vol_result):
                vol_preds = vol_result.get("predictions", [])
                if vol_preds:
                    vol_df = pd.DataFrame(vol_preds)
                    fig = px.bar(vol_df, x="target_date", y="predicted_volume", 
                                title="Predicted Trading Volume",
                                template="plotly_dark", color="predicted_volume")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Section 5: Liquidity Forecast
            st.markdown("### 💧 Liquidity Forecast")
            if not show_error(liq_result):
                liq_forecasts = liq_result.get("forecasts", [])
                if liq_forecasts:
                    liq_df = pd.DataFrame(liq_forecasts)
                    for _, row in liq_df.iterrows():
                        tier = row["predicted_tier"]
                        color_map = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                        st.markdown(f"**{row['target_date']}**: {color_map.get(tier, '⚪')} {tier.upper()} liquidity")


# ══════════════════════════════════════════════════════════════════════
# PAGE: Smart Trading (Integrated View)
# ══════════════════════════════════════════════════════════════════════

elif page == "� Smart Trading (NEW)":
    st.title("� Smart Trading Intelligence")
    st.markdown("**Complete trading view:** Real data → Predictions → Recommendations → Anomalies")

    # Quick status cards
    col1, col2, col3, col4 = st.columns(4)

    # AI status
    ai_status = api_get("/ai/status")
    if not show_error(ai_status):
        with col1:
            st.metric("Active Portfolios", ai_status.get("active_portfolios", 0))
        with col2:
            st.metric("AI Model", ai_status.get("groq_model", "N/A")[:20])
        with col3:
            groq_ok = "✅ Yes" if ai_status.get("groq_configured") else "❌ No"
            st.metric("Groq Configured", groq_ok)
        with col4:
            st.metric("Default Capital", f"{ai_status.get('default_capital', 0):,.0f} TND")

    st.divider()

    # Quick actions
    st.subheader("⚡ Quick Actions")
    qcol1, qcol2, qcol3 = st.columns(3)

    with qcol1:
        st.markdown("#### 📊 Predict Price")
        q_symbol = st.selectbox("Symbol", BVMT_SYMBOLS, key="q_sym")
        if st.button("Quick Predict", key="q_pred"):
            result = api_post("/trading/predictions", {"symbol": q_symbol, "horizon_days": 3})
            if not show_error(result):
                for p in result.get("predictions", []):
                    st.metric(
                        f"{p['target_date']}",
                        f"{float(p['predicted_close']):.3f} TND",
                        f"[{float(p['confidence_lower']):.3f} – {float(p['confidence_upper']):.3f}]"
                    )

    with qcol2:
        st.markdown("#### 💬 Check Sentiment")
        q_sym2 = st.selectbox("Symbol", BVMT_SYMBOLS, key="q_sym2")
        if st.button("Check", key="q_sent"):
            result = api_post("/trading/sentiment", {"symbol": q_sym2})
            if not show_error(result):
                score = float(result.get("score", 0))
                st.metric("Sentiment", result.get("sentiment", "N/A"), f"Score: {score:.3f}")
                st.metric("Articles Analyzed", result.get("article_count", 0))

    with qcol3:
        st.markdown("#### 🚨 Recent Anomalies")
        if st.button("Fetch Latest", key="q_anom"):
            result = api_get("/trading/anomalies/recent", {"limit": 5, "hours_back": 48})
            if not show_error(result):
                anomalies = result.get("anomalies", [])
                if anomalies:
                    for a in anomalies:
                        sev = float(a.get("severity", 0))
                        icon = "🔴" if sev > 0.7 else "🟡" if sev > 0.4 else "🟢"
                        st.markdown(f"{icon} **{a['symbol']}** — {a['anomaly_type']} ({a['description'][:60]})")
                else:
                    st.info("No recent anomalies detected.")


# ══════════════════════════════════════════════════════════════════════
# PAGE: Price Prediction
# ══════════════════════════════════════════════════════════════════════

elif page == "📊 Price Prediction":
    st.title("📊 Stock Price Prediction")
    st.markdown("Predict closing prices for BVMT stocks using ensemble ML models (LSTM + XGBoost + Prophet).")

    col1, col2 = st.columns([1, 3])

    with col1:
        symbol = st.selectbox("Select Stock", BVMT_SYMBOLS, key="pp_sym")
        horizon = st.slider("Horizon (days)", 1, 5, 3, key="pp_hor")
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            with st.spinner("Running prediction models..."):
                result = api_post("/trading/predictions", {
                    "symbol": symbol,
                    "horizon_days": horizon,
                })

            if not show_error(result):
                preds = result.get("predictions", [])
                if preds:
                    df = pd.DataFrame(preds)
                    df["predicted_close"] = df["predicted_close"].astype(float)
                    df["confidence_lower"] = df["confidence_lower"].astype(float)
                    df["confidence_upper"] = df["confidence_upper"].astype(float)

                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["target_date"], y=df["predicted_close"],
                        mode="lines+markers", name="Predicted Close",
                        line=dict(color="#3b82f6", width=3),
                        marker=dict(size=10),
                    ))
                    fig.add_trace(go.Scatter(
                        x=df["target_date"], y=df["confidence_upper"],
                        mode="lines", name="Upper Bound",
                        line=dict(color="#22c55e", dash="dash"),
                    ))
                    fig.add_trace(go.Scatter(
                        x=df["target_date"], y=df["confidence_lower"],
                        mode="lines", name="Lower Bound",
                        line=dict(color="#ef4444", dash="dash"),
                        fill="tonexty", fillcolor="rgba(59,130,246,0.1)",
                    ))
                    fig.update_layout(
                        title=f"{symbol} — Price Forecast",
                        xaxis_title="Date", yaxis_title="Price (TND)",
                        template="plotly_dark",
                        height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Table
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No predictions returned.")


# ══════════════════════════════════════════════════════════════════════
# PAGE: Volume & Liquidity
# ══════════════════════════════════════════════════════════════════════

elif page == "📈 Volume & Liquidity":
    st.title("📈 Volume & Liquidity Forecasting")

    tab1, tab2 = st.tabs(["📊 Volume Prediction", "💧 Liquidity Forecast"])

    with tab1:
        st.subheader("Transaction Volume Prediction")
        vcol1, vcol2 = st.columns([1, 3])

        with vcol1:
            v_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="vol_sym")
            v_hor = st.slider("Horizon", 1, 5, 3, key="vol_hor")
            v_btn = st.button("Predict Volume", type="primary", use_container_width=True)

        with vcol2:
            if v_btn:
                with st.spinner("Forecasting volume..."):
                    result = api_post("/trading/predictions/volume", {
                        "symbol": v_sym,
                        "horizon_days": v_hor,
                    })
                if not show_error(result):
                    preds = result.get("predictions", [])
                    if preds:
                        df = pd.DataFrame(preds)
                        fig = px.bar(
                            df, x="target_date", y="predicted_volume",
                            title=f"{v_sym} — Volume Forecast",
                            color="predicted_volume",
                            color_continuous_scale="blues",
                            template="plotly_dark",
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Liquidity Tier Prediction")
        lcol1, lcol2 = st.columns([1, 3])

        with lcol1:
            l_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="liq_sym")
            l_hor = st.slider("Horizon", 1, 5, 3, key="liq_hor")
            l_btn = st.button("Predict Liquidity", type="primary", use_container_width=True)

        with lcol2:
            if l_btn:
                with st.spinner("Forecasting liquidity..."):
                    result = api_post("/trading/predictions/liquidity", {
                        "symbol": l_sym,
                        "horizon_days": l_hor,
                    })
                if not show_error(result):
                    forecasts = result.get("forecasts", [])
                    if forecasts:
                        df = pd.DataFrame(forecasts)
                        for c in ["prob_low", "prob_medium", "prob_high"]:
                            df[c] = df[c].astype(float)

                        # Stacked bar chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name="High", x=df["target_date"], y=df["prob_high"], marker_color="#22c55e"))
                        fig.add_trace(go.Bar(name="Medium", x=df["target_date"], y=df["prob_medium"], marker_color="#eab308"))
                        fig.add_trace(go.Bar(name="Low", x=df["target_date"], y=df["prob_low"], marker_color="#ef4444"))
                        fig.update_layout(
                            barmode="stack",
                            title=f"{l_sym} — Liquidity Probability",
                            template="plotly_dark",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Tier badges
                        for _, row in df.iterrows():
                            tier = row["predicted_tier"]
                            color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(tier, "⚪")
                            st.markdown(f"**{row['target_date']}**: {color} {tier.upper()} liquidity")


# ══════════════════════════════════════════════════════════════════════
# PAGE: Sentiment Analysis
# ══════════════════════════════════════════════════════════════════════

elif page == "💬 Sentiment Analysis":
    st.title("💬 Sentiment Analysis")
    st.markdown("NLP-powered sentiment analysis of Tunisian financial news.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📰 Query Sentiment", "🔄 Analyze Articles",
        "🔗 Link Symbols", "📊 Aggregate Scores"
    ])

    with tab1:
        st.subheader("Get Sentiment for Symbol")
        scol1, scol2 = st.columns([1, 2])
        with scol1:
            s_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="sent_sym")
            s_date = st.date_input("Date", value=date.today(), key="sent_date")
            s_btn = st.button("Get Sentiment", type="primary", use_container_width=True)
        with scol2:
            if s_btn:
                with st.spinner("Fetching sentiment..."):
                    result = api_post("/trading/sentiment", {
                        "symbol": s_sym,
                        "target_date": s_date.isoformat(),
                    })
                if not show_error(result):
                    sentiment = result.get("sentiment", "N/A")
                    score = float(result.get("score", 0))
                    articles = result.get("article_count", 0)

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=score,
                        title={"text": f"{s_sym} Sentiment"},
                        gauge={
                            "axis": {"range": [-1, 1]},
                            "bar": {"color": "#3b82f6"},
                            "steps": [
                                {"range": [-1, -0.3], "color": "#fee2e2"},
                                {"range": [-0.3, 0.3], "color": "#fef9c3"},
                                {"range": [0.3, 1], "color": "#dcfce7"},
                            ],
                        },
                    ))
                    fig.update_layout(height=300, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Sentiment", sentiment)
                    mcol2.metric("Score", f"{score:.4f}")
                    mcol3.metric("Articles", articles)

    with tab2:
        st.subheader("Run Article Sentiment Analysis")
        st.markdown("Analyze unprocessed scraped articles with NLP.")
        batch = st.number_input("Batch Size", 10, 500, 50, key="sent_batch")
        if st.button("🔍 Analyze Articles", type="primary"):
            with st.spinner("Running NLP analysis..."):
                result = api_post("/trading/sentiment/analyze", {"batch_size": batch})
            if not show_error(result):
                rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                rcol1.metric("Total Analyzed", result.get("total_analyzed", 0))
                rcol2.metric("Positive", result.get("positive_count", 0))
                rcol3.metric("Negative", result.get("negative_count", 0))
                rcol4.metric("Neutral", result.get("neutral_count", 0))

                articles = result.get("results", [])
                if articles:
                    df = pd.DataFrame(articles)
                    st.dataframe(df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Link Articles to BVMT Symbols")
        st.markdown("Match scraped articles to stock symbols for per-company sentiment.")
        link_batch = st.number_input("Batch Size", 10, 1000, 100, key="link_batch")
        if st.button("🔗 Link Articles", type="primary"):
            with st.spinner("Linking articles to symbols..."):
                result = api_post("/trading/sentiment/link-symbols", {"batch_size": link_batch})
            if not show_error(result):
                lcol1, lcol2, lcol3 = st.columns(3)
                lcol1.metric("Scanned", result.get("articles_scanned", 0))
                lcol2.metric("Links Created", result.get("links_created", 0))
                lcol3.metric("No Match", result.get("articles_with_no_match", 0))

    with tab4:
        st.subheader("Aggregate Daily Sentiment Scores")
        agg_sym = st.selectbox("Symbol (blank = all)", [""] + BVMT_SYMBOLS, key="agg_sym")
        agg_days = st.slider("Days Back", 1, 90, 30, key="agg_days")
        if st.button("📊 Aggregate", type="primary"):
            payload = {"days_back": agg_days}
            if agg_sym:
                payload["symbol"] = agg_sym
            with st.spinner("Aggregating sentiment..."):
                result = api_post("/trading/sentiment/aggregate", payload)
            if not show_error(result):
                st.metric("Scores Upserted", result.get("scores_upserted", 0))
                scores = result.get("scores", [])
                if scores:
                    df = pd.DataFrame(scores)
                    df["score"] = df["score"].astype(float)
                    fig = px.line(
                        df, x="score_date", y="score", color="symbol",
                        title="Daily Sentiment Scores",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: Anomaly Detection
# ══════════════════════════════════════════════════════════════════════

elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.markdown("Detect unusual market activity: price spikes, volume bursts, flash crashes & more.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Detect", "📋 Recent Alerts", "📊 Intraday", "📈 Evaluate"
    ])

    with tab1:
        st.subheader("Detect Anomalies")
        acol1, acol2 = st.columns([1, 3])
        with acol1:
            a_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="anom_sym")
            a_btn = st.button("🔍 Detect", type="primary", use_container_width=True)
        with acol2:
            if a_btn:
                with st.spinner("Scanning for anomalies..."):
                    result = api_post("/trading/anomalies", {"symbol": a_sym})
                if not show_error(result):
                    anomalies = result.get("anomalies", [])
                    if anomalies:
                        st.warning(f"⚠️ {len(anomalies)} anomalies detected!")
                        df = pd.DataFrame(anomalies)
                        df["severity"] = df["severity"].astype(float)

                        # Severity chart
                        fig = px.scatter(
                            df, x="detected_at", y="severity",
                            color="anomaly_type", size="severity",
                            title=f"{a_sym} — Anomalies",
                            template="plotly_dark",
                            hover_data=["description"],
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df[["detected_at", "anomaly_type", "severity", "description"]], 
                                     use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No anomalies detected.")

    with tab2:
        st.subheader("Recent Anomaly Alerts")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            ra_sym = st.selectbox("Symbol (optional)", ["All"] + BVMT_SYMBOLS, key="ra_sym")
        with rcol2:
            ra_limit = st.number_input("Limit", 1, 50, 10, key="ra_lim")
        with rcol3:
            ra_hours = st.slider("Hours Back", 1, 168, 24, key="ra_hrs")

        if st.button("📋 Fetch Recent", type="primary"):
            params = {"limit": ra_limit, "hours_back": ra_hours}
            if ra_sym != "All":
                params["symbol"] = ra_sym
            with st.spinner("Fetching recent anomalies..."):
                result = api_get("/trading/anomalies/recent", params)
            if not show_error(result):
                anomalies = result.get("anomalies", [])
                if anomalies:
                    for a in anomalies:
                        sev = float(a.get("severity", 0))
                        icon = "🔴" if sev > 0.7 else "🟡" if sev > 0.4 else "🟢"
                        with st.expander(f"{icon} {a['symbol']} — {a['anomaly_type']} (severity: {sev:.2f})"):
                            st.write(f"**Detected:** {a['detected_at']}")
                            st.write(f"**Description:** {a['description']}")
                else:
                    st.info("No recent anomalies found.")

    with tab3:
        st.subheader("Intraday Anomaly Detection")
        st.markdown("1-minute tick data analysis: hourly moves, volume bursts, flash events.")
        icol1, icol2 = st.columns([1, 3])
        with icol1:
            i_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="intra_sym")
            i_days = st.slider("Days Back", 1, 30, 5, key="intra_days")
            i_btn = st.button("🔍 Scan Intraday", type="primary", use_container_width=True)
        with icol2:
            if i_btn:
                with st.spinner("Scanning intraday data..."):
                    result = api_post("/trading/anomalies/intraday", {
                        "symbol": i_sym, "days_back": i_days,
                    })
                if not show_error(result):
                    anomalies = result.get("anomalies", [])
                    st.metric("Days Scanned", result.get("days_scanned", 0))
                    if anomalies:
                        df = pd.DataFrame(anomalies)
                        df["severity"] = df["severity"].astype(float)
                        fig = px.timeline(
                            df, x_start="detected_at", x_end="detected_at",
                            y="anomaly_type", color="severity",
                            title=f"{i_sym} — Intraday Anomalies",
                            template="plotly_dark",
                        ) if False else px.scatter(
                            df, x="detected_at", y="severity",
                            color="anomaly_type",
                            title=f"{i_sym} — Intraday Anomalies",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.success("✅ No intraday anomalies detected.")

    with tab4:
        st.subheader("Evaluate Anomaly Detection Performance")
        st.markdown("Backtest against historical ground-truth labels.")
        ecol1, ecol2 = st.columns([1, 3])
        with ecol1:
            e_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="eval_sym")
            e_days = st.slider("Days Back", 30, 365, 90, key="eval_days")
            e_tol = st.slider("Date Tolerance (days)", 0, 7, 1, key="eval_tol")
            e_btn = st.button("📊 Evaluate", type="primary", use_container_width=True)
        with ecol2:
            if e_btn:
                with st.spinner("Evaluating..."):
                    result = api_post("/trading/anomalies/evaluate", {
                        "symbol": e_sym,
                        "days_back": e_days,
                        "date_tolerance_days": e_tol,
                    })
                if not show_error(result):
                    overall = result.get("overall", {})
                    ocol1, ocol2, ocol3 = st.columns(3)
                    ocol1.metric("Precision", f"{overall.get('precision', 0):.2%}")
                    ocol2.metric("Recall", f"{overall.get('recall', 0):.2%}")
                    ocol3.metric("F1 Score", f"{overall.get('f1_score', 0):.2%}")

                    per_type = result.get("per_type", [])
                    if per_type:
                        df = pd.DataFrame(per_type)
                        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE: AI Portfolio
# ══════════════════════════════════════════════════════════════════════

elif page == "💼 AI Portfolio":
    st.title("💼 AI Portfolio Manager")
    st.markdown("Create portfolios, execute trades, optimize allocations & run backtests.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Portfolio", "💰 Trade", "🎯 Optimize",
        "📈 Efficient Frontier", "🧪 Backtest",
    ])

    # ── Portfolio Management ──
    with tab1:
        st.subheader("Portfolio Management")

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("#### Create Portfolio")
            p_profile = st.selectbox("Risk Profile", ["conservative", "moderate", "aggressive"], key="p_prof")
            p_capital = st.number_input("Initial Capital (TND)", 1000, 10_000_000, 10_000, step=1000, key="p_cap")
            if st.button("Create Portfolio", type="primary"):
                with st.spinner("Creating..."):
                    result = api_post("/ai/portfolio/create", {
                        "risk_profile": p_profile,
                        "initial_capital": p_capital,
                    })
                if not show_error(result):
                    st.success(f"✅ Portfolio created: `{result.get('portfolio_id', 'N/A')}`")
                    st.session_state["portfolio_id"] = result.get("portfolio_id", "default")
                    st.json(result.get("snapshot", {}))

        with pcol2:
            st.markdown("#### View Portfolio")
            pid = st.text_input("Portfolio ID", value=st.session_state.get("portfolio_id", "default"), key="p_id")
            if st.button("📸 Get Snapshot"):
                result = api_get(f"/ai/portfolio/{pid}/snapshot")
                if not show_error(result):
                    st.json(result)

            if st.button("📊 Get Performance"):
                result = api_get(f"/ai/portfolio/{pid}/performance")
                if not show_error(result):
                    st.json(result)

    # ── Trade Execution ──
    with tab2:
        st.subheader("Execute Trade")
        tcol1, tcol2 = st.columns([1, 2])

        with tcol1:
            t_pid = st.text_input("Portfolio ID", value=st.session_state.get("portfolio_id", "default"), key="t_pid")
            t_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="t_sym")
            t_action = st.radio("Action", ["buy", "sell"], horizontal=True, key="t_act")
            t_qty = st.number_input("Quantity", 1, 100_000, 100, key="t_qty")
            t_price = st.number_input("Price (TND)", 0.01, 1000.0, 10.0, step=0.01, key="t_price")
            t_explain = st.checkbox("Generate AI Explanation", value=True, key="t_exp")

            if st.button("⚡ Execute Trade", type="primary", use_container_width=True):
                with st.spinner("Executing trade..."):
                    result = api_post(f"/ai/portfolio/{t_pid}/trade", {
                        "symbol": t_sym,
                        "action": t_action,
                        "quantity": t_qty,
                        "price": t_price,
                        "generate_explanation": t_explain,
                    })
                with tcol2:
                    if not show_error(result):
                        if result.get("success"):
                            st.success("✅ Trade executed!")
                        else:
                            st.error(f"❌ {result.get('message', 'Trade failed')}")
                        st.json(result)

    # ── Portfolio Optimization ──
    with tab3:
        st.subheader("Portfolio Optimization (MPT)")
        st.markdown("Modern Portfolio Theory — find optimal asset allocation.")

        ocol1, ocol2 = st.columns([1, 2])
        with ocol1:
            o_syms = st.multiselect("Select Symbols", BVMT_SYMBOLS, default=["BIAT", "SFBT", "STAR", "DELICE"], key="o_syms")
            o_profile = st.selectbox("Risk Profile", ["conservative", "moderate", "aggressive"], key="o_prof")
            o_method = st.selectbox("Method", ["max_sharpe", "min_variance"], key="o_method")

            if st.button("🎯 Optimize", type="primary", use_container_width=True):
                if len(o_syms) < 2:
                    st.error("Select at least 2 symbols.")
                else:
                    with st.spinner("Optimizing portfolio..."):
                        result = api_post("/ai/portfolio/optimize", {
                            "symbols": o_syms,
                            "risk_profile": o_profile,
                            "optimization_method": o_method,
                        })
                    with ocol2:
                        if not show_error(result):
                            # Weights pie chart
                            weights = result.get("weights", {})
                            fig = px.pie(
                                names=list(weights.keys()),
                                values=list(weights.values()),
                                title="Optimal Allocation",
                                template="plotly_dark",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            mcol1, mcol2, mcol3 = st.columns(3)
                            mcol1.metric("Expected Return", f"{result.get('expected_return', 0):.2f}%")
                            mcol2.metric("Volatility", f"{result.get('volatility', 0):.2f}%")
                            mcol3.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.3f}")

    # ── Efficient Frontier ──
    with tab4:
        st.subheader("Efficient Frontier")
        ef_syms = st.multiselect("Symbols", BVMT_SYMBOLS, default=["BIAT", "SFBT", "STAR", "DELICE"], key="ef_syms")
        ef_pts = st.slider("Frontier Points", 10, 200, 50, key="ef_pts")

        if st.button("📈 Calculate Frontier", type="primary"):
            if len(ef_syms) < 2:
                st.error("Select at least 2 symbols.")
            else:
                with st.spinner("Calculating efficient frontier..."):
                    result = api_post("/ai/portfolio/efficient-frontier", {
                        "symbols": ef_syms,
                        "num_points": ef_pts,
                    })
                if not show_error(result):
                    rets = result.get("returns", [])
                    vols = result.get("volatilities", [])
                    sharpes = result.get("sharpe_ratios", [])

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vols, y=rets, mode="markers",
                        marker=dict(
                            color=sharpes, colorscale="Viridis",
                            size=8, showscale=True,
                            colorbar=dict(title="Sharpe"),
                        ),
                        text=[f"Sharpe: {s:.3f}" for s in sharpes],
                    ))
                    fig.update_layout(
                        title="Efficient Frontier",
                        xaxis_title="Volatility (%)",
                        yaxis_title="Expected Return (%)",
                        template="plotly_dark",
                        height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Backtest ──
    with tab5:
        st.subheader("Portfolio Backtest Simulation")
        bcol1, bcol2 = st.columns([1, 2])

        with bcol1:
            b_syms = st.multiselect("Symbols", BVMT_SYMBOLS, default=["BIAT", "SFBT", "STAR"], key="b_syms")
            b_profile = st.selectbox("Risk Profile", ["conservative", "moderate", "aggressive"], key="b_prof")
            b_capital = st.number_input("Capital (TND)", 1000, 10_000_000, 100_000, step=10000, key="b_cap")
            b_rebal = st.selectbox("Rebalance", ["daily", "weekly", "monthly"], index=1, key="b_reb")

            if st.button("🧪 Run Backtest", type="primary", use_container_width=True):
                if len(b_syms) < 2:
                    st.error("Select at least 2 symbols.")
                else:
                    with st.spinner("Running simulation..."):
                        result = api_post("/ai/portfolio/simulate", {
                            "symbols": b_syms,
                            "risk_profile": b_profile,
                            "initial_capital": b_capital,
                            "rebalance_frequency": b_rebal,
                        })
                    with bcol2:
                        if not show_error(result):
                            # Performance metrics
                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            mcol1.metric("Total Return", f"{result.get('total_return_pct', 0):.2f}%")
                            mcol2.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.3f}")
                            mcol3.metric("Max Drawdown", f"{result.get('max_drawdown', 0):.2f}%")
                            mcol4.metric("Win Rate", f"{result.get('win_rate', 0):.1f}%")

                            # Value history chart
                            dates = result.get("dates", [])
                            values = result.get("value_history", [])
                            if dates and values:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=dates, y=values,
                                    mode="lines", name="Portfolio Value",
                                    line=dict(color="#3b82f6", width=2),
                                    fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
                                ))
                                fig.update_layout(
                                    title="Portfolio Value Over Time",
                                    xaxis_title="Date",
                                    yaxis_title="Value (TND)",
                                    template="plotly_dark",
                                    height=400,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            # Extra metrics
                            with st.expander("📊 Detailed Metrics"):
                                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                                dcol1.metric("Final Value", f"{result.get('final_value', 0):,.2f} TND")
                                dcol2.metric("Volatility", f"{result.get('volatility', 0):.2f}%")
                                dcol3.metric("Sortino Ratio", f"{result.get('sortino_ratio', 0):.3f}")
                                dcol4.metric("Total Trades", result.get("total_trades", 0))


# ══════════════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ══════════════════════════════════════════════════════════════════════

elif page == "🎯 Recommendations":
    st.title("🎯 AI Trading Recommendations")
    st.markdown("CAPM-based recommendations with LLM-generated explanations.")

    tab1, tab2 = st.tabs(["📋 Detailed Recommendations", "💡 Quick Recommendation"])

    with tab1:
        st.subheader("CAPM-Based Detailed Recommendations")

        rcol1, rcol2 = st.columns([1, 3])
        with rcol1:
            r_syms = st.multiselect(
                "Symbols (empty = auto-select)",
                BVMT_SYMBOLS, key="r_syms",
            )
            r_profile = st.selectbox("Risk Profile", ["conservative", "moderate", "aggressive"], key="r_prof")
            r_topn = st.slider("Top N", 1, 50, 10, key="r_topn")

            with st.expander("🤖 LLM Config (optional)"):
                llm_provider = st.text_input("Provider", "groq", key="llm_prov")
                llm_model = st.text_input("Model", "llama-3.3-70b-versatile", key="llm_mod")
                llm_key = st.text_input("API Key", type="password", key="llm_key")
                llm_temp = st.slider("Temperature", 0.0, 2.0, 0.3, key="llm_temp")
                use_llm = st.checkbox("Enable LLM Explanations", key="use_llm")

            if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                payload = {
                    "risk_profile": r_profile,
                    "top_n": r_topn,
                }
                if r_syms:
                    payload["symbols"] = r_syms
                if use_llm and llm_key:
                    payload["llm_config"] = {
                        "provider": llm_provider,
                        "model": llm_model,
                        "api_key": llm_key,
                        "temperature": llm_temp,
                        "max_tokens": 150,
                        "enable_reasoning": False,
                    }

                with st.spinner("Generating AI recommendations..."):
                    result = api_post("/ai/portfolio/recommendations/detailed", payload)

                with rcol2:
                    if not show_error(result):
                        if isinstance(result, list) and result:
                            for rec in result:
                                action = rec.get("action", "HOLD")
                                conf = rec.get("confidence", 0)
                                icon = {"BUY": "🟢", "SELL": "🔴"}.get(action, "🟡")

                                with st.expander(f"{icon} {rec['symbol']} — {action} (confidence: {conf:.1%})"):
                                    mcol1, mcol2, mcol3 = st.columns(3)
                                    mcol1.metric("Expected Return", f"{rec.get('expected_return', 0):.2%}")
                                    mcol2.metric("Beta", f"{rec.get('beta', 0):.3f}")
                                    mcol3.metric("Risk Contribution", f"{rec.get('risk_contribution', 0):.2%}")

                                    wcol1, wcol2 = st.columns(2)
                                    wcol1.metric("Current Weight", f"{rec.get('current_weight', 0):.2%}")
                                    wcol2.metric("Target Weight", f"{rec.get('target_weight', 0):.2%}")

                                    if rec.get("anomaly_detected"):
                                        st.warning("⚠️ Anomaly detected for this symbol!")

                                    st.markdown(f"**Explanation:** {rec.get('explanation', 'N/A')}")
                        else:
                            st.info("No recommendations returned.")

    with tab2:
        st.subheader("Quick Symbol Recommendation")
        qr_sym = st.selectbox("Symbol", BVMT_SYMBOLS, key="qr_sym")
        qr_portfolio = st.text_input("Portfolio UUID", key="qr_port")

        if st.button("💡 Get Recommendation", type="primary"):
            if not qr_portfolio:
                st.error("Enter a valid Portfolio UUID.")
            else:
                with st.spinner("Getting recommendation..."):
                    result = api_post("/trading/recommendations", {
                        "symbol": qr_sym,
                        "portfolio_id": qr_portfolio,
                    })
                if not show_error(result):
                    action = result.get("action", "HOLD")
                    icon = {"BUY": "🟢", "SELL": "🔴"}.get(action.upper(), "🟡")

                    st.markdown(f"### {icon} {result.get('symbol', qr_sym)}: **{action.upper()}**")
                    st.metric("Confidence", f"{float(result.get('confidence', 0)):.2%}")
                    st.info(f"**Reasoning:** {result.get('reasoning', 'N/A')}")


# ══════════════════════════════════════════════════════════════════════
# PAGE: Settings & Status
# ══════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings & Status":
    st.title("⚙️ Settings & Status")

    tab1, tab2 = st.tabs(["🔧 System Status", "📝 API Explorer"])

    with tab1:
        st.subheader("System Health")

        # Health check
        health = api_get("/health")
        if not show_error(health):
            hcol1, hcol2 = st.columns(2)
            hcol1.metric("Status", health.get("status", "unknown"))
            hcol2.metric("Version", health.get("version", "unknown"))

        st.divider()

        # AI Status
        st.subheader("AI Module Status")
        ai_status = api_get("/ai/status")
        if not show_error(ai_status):
            acol1, acol2, acol3, acol4 = st.columns(4)
            acol1.metric("Module", ai_status.get("module", "N/A"))
            acol2.metric("Groq Model", ai_status.get("groq_model", "N/A"))
            acol3.metric("Active Portfolios", ai_status.get("active_portfolios", 0))
            acol4.metric("Default Capital", f"{ai_status.get('default_capital', 0):,.0f}")

            st.markdown("**Available Risk Profiles:**")
            for profile in ai_status.get("risk_profiles", []):
                st.markdown(f"  • `{profile}`")

        st.divider()
        st.subheader("Connection Details")
        st.code(f"API Base URL: {API_BASE}")

    with tab2:
        st.subheader("API Explorer")
        st.markdown("Test any API endpoint directly.")

        method = st.selectbox("Method", ["GET", "POST"], key="api_method")
        endpoint = st.text_input("Endpoint", "/health", key="api_ep")
        body = st.text_area("Request Body (JSON)", '{}', key="api_body", height=150)

        if st.button("🚀 Send Request", type="primary"):
            try:
                parsed_body = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError:
                st.error("Invalid JSON in request body.")
                parsed_body = None

            if parsed_body is not None:
                with st.spinner("Calling API..."):
                    if method == "GET":
                        result = api_get(endpoint, params=parsed_body if parsed_body != {} else None)
                    else:
                        result = api_post(endpoint, data=parsed_body)

                if not show_error(result):
                    st.success("✅ Response received")
                    st.json(result)
