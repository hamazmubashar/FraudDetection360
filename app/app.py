import streamlit as st
import streamlit.components.v1 as components
import time
import plotly.graph_objects as go
import pandas as pd
import os
import joblib
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# ---------------------------------------------------------
# Model LOADING (config-driven path, cached with hourly refresh)
# ---------------------------------------------------------
MODEL_PATH = os.getenv("FRAUD_MODEL_PATH", os.path.join(REPO_ROOT, "artifacts", "final_fraud_model.pkl"))

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------------------------------------------------
# Defaults LOADING (config-driven path, cached with hourly refresh)
# ---------------------------------------------------------
DEFAULTS_PATH = os.getenv("FRAUD_DEFAULTS_PATH", os.path.join(REPO_ROOT, "artifacts", "feature_defaults.json"))

@st.cache_data
def load_defaults():
    with open(DEFAULTS_PATH, 'r') as f:
        return json.load(f)

feature_defaults_raw = load_defaults()
feature_defaults = feature_defaults_raw["defaults"]
categorical_categories = feature_defaults_raw["categorical_categories"]

# ---------------------------------------------------------
# DATA LOADING (config-driven path, cached with hourly refresh)
# ---------------------------------------------------------
DATA_PATH = os.getenv("FRAUD_DATA_PATH", os.path.join(REPO_ROOT, "datasets", "processed", "fraud_dashboard_export.csv"))

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()


# ---------------------------------------------------------
# 1. PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="FraudDetection360",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# 2. THEME & STYLING
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0F1729;
    color: #FFFFFF;
}
.intro-text {
    color: #94A3B8;
    font-size: 1.15rem;
    line-height: 1.7;
    max-width: 720px;
    margin: 18px auto 28px auto;
    text-align: center !important;
}
.badge-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 12px;
    margin-bottom: 35px;
    flex-wrap: wrap;
}
.badge-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #E2E8F0;
    padding: 8px 18px;
    border-radius: 20px;
    font-size: 0.88rem;
    font-weight: 500;
    letter-spacing: 0.3px;
}
.badge-teal {
    color: #4ECDC4;
    border-color: rgba(78, 205, 196, 0.4);
    background: rgba(78, 205, 196, 0.08);
}
.badge-coral {
    color: #FF6B6B;
    border-color: rgba(255, 107, 107, 0.4);
    background: rgba(255, 107, 107, 0.08);
}
div.stButton > button {
    background: linear-gradient(135deg, #4ECDC4 0%, #3BABA3 100%) !important;
    color: #0F1729 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px 36px !important;
    border-radius: 12px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(78, 205, 196, 0.35) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(78, 205, 196, 0.5) !important;
}
.kpi-card {
    background: rgba(26, 39, 64, 0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.kpi-label {
    color: #8B9BB8;
    font-size: 0.95rem;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
}
.kpi-white { color: #FFFFFF; }
.kpi-coral { color: #FF6B6B; }
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #0F1729;
}
::-webkit-scrollbar-thumb {
    background: #2A3A5C;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4ECDC4;
}
.status-pulse {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    color: #8B9BB8;
}
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4ECDC4;
    box-shadow: 0 0 0 0 rgba(78, 205, 196, 0.7);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(78, 205, 196, 0.6); }
    70% { box-shadow: 0 0 0 8px rgba(78, 205, 196, 0); }
    100% { box-shadow: 0 0 0 0 rgba(78, 205, 196, 0); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. SESSION STATE LOGIC
# ---------------------------------------------------------
if "entered_app" not in st.session_state:
    st.session_state.entered_app = False
if "typing_completed" not in st.session_state:
    st.session_state.typing_completed = False

# ---------------------------------------------------------
# 4. WELCOME SCREEN
# ---------------------------------------------------------
if not st.session_state.entered_app:

    _, col_center, _ = st.columns([0.15, 0.7, 0.15])

    with col_center:
        title_placeholder = st.empty()
        full_title = "🛡️ Welcome to FraudDetection360"

        if not st.session_state.typing_completed:
            displayed = ""
            for letter in full_title:
                displayed += letter
                title_placeholder.markdown(
f"<h1 style='text-align: center; color: #FFFFFF; font-size: 2.7rem; font-weight: 700; margin-top: 40px; margin-bottom: 0;'>{displayed}<span style='color: #4ECDC4;'>|</span></h1>",
                    unsafe_allow_html=True
                )
                time.sleep(0.07)

            title_placeholder.markdown(
f"<h1 style='text-align: center; color: #FFFFFF; font-size: 2.7rem; font-weight: 700; margin-top: 40px; margin-bottom: 0;'>{full_title}</h1>",
                unsafe_allow_html=True
            )
            st.session_state.typing_completed = True
        else:
            title_placeholder.markdown(
f"<h1 style='text-align: center; color: #FFFFFF; font-size: 2.7rem; font-weight: 700; margin-top: 40px; margin-bottom: 0;'>{full_title}</h1>",
                unsafe_allow_html=True
            )

        st.markdown(
"""<div style="width: 100%; text-align: center;">
<p class="intro-text" style="text-align: center; margin-left: auto; margin-right: auto;">
A real-time fraud risk scoring engine trained on <b>590,540 e-commerce transactions</b>.
Optimized with dynamic cost-threshold modeling to minimize monetary losses while protecting good customers.
</p>
</div>""",
            unsafe_allow_html=True
        )

        st.markdown(
"""<div class="badge-container">
<span class="badge-pill badge-teal">⚡ LightGBM Model (0.90 ROC-AUC)</span>
<span class="badge-pill">$79.7M Dataset Volume</span>
<span class="badge-pill badge-coral">91.1% Risk Reduction</span>
</div>""",
            unsafe_allow_html=True
        )

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1.2, 1])
        with btn_col2:
            if st.button("Explore Risk Dashboard →", use_container_width=True):
                st.session_state.entered_app = True
                st.rerun()

        st.markdown(
"""<p style="text-align: center; color: #4A5A78; font-size: 0.85rem; margin-top: 24px;">
Developed by Hamaz Mubashar
</p>""",
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# 5. MAIN APP DASHBOARD
# ---------------------------------------------------------
else:
    with st.sidebar:
        st.markdown("### Transaction Scoring Simulator")
        input_amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=10.0)
        input_product = st.selectbox("Product Type", options=["W", "C", "R", "H", "S"])
        input_card = st.selectbox("Card Type", options=["debit", "credit"])
        input_hour = st.slider("Hour of Day", 0, 23, 12)
        input_device = st.selectbox("Device Info", options=["Untracked", "mobile", "desktop"])

        st.markdown("---")
        fp_cost = st.slider("False Positive Cost ($)", min_value=5, max_value=50, value=10, step=1)
        # Interpolate optimal threshold from Notebook 06 sensitivity analysis
        _known_costs = [5, 10, 20, 30, 50]
        _known_thresholds = [0.40, 0.55, 0.70, 0.75, 0.85]
        import numpy as np
        current_threshold = float(np.interp(fp_cost, _known_costs, _known_thresholds))
        st.caption(f"Optimal threshold for ${fp_cost} FP cost: **{current_threshold:.2f}**")

        score_clicked = st.button("Score Transaction", use_container_width=True)
        if score_clicked:
            # Start with the defaults for every column
            input_row = feature_defaults.copy()

            # Override with the user's actual inputs
            input_row['TransactionAmt'] = input_amount
            input_row['ProductCD'] = input_product
            input_row['card6'] = input_card
            input_row['hour_of_day'] = float(input_hour)
            input_row['hour_bucket'] = int(input_hour)
            input_row['DeviceType'] = None if input_device == "Untracked" else input_device
            input_row['has_identity'] = 0 if input_device == "Untracked" else 1

            # Convert to a single-row DataFrame matching model's expected columns
            input_df = pd.DataFrame([input_row])

            # Rebuild categorical dtypes to EXACTLY match what LightGBM saw during training
            for col, cats in categorical_categories.items():
                if col in input_df.columns:
                    input_df[col] = pd.Categorical(input_df[col], categories=cats)

            input_df = input_df[model.feature_name_]

            # Predict probability of fraud
            fraud_probability = model.predict_proba(input_df)[0][1]

            st.session_state.last_score = fraud_probability
            st.session_state.last_inputs = {
                "hour": input_hour,
                "device": input_device,
                "card": input_card,
                "product": input_product,
            }

        # ---- Step D & E: Show result, decision, and "Why?" panel ----
        if "last_score" in st.session_state:
            st.markdown("---")
            prob = st.session_state.last_score
            last = st.session_state.last_inputs
            threshold = current_threshold
            gauge_color = "#FF6B6B" if prob >= threshold else "#4ECDC4"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'color': '#FFFFFF', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8B9BB8', 'tickfont': {'color': '#8B9BB8'}},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#1A2740",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, threshold * 100], 'color': 'rgba(78, 205, 196, 0.15)'},
                        {'range': [threshold * 100, 100], 'color': 'rgba(255, 107, 107, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': '#FFFFFF', 'width': 3},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0F1729",
                font={'color': '#E2E8F0'},
                height=220,
                margin=dict(l=20, r=20, t=20, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={'staticPlot': True})

            threshold = current_threshold
            if prob >= threshold:
                st.markdown(
f"""<div style="background-color: #FF6B6B; color: #0F1729; padding: 16px; border-radius: 10px; text-align: center; font-weight: 700; font-size: 1.3rem;">
BLOCK
</div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
f"""<div style="background-color: #4ECDC4; color: #0F1729; padding: 16px; border-radius: 10px; text-align: center; font-weight: 700; font-size: 1.3rem;">
APPROVE
</div>""",
                    unsafe_allow_html=True
                )

            # Why panel — based on the ACTUAL inputs used for this score
            reasons = []
            if 6 <= last["hour"] <= 9:
                reasons.append(f"{last['hour']}:00 falls in the known 6-9AM high-risk window")
            if last["device"] == "mobile":
                reasons.append("Mobile transactions show the highest device-based fraud rate (10.2%)")
            if last["card"] == "credit":
                reasons.append("Credit cards show higher fraud rate (6.7%) than debit (2.4%)")
            if last["product"] == "C":
                reasons.append("Product Category 'C' has the highest fraud rate (11.7%) in historical data")
            if not reasons:
                reasons.append("No major known risk patterns matched — score driven by other factors")

            st.markdown("**Why?**")
            for r in reasons:
                st.markdown(f"- {r}")

    # ---- Apply all active cross-filters (AND logic across all three) ----
    filtered_df = df.copy()
    if "selected_product" in st.session_state:
        filtered_df = filtered_df[filtered_df['ProductCD'] == st.session_state.selected_product]
    if "selected_card" in st.session_state:
        filtered_df = filtered_df[filtered_df['card_type'] == st.session_state.selected_card]
    if "selected_device" in st.session_state:
        filtered_df = filtered_df[filtered_df['DeviceType'].fillna('Untracked') == st.session_state.selected_device]

    st.markdown(
"""<h2 style="color: #FFFFFF; margin-bottom: 4px;">🛡️ FraudDetection360 Dashboard</h2>
<p style="color: #8B9BB8; margin-top: 0;">
<span class="status-pulse"><span class="status-dot"></span>Real-Time Decision Engine Active</span>
</p>""",
        unsafe_allow_html=True
    )

    if st.sidebar.button("← Back to Welcome Screen"):
        st.session_state.entered_app = False
        st.session_state.typing_completed = False
        st.rerun()

    # Show a "clear all filters" option if any filter is active
    active_filters = [k for k in ["selected_product", "selected_card", "selected_device"] if k in st.session_state]
    if active_filters:
        if st.button("✕ Clear all filters"):
            for k in active_filters:
                del st.session_state[k]
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- KPI ROW (live, respects all active filters, animated count-up) ----
    live_fraud_rate = round(filtered_df['isFraud'].mean() * 100, 1) if len(filtered_df) else 0
    live_total_volume = round(filtered_df['TransactionAmt'].sum() / 1_000_000, 2)
    live_fraud_exposure = round(filtered_df[filtered_df['isFraud'] == 1]['TransactionAmt'].sum() / 1_000_000, 2)

    kpi_html = f"""
    <div style="font-family: 'Source Sans Pro', sans-serif; display:flex; gap:16px;">
      <div class="kc" style="flex:1; background: rgba(26,39,64,0.55); backdrop-filter: blur(12px); border-radius:14px; padding:20px 24px; border:1px solid rgba(255,255,255,0.12); box-shadow:0 4px 24px rgba(0,0,0,0.25);">
        <div style="color:#8B9BB8; font-size:0.95rem; margin-bottom:6px;">Fraud Rate</div>
        <div id="v1" style="color:#FF6B6B; font-size:2.2rem; font-weight:700;">0%</div>
      </div>
      <div class="kc" style="flex:1; background: rgba(26,39,64,0.55); backdrop-filter: blur(12px); border-radius:14px; padding:20px 24px; border:1px solid rgba(255,255,255,0.12); box-shadow:0 4px 24px rgba(0,0,0,0.25);">
        <div style="color:#8B9BB8; font-size:0.95rem; margin-bottom:6px;">Total Volume</div>
        <div id="v2" style="color:#FFFFFF; font-size:2.2rem; font-weight:700;">$0M</div>
      </div>
      <div class="kc" style="flex:1; background: rgba(26,39,64,0.55); backdrop-filter: blur(12px); border-radius:14px; padding:20px 24px; border:1px solid rgba(255,255,255,0.12); box-shadow:0 4px 24px rgba(0,0,0,0.25);">
        <div style="color:#8B9BB8; font-size:0.95rem; margin-bottom:6px;">Fraud Exposure</div>
        <div id="v3" style="color:#FF6B6B; font-size:2.2rem; font-weight:700;">$0M</div>
      </div>
    </div>
    <script>
      function countUp(id, target, suffix, duration) {{
        const el = document.getElementById(id);
        const start = performance.now();
        function tick(now) {{
          const progress = Math.min((now - start) / duration, 1);
          const value = (target * progress).toFixed(1);
          el.textContent = (suffix.startsWith('$') ? '$' : '') + value + suffix.replace('$', '');
          if (progress < 1) requestAnimationFrame(tick);
        }}
        requestAnimationFrame(tick);
      }}
      countUp('v1', {live_fraud_rate}, '%', 900);
      countUp('v2', {live_total_volume}, '$M', 900);
      countUp('v3', {live_fraud_exposure}, '$M', 900);
    </script>
    """
    components.html(kpi_html, height=140)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #FFFFFF;'>Key Findings</h4>", unsafe_allow_html=True)

    chart1, chart2, chart3 = st.columns(3)

    # ---------- CHART 1: Product Analysis (always shows ALL products; click to filter) ----------
    with chart1:
        product_summary = df.groupby('ProductCD').agg(
            fraud_rate=('isFraud', lambda x: x.mean() * 100),
            dollar_impact=('TransactionAmt', lambda x: (x * df.loc[x.index, 'isFraud']).sum() / 1_000_000)
        ).reset_index().sort_values('fraud_rate', ascending=False)

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=product_summary['ProductCD'], y=product_summary['fraud_rate'],
            name="Fraud Rate %", marker_color="#FF6B6B"
        ))
        fig1.add_trace(go.Bar(
            x=product_summary['ProductCD'], y=product_summary['dollar_impact'],
            name="Dollar Impact ($M)", marker_color="#4ECDC4", yaxis="y2"
        ))
        fig1.update_layout(
            title="Fraud by Product Category (click to filter)",
            plot_bgcolor="#1A2740", paper_bgcolor="#1A2740", font=dict(color="#E2E8F0"),
            yaxis=dict(title="Fraud Rate %"),
            yaxis2=dict(title="Dollar Impact ($M)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2),
            height=350, margin=dict(l=10, r=10, t=40, b=10)
        )
        selection1 = st.plotly_chart(fig1, use_container_width=True, on_select="rerun", key="chart1")
        if selection1 and selection1.selection and selection1.selection.points:
            st.session_state.selected_product = selection1.selection.points[0]["x"]
            st.rerun()
        if "selected_product" in st.session_state:
            st.caption(f"Filtered by: Product Category = {st.session_state.selected_product}")

    # ---------- CHART 2: Time-of-Day Risk (respects filters) ----------
    with chart2:
        hourly_summary = filtered_df.groupby('hour_bucket')['isFraud'].mean().reset_index()
        hourly_summary['fraud_rate'] = hourly_summary['isFraud'] * 100

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hourly_summary['hour_bucket'], y=hourly_summary['fraud_rate'], mode="lines",
            line=dict(color="#FF6B6B", width=3),
            fill="tozeroy", fillcolor="rgba(255,107,107,0.15)"
        ))
        fig2.update_layout(
            title="Time-of-Day Risk Window",
            plot_bgcolor="#1A2740", paper_bgcolor="#1A2740", font=dict(color="#E2E8F0"),
            xaxis=dict(title="Hour of Day"), yaxis=dict(title="Fraud Rate %"),
            height=350, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---------- CHART 3: Optimal Threshold Decision ----------
    with chart3:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                      0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        total_cost_k = [971.4, 720.8, 560.2, 453.4, 383.8, 337.9, 309.5, 291.2,
                         280.8, 274.2, 273.0, 278.6, 282.3, 296.1, 316.9, 340.7,
                         365.1, 400.9]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=thresholds, y=total_cost_k, mode="lines+markers",
            line=dict(color="#4ECDC4", width=3), marker=dict(size=5)
        ))
        fig3.add_vline(x=current_threshold, line_dash="dash", line_color="#FF6B6B",
                       annotation_text=f"Optimal: {current_threshold:.2f}", annotation_font_color="#FF6B6B")
        fig3.update_layout(
            title="Optimal Threshold Decision",
            plot_bgcolor="#1A2740", paper_bgcolor="#1A2740", font=dict(color="#E2E8F0"),
            xaxis=dict(title="Threshold"), yaxis=dict(title="Total Cost ($K)"),
            height=350, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    chart4, chart5 = st.columns(2)

    # ---------- CHART 4: Card Type Risk (respects filters, clickable) ----------
    with chart4:
        card_summary = filtered_df.groupby('card_type')['isFraud'].mean().reset_index().dropna(subset=['card_type'])
        card_summary['fraud_rate'] = card_summary['isFraud'] * 100

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=card_summary['card_type'], y=card_summary['fraud_rate'],
            marker_color=["#FF6B6B" if v > card_summary['fraud_rate'].mean() else "#4ECDC4" for v in card_summary['fraud_rate']]
        ))
        fig4.update_layout(
            title="Card Type Risk (click to filter)",
            plot_bgcolor="#1A2740", paper_bgcolor="#1A2740", font=dict(color="#E2E8F0"),
            yaxis=dict(title="Fraud Rate %"),
            height=320, margin=dict(l=10, r=10, t=40, b=10)
        )
        selection4 = st.plotly_chart(fig4, use_container_width=True, on_select="rerun", key="chart4")
        if selection4 and selection4.selection and selection4.selection.points:
            st.session_state.selected_card = selection4.selection.points[0]["x"]
            st.rerun()
        if "selected_card" in st.session_state:
            st.caption(f"Filtered by: Card Type = {st.session_state.selected_card}")

    # ---------- CHART 5: Device Type Risk (respects filters, clickable) ----------
    with chart5:
        device_summary = filtered_df.copy()
        device_summary['DeviceType'] = device_summary['DeviceType'].fillna('Untracked')
        device_summary = device_summary.groupby('DeviceType')['isFraud'].mean().reset_index()
        device_summary['fraud_rate'] = device_summary['isFraud'] * 100

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=device_summary['DeviceType'], y=device_summary['fraud_rate'],
            marker_color=["#FF6B6B" if v > device_summary['fraud_rate'].mean() else "#4ECDC4" for v in device_summary['fraud_rate']]
        ))
        fig5.update_layout(
            title="Device Type Risk (click to filter)",
            plot_bgcolor="#1A2740", paper_bgcolor="#1A2740", font=dict(color="#E2E8F0"),
            yaxis=dict(title="Fraud Rate %"),
            height=320, margin=dict(l=10, r=10, t=40, b=10)
        )
        selection5 = st.plotly_chart(fig5, use_container_width=True, on_select="rerun", key="chart5")
        if selection5 and selection5.selection and selection5.selection.points:
            st.session_state.selected_device = selection5.selection.points[0]["x"]
            st.rerun()
        if "selected_device" in st.session_state:
            st.caption(f"Filtered by: Device Type = {st.session_state.selected_device}")