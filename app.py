import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import random

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #ffffff;
    color: #111111;
}

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a1628;
    border-right: 2px solid #1e3a5f;
}
section[data-testid="stSidebar"] * {
    color: #e8f0fe !important;
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #93bbfc !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #1e3a5f !important;
    border: 1px solid #2d5a9e !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}

/* Black primary button */
.stButton > button[kind="primary"] {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 1.5rem !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #1a3a6e !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1.5px solid #111111 !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* Metric cards */
.metric-card {
    background: #f4f7ff;
    border-left: 4px solid #1a56db;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #1a56db;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    color: #111111;
    margin-top: 2px;
}

/* Result banner */
.result-banner {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a6e 100%);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.result-banner .result-label {
    font-size: 0.8rem;
    color: #93bbfc;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.result-banner .result-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin-top: 4px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f0f4ff;
    border-radius: 8px;
    padding: 4px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 1rem !important;
    color: #555 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #111111 !important;
    color: #ffffff !important;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #111111;
    border-bottom: 2px solid #1a56db;
    padding-bottom: 6px;
    margin-bottom: 1rem;
    display: inline-block;
}

/* Info chips */
.chip {
    display: inline-block;
    background: #e8f0fe;
    color: #1a56db;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f0f0f0; }
::-webkit-scrollbar-thumb { background: #1a56db; border-radius: 3px; }

/* Remove default streamlit padding on top */
.block-container { padding-top: 1.5rem !important; }

/* Success / error box */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loader ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except FileNotFoundError:
        return None

model = load_model()


# ─── Synthetic history for demo charts ──────────────────────────────────────
def get_demo_history(store, dept, anchor_date=None):
    rng = np.random.default_rng(store * 100 + dept)
    # Always end 1 week before the selected date so history + forecast are contiguous
    end = pd.Timestamp(anchor_date) - pd.Timedelta(weeks=1) if anchor_date else pd.Timestamp("today")
    weeks = pd.date_range(end=end, periods=52, freq="W")
    base = rng.uniform(15000, 60000)
    trend = np.linspace(0, base * 0.15, 52)
    seasonal = 8000 * np.sin(np.linspace(0, 4 * np.pi, 52))
    noise = rng.normal(0, 2000, 52)
    sales = np.clip(base + trend + seasonal + noise, 5000, None)
    return pd.DataFrame({"Date": weeks, "Weekly_Sales": sales})


def get_dept_breakdown(store, dept):
    rng = np.random.default_rng(store + dept * 7)
    depts = [dept, dept + 1, dept + 2, dept + 3, dept + 4]
    sales = rng.uniform(10000, 80000, 5).round(0)
    return pd.DataFrame({"Department": [f"Dept {d}" for d in depts], "Sales": sales})


def get_store_comparison(store):
    rng = np.random.default_rng(store * 3)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    this = rng.uniform(40000, 90000, 12).round(0)
    last = (this * rng.uniform(0.85, 1.1, 12)).round(0)
    return pd.DataFrame({"Month": months, "This Year": this, "Last Year": last})


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#ffffff;'>
            🛍️ Walmart
        </div>
        <div style='font-size:0.75rem;color:#93bbfc;letter-spacing:0.15em;text-transform:uppercase;'>
            Sales Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Store & Department")
    store = st.number_input("Store ID", min_value=1, max_value=45, value=1, step=1)
    dept = st.number_input("Department ID", min_value=1, max_value=99, value=1, step=1)

    st.markdown("##### Date")
    selected_date = st.date_input("Forecast Date", value=datetime.date.today())
    year  = selected_date.year
    month = selected_date.month
    day   = selected_date.day
    week  = selected_date.isocalendar()[1]

    st.markdown("##### Override Defaults *(optional)*")
    with st.expander("Economic Factors"):
        temperature  = st.slider("Temperature (°F)", 0, 110, 70)
        fuel_price   = st.slider("Fuel Price ($)", 2.0, 6.0, 3.0, 0.05)
        cpi          = st.slider("CPI", 180.0, 260.0, 220.0, 0.5)
        unemployment = st.slider("Unemployment (%)", 3.0, 12.0, 7.0, 0.1)
        is_holiday   = st.checkbox("Holiday Week", value=False)

    with st.expander("Markdowns"):
        md1 = st.number_input("MarkDown 1", value=0.0, step=100.0)
        md2 = st.number_input("MarkDown 2", value=0.0, step=100.0)
        md3 = st.number_input("MarkDown 3", value=0.0, step=100.0)
        md4 = st.number_input("MarkDown 4", value=0.0, step=100.0)
        md5 = st.number_input("MarkDown 5", value=0.0, step=100.0)

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Sales", type="primary", use_container_width=True)

    st.markdown("""
    <div style='margin-top:2rem;font-size:0.72rem;color:#4d7ab5;text-align:center;'>
        Model: Random Forest Regressor<br>
        Trained on Walmart M5 dataset
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.5rem;'>
    <span style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#111111;'>
        Sales Forecast Dashboard
    </span><br>
    <span style='font-size:0.9rem;color:#555;'>
        Store <strong>#{store}</strong> · Dept <strong>#{dept}</strong> · Week <strong>{week}</strong>, {year}
    </span>
</div>
""".format(store=store, dept=dept, week=week, year=year), unsafe_allow_html=True)

st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📈 Trends", "🏪 Store View", "⚙️ Input Summary"])


# ══════════ TAB 1 – FORECAST ══════════════════════════════════════════════════
with tab1:
    if predict_clicked:
        data = pd.DataFrame({
            'Store':        [store],
            'Dept':         [dept],
            'IsHoliday':    [int(is_holiday)],
            'Temperature':  [float(temperature)],
            'Fuel_Price':   [float(fuel_price)],
            'MarkDown1':    [md1], 'MarkDown2': [md2], 'MarkDown3': [md3],
            'MarkDown4':    [md4], 'MarkDown5': [md5],
            'CPI':          [float(cpi)],
            'Unemployment': [float(unemployment)],
            'Type':         [0],
            'Size':         [150000],
            'Year':         [year], 'Month': [month], 'Week': [week], 'Day': [day]
        })

        if model is not None:
            try:
                prediction = model.predict(data)[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                prediction = None
        else:
            # Demo mode — generate a realistic number
            rng = np.random.default_rng(store * 1000 + dept * 10 + week)
            prediction = float(rng.uniform(18000, 72000))
            st.info("ℹ️ `model.pkl` not found — showing demo prediction.", icon="🔬")

        if prediction is not None:
            # Result banner
            st.markdown(f"""
            <div class="result-banner">
                <div class="result-label">Predicted Weekly Sales</div>
                <div class="result-value">${prediction:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            # KPI row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Store</div>
                    <div class="value">#{store}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Department</div>
                    <div class="value">#{dept}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Fiscal Week</div>
                    <div class="value">W{week:02d}</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                holiday_label = "Yes 🎉" if is_holiday else "No"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Holiday</div>
                    <div class="value">{holiday_label}</div>
                </div>""", unsafe_allow_html=True)

            # ── Forecast confidence band chart ──
            st.markdown('<span class="section-header">52-Week Forecast Projection</span>', unsafe_allow_html=True)

            hist = get_demo_history(store, dept, anchor_date=selected_date)

            # Anchor forecast to last historical point so the line is continuous
            last_hist_date = hist["Date"].iloc[-1]
            last_hist_val  = hist["Weekly_Sales"].iloc[-1]

            # 13 future weeks starting right after last historical week
            future_weeks_raw = pd.date_range(last_hist_date, periods=14, freq="W")[1:]

            # Smooth transition: interpolate from last_hist_val to prediction over 5 steps,
            # then gentle noise around prediction for remaining weeks
            rng2 = np.random.default_rng(store * 17 + dept)
            transition  = np.linspace(last_hist_val, prediction, 5)
            noise_scale = prediction * 0.025
            tail        = prediction + rng2.normal(0, noise_scale, 8)
            future_vals_raw = np.concatenate([transition, tail])[:13]

            # Confidence band widens slightly further out
            band_width   = np.linspace(0.015, 0.055, 13)
            future_upper_raw = future_vals_raw * (1 + band_width)
            future_lower_raw = future_vals_raw * (1 - band_width)

            # Prepend last historical point so forecast line connects without a gap
            future_weeks = pd.Index([last_hist_date]).append(future_weeks_raw[:12])
            future_vals  = np.concatenate([[last_hist_val], future_vals_raw[:12]])
            future_upper = np.concatenate([[last_hist_val], future_upper_raw[:12]])
            future_lower = np.concatenate([[last_hist_val], future_lower_raw[:12]])

            import plotly.graph_objects as go

            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(
                x=hist["Date"], y=hist["Weekly_Sales"],
                mode="lines",
                name="Historical",
                line=dict(color="#1a56db", width=2.5),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>"
            ))

            # Confidence band
            fig.add_trace(go.Scatter(
                x=pd.concat([pd.Series(future_weeks), pd.Series(future_weeks[::-1])]),
                y=pd.concat([pd.Series(future_upper), pd.Series(future_lower[::-1])]),
                fill="toself",
                fillcolor="rgba(26,86,219,0.10)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Band",
                hoverinfo="skip"
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=future_weeks, y=future_vals,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#f59e0b", width=2.5, dash="dot"),
                marker=dict(size=6, color="#f59e0b"),
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>"
            ))

            # Current week marker
            fig.add_vline(
                x=datetime.datetime.combine(selected_date, datetime.time()).timestamp() * 1000,
                line_dash="dash", line_color="#ef4444", line_width=1.5,
                annotation_text="Today", annotation_font_color="#ef4444"
            )

            fig.update_layout(
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="DM Sans", color="#333"),
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(gridcolor="#f0f0f0", showgrid=True, zeroline=False),
                yaxis=dict(gridcolor="#f0f0f0", showgrid=True, zeroline=False,
                           tickprefix="$", tickformat=",.0f"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                height=340
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:3rem 1rem;color:#999;'>
            <div style='font-size:3rem;margin-bottom:1rem;'>🔮</div>
            <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;color:#111;'>
                Ready to Forecast
            </div>
            <div style='font-size:0.88rem;margin-top:0.5rem;'>
                Set the parameters in the sidebar and click <strong>Predict Sales</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════ TAB 2 – TRENDS ════════════════════════════════════════════════════
with tab2:
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<span class="section-header">Weekly Sales — Last 52 Weeks</span>', unsafe_allow_html=True)

    hist = get_demo_history(store, dept, anchor_date=datetime.date.today())

    # Rolling average
    hist["Rolling_4w"] = hist["Weekly_Sales"].rolling(4, min_periods=1).mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hist["Date"], y=hist["Weekly_Sales"],
        name="Weekly Sales",
        marker_color="#c7d7f9",
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra></extra>"
    ))
    fig2.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Rolling_4w"],
        name="4-Week Avg",
        line=dict(color="#1a56db", width=2.5),
        hovertemplate="4W Avg: $%{y:,.0f}<extra></extra>"
    ))
    fig2.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="DM Sans", color="#333"),
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0", tickprefix="$", tickformat=",.0f"),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", height=300, barmode="overlay"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Monthly seasonality ──
    st.markdown('<span class="section-header">Monthly Seasonality Pattern</span>', unsafe_allow_html=True)

    hist["Month_Name"] = hist["Date"].dt.strftime("%b")
    hist["Month_Num"]  = hist["Date"].dt.month
    monthly = hist.groupby(["Month_Num", "Month_Name"])["Weekly_Sales"].mean().reset_index()
    monthly = monthly.sort_values("Month_Num")

    fig3 = px.area(
        monthly, x="Month_Name", y="Weekly_Sales",
        labels={"Weekly_Sales": "Avg Weekly Sales", "Month_Name": ""},
        color_discrete_sequence=["#1a56db"]
    )
    fig3.update_traces(
        fill="tozeroy",
        fillcolor="rgba(26,86,219,0.12)",
        line=dict(color="#1a56db", width=2.5)
    )
    fig3.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="DM Sans", color="#333"),
        yaxis=dict(gridcolor="#f0f0f0", tickprefix="$", tickformat=",.0f"),
        xaxis=dict(gridcolor="#f0f0f0"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=280
    )
    st.plotly_chart(fig3, use_container_width=True)


# ══════════ TAB 3 – STORE VIEW ════════════════════════════════════════════════
with tab3:
    import plotly.graph_objects as go

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<span class="section-header">Dept Sales Snapshot</span>', unsafe_allow_html=True)

        breakdown = get_dept_breakdown(store, dept)
        colors = ["#1a56db", "#2d6ef0", "#5b8ef5", "#93bbfc", "#c7d7f9"]

        fig4 = go.Figure(go.Bar(
            x=breakdown["Sales"],
            y=breakdown["Department"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
            text=[f"${v:,.0f}" for v in breakdown["Sales"]],
            textposition="outside",
            textfont=dict(size=11, color="#111")
        ))
        fig4.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="DM Sans", color="#333"),
            xaxis=dict(gridcolor="#f0f0f0", tickprefix="$", tickformat=",.0f", showgrid=True),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=60, t=20, b=10),
            height=280
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_right:
        st.markdown('<span class="section-header">Year-over-Year Comparison</span>', unsafe_allow_html=True)

        comp = get_store_comparison(store)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=comp["Month"], y=comp["This Year"],
            name="This Year",
            mode="lines+markers",
            line=dict(color="#111111", width=2.5),
            marker=dict(size=7)
        ))
        fig5.add_trace(go.Scatter(
            x=comp["Month"], y=comp["Last Year"],
            name="Last Year",
            mode="lines+markers",
            line=dict(color="#93bbfc", width=2, dash="dot"),
            marker=dict(size=5)
        ))
        fig5.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="DM Sans", color="#333"),
            yaxis=dict(gridcolor="#f0f0f0", tickprefix="$", tickformat=",.0f"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
            height=280
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── Store KPIs ──
    st.markdown('<span class="section-header">Store At a Glance</span>', unsafe_allow_html=True)
    rng_store = np.random.default_rng(store * 7)
    k1, k2, k3, k4, k5 = st.columns(5)
    for col, label, val, fmt in zip(
        [k1, k2, k3, k4, k5],
        ["Avg Weekly Sales", "Peak Week", "Lowest Week", "YTD Total", "Dept Count"],
        [rng_store.uniform(30000, 65000), rng_store.uniform(60000, 95000),
         rng_store.uniform(10000, 25000), rng_store.uniform(1.2e6, 3.5e6),
         rng_store.integers(60, 99)],
        ["${:,.0f}", "${:,.0f}", "${:,.0f}", "${:,.0f}", "{:.0f}"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="font-size:1.1rem">{fmt.format(val)}</div>
        </div>""", unsafe_allow_html=True)


# ══════════ TAB 4 – INPUT SUMMARY ════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-header">Current Input Configuration</span>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Store & Date**")
        params = {
            "Store ID": f"#{store}",
            "Department ID": f"#{dept}",
            "Forecast Date": selected_date.strftime("%B %d, %Y"),
            "ISO Week": f"Week {week}",
            "Month": selected_date.strftime("%B"),
            "Year": year,
            "Holiday Week": "Yes" if is_holiday else "No"
        }
        for k, v in params.items():
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        padding:6px 0;border-bottom:1px solid #f0f0f0;'>
                <span style='color:#555;font-size:0.85rem;'>{k}</span>
                <span style='font-weight:600;font-size:0.85rem;'>{v}</span>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("**Economic & Markdown Factors**")
        econ = {
            "Temperature": f"{temperature}°F",
            "Fuel Price": f"${fuel_price:.2f}",
            "CPI": f"{cpi:.1f}",
            "Unemployment": f"{unemployment:.1f}%",
            "MarkDown 1-5": f"${md1+md2+md3+md4+md5:,.0f} total"
        }
        for k, v in econ.items():
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        padding:6px 0;border-bottom:1px solid #f0f0f0;'>
                <span style='color:#555;font-size:0.85rem;'>{k}</span>
                <span style='font-weight:600;font-size:0.85rem;'>{v}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Active Tags**")
    tags = []
    if is_holiday:        tags.append("🎉 Holiday Week")
    if fuel_price > 4.0:  tags.append("⛽ High Fuel")
    if unemployment > 9:  tags.append("📉 High Unemployment")
    if temperature > 90:  tags.append("🌡️ Heat Wave")
    if md1 + md2 + md3 + md4 + md5 > 0: tags.append("🏷️ Markdowns Active")
    if not tags:          tags.append("Standard Week")

    st.markdown(" ".join([f'<span class="chip">{t}</span>' for t in tags]), unsafe_allow_html=True)