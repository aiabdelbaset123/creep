"""
╔══════════════════════════════════════════════════════════════════╗
║       Stress-Strain-Temperature Material Analysis Platform       ║
║                  Advanced Engineering Dashboard                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Material Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Global ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Background ─── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #58a6ff;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 12px;
}

/* ─── Metric Cards ─── */
.metric-card {
    background: linear-gradient(145deg, #161b22, #21262d);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(88,166,255,0.15);
    border-color: #58a6ff;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 4px;
}
.blue   { color: #58a6ff; }
.green  { color: #3fb950; }
.orange { color: #f0883e; }
.purple { color: #bc8cff; }
.red    { color: #f85149; }
.cyan   { color: #39d0d8; }

/* ─── Section Headers ─── */
.section-header {
    background: linear-gradient(90deg, #58a6ff22, transparent);
    border-left: 3px solid #58a6ff;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 14px 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: 0.03em;
}

/* ─── Module title ─── */
.module-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e6edf3;
    margin-bottom: 4px;
}
.module-subtitle {
    font-size: 0.88rem;
    color: #8b949e;
    margin-bottom: 20px;
}

/* ─── Info Box ─── */
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.85rem;
    color: #c9d1d9;
    line-height: 1.6;
}
.info-box b { color: #58a6ff; }

/* ─── Table ─── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ─── Tabs ─── */
[data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #8b949e !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
[aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
}

/* ─── Selectbox / Slider ─── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label {
    color: #c9d1d9 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* ─── Divider ─── */
hr { border-color: #21262d !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

/* ─── Alert / Info ─── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────
TEMP_COLORS = {
    25: "#58a6ff",
    35: "#3fb950",
    45: "#f0883e",
    55: "#bc8cff",
    65: "#f85149",
    70: "#39d0d8",
    75: "#ffa657",
    80: "#ff7b72",
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,27,34,0.6)",
        font=dict(family="Inter", color="#c9d1d9", size=12),
        xaxis=dict(
            gridcolor="#21262d", linecolor="#30363d",
            tickfont=dict(color="#8b949e"),
            title_font=dict(color="#c9d1d9"),
            zerolinecolor="#30363d",
        ),
        yaxis=dict(
            gridcolor="#21262d", linecolor="#30363d",
            tickfont=dict(color="#8b949e"),
            title_font=dict(color="#c9d1d9"),
            zerolinecolor="#30363d",
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(color="#c9d1d9", size=11),
        ),
        hoverlabel=dict(
            bgcolor="#21262d",
            bordercolor="#30363d",
            font=dict(color="#e6edf3", family="Inter"),
        ),
        margin=dict(l=60, r=30, t=50, b=60),
    )
)

# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("stress_strain_tempData.csv", encoding="latin-1")
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "time (sec.)":     "time",
        "stress (Mpa)":    "stress",
        "strain (%)":      "strain",
        "resistance":      "resistance",
        "temperature (°C)":"temperature",
    }, inplace=True)
    df["stress_MPa"]   = df["stress"] / 1e6
    df["stress_abs"]   = df["stress"].abs() / 1e6
    df["temp_label"]   = df["temperature"].astype(str) + " °C"
    df["color"]        = df["temperature"].map(TEMP_COLORS)
    return df

df = load_data()
temps_all = sorted(df["temperature"].unique())

# ──────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────
def apply_template(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    return fig

def smooth(series, window=11, poly=2):
    if len(series) > window:
        return savgol_filter(series, window_length=window, polyorder=poly)
    return series

def fmt_metric(val, decimals=2, unit=""):
    return f"{val:,.{decimals}f}{' '+unit if unit else ''}"

def get_temp_df(temperature):
    return df[df["temperature"] == temperature].copy()

def compute_elasticity(sub):
    """Estimate Young's modulus from linear portion of stress-strain."""
    pos = sub[sub["strain"] > 0.001].copy()
    if len(pos) < 5:
        return np.nan, np.nan
    q25, q75 = pos["strain"].quantile(0.1), pos["strain"].quantile(0.4)
    linear = pos[(pos["strain"] >= q25) & (pos["strain"] <= q75)]
    if len(linear) < 3:
        return np.nan, np.nan
    slope, intercept, r, _, _ = stats.linregress(linear["strain"], linear["stress_abs"])
    return slope, r**2

def compute_material_props(sub):
    """Compute per-temperature material properties."""
    E, r2 = compute_elasticity(sub)
    max_stress  = sub["stress_abs"].max()
    max_strain  = sub["strain"].max()
    min_resist  = sub["resistance"].min()
    max_resist  = sub["resistance"].max()
    delta_r     = max_resist - min_resist
    mean_resist = sub["resistance"].mean()
    # UTS at peak
    uts_idx = sub["stress_abs"].idxmax()
    uts_strain = sub.loc[uts_idx, "strain"]
    return {
        "Young's Modulus (MPa)":  round(E, 1)        if not np.isnan(E) else "—",
        "R² (fit)":               round(r2, 4)        if not np.isnan(r2) else "—",
        "UTS (MPa)":              round(max_stress, 2),
        "Strain at UTS (%)":      round(uts_strain, 5),
        "Max Strain (%)":         round(max_strain, 5),
        "ΔResistance (Ω)":        round(delta_r, 4),
        "Mean Resistance (Ω)":    round(mean_resist, 3),
    }

@st.cache_data
def build_material_table():
    rows = []
    for t in temps_all:
        sub = get_temp_df(t)
        props = compute_material_props(sub)
        props["Temperature (°C)"] = t
        rows.append(props)
    return pd.DataFrame(rows).set_index("Temperature (°C)")

mat_table = build_material_table()

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Navigation")

    module = st.radio(
        "Select Module",
        [
            "🏠  Dashboard Overview",
            "📈  Stress–Strain Analysis",
            "⏱️  Time Series Explorer",
            "🌡️  Temperature Effects",
            "⚡  Resistance Analysis",
            "📊  Statistical Analysis",
            "🔧  Material Properties",
            "🔍  Raw Data Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("## ⚙️ Global Filters")

    selected_temps = st.multiselect(
        "Temperatures (°C)",
        options=temps_all,
        default=temps_all,
        format_func=lambda x: f"{x} °C",
    )
    if not selected_temps:
        selected_temps = temps_all

    smooth_data = st.checkbox("🔄 Apply Savitzky-Golay Smoothing", value=False)

    st.markdown("---")
    st.markdown("## 📁 Dataset Info")
    st.markdown(f"""
<div class="info-box">
<b>Rows:</b> {len(df):,}<br>
<b>Temperatures:</b> {len(temps_all)} levels<br>
<b>Time range:</b> 0 – 69.9 s<br>
<b>Sampling:</b> 0.1 s<br>
<b>Columns:</b> 5
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🧪 Material Testing Lab  \nv2.0 · Advanced Analysis Platform")

# Filter data by selected temps
dff = df[df["temperature"].isin(selected_temps)].copy()

# ══════════════════════════════════════════════════════════════════
# MODULE 1 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════
if module == "🏠  Dashboard Overview":
    st.markdown('<div class="module-title">🏠 Dashboard Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">High-level summary of the material testing dataset across all temperature conditions.</div>', unsafe_allow_html=True)

    # ── KPI Row ──
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpi_data = [
        (col1, "Total Data Points",   f"{len(df):,}",         "",                   "blue"),
        (col2, "Temperature Levels",  str(len(temps_all)),    "from 25–80 °C",       "green"),
        (col3, "Max Stress",          fmt_metric(df["stress_abs"].max(), 1, "MPa"), "peak absolute","orange"),
        (col4, "Max Strain",          fmt_metric(df["strain"].max()*100, 3, "%"),   "max recorded", "purple"),
        (col5, "Resistance Range",    fmt_metric(df["resistance"].max() - df["resistance"].min(), 3, "Ω"), "Δ over test", "cyan"),
        (col6, "Duration",            "69.9 s",               "per temperature",     "red"),
    ]
    for col, label, val, sub, clr in kpi_data:
        col.markdown(f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value {clr}">{val}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Stress vs Time overview | Strain vs Time overview ──
    st.markdown('<div class="section-header">📈 Multi-Temperature Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            y = smooth(sub["stress_abs"].values) if smooth_data else sub["stress_abs"].values
            fig.add_trace(go.Scatter(
                x=sub["time"], y=y,
                mode="lines", name=f"{t} °C",
                line=dict(color=TEMP_COLORS[t], width=1.5),
                hovertemplate=f"<b>{t} °C</b><br>Time: %{{x:.1f}} s<br>|Stress|: %{{y:.2f}} MPa<extra></extra>",
            ))
        fig.update_layout(
            title="Absolute Stress vs Time",
            xaxis_title="Time (s)", yaxis_title="|Stress| (MPa)",
            height=320, showlegend=True,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            y = smooth(sub["strain"].values) if smooth_data else sub["strain"].values
            fig.add_trace(go.Scatter(
                x=sub["time"], y=y * 100,
                mode="lines", name=f"{t} °C",
                line=dict(color=TEMP_COLORS[t], width=1.5),
                hovertemplate=f"<b>{t} °C</b><br>Time: %{{x:.1f}} s<br>Strain: %{{y:.4f}} %<extra></extra>",
            ))
        fig.update_layout(
            title="Strain vs Time",
            xaxis_title="Time (s)", yaxis_title="Strain (%)",
            height=320, showlegend=True,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Resistance overview | Stress distribution ──
    c3, c4 = st.columns(2)

    with c3:
        fig = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            y = smooth(sub["resistance"].values) if smooth_data else sub["resistance"].values
            fig.add_trace(go.Scatter(
                x=sub["time"], y=y,
                mode="lines", name=f"{t} °C",
                line=dict(color=TEMP_COLORS[t], width=1.5),
            ))
        fig.update_layout(
            title="Resistance vs Time",
            xaxis_title="Time (s)", yaxis_title="Resistance (Ω)",
            height=320, showlegend=True,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            fig.add_trace(go.Box(
                y=sub["stress_abs"],
                name=f"{t} °C",
                marker_color=TEMP_COLORS[t],
                boxmean='sd',
            ))
        fig.update_layout(
            title="Stress Distribution by Temperature",
            yaxis_title="|Stress| (MPa)",
            height=320, showlegend=False,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Material Table ──
    st.markdown('<div class="section-header">📋 Material Properties Summary</div>', unsafe_allow_html=True)
    display_mat = mat_table.loc[mat_table.index.isin(selected_temps)]
    st.dataframe(
        display_mat.style.background_gradient(cmap="Blues", axis=0),
        use_container_width=True, height=280,
    )


# ══════════════════════════════════════════════════════════════════
# MODULE 2 — STRESS–STRAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif module == "📈  Stress–Strain Analysis":
    st.markdown('<div class="module-title">📈 Stress–Strain Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">Classic engineering stress-strain curves with optional smoothing and polynomial curve fitting.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Stress–Strain Curves", "🧮 Curve Fitting", "📐 Hysteresis & Energy"])

    with tab1:
        col_opt1, col_opt2 = st.columns([3, 1])
        with col_opt2:
            show_fill  = st.checkbox("Fill under curves", value=False)
            show_marks = st.checkbox("Show UTS markers", value=True)

        fig = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t).sort_values("strain")
            xs = sub["strain"].values * 100
            ys = sub["stress_abs"].values
            if smooth_data:
                ys = smooth(ys)
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines", name=f"{t} °C",
                line=dict(color=TEMP_COLORS[t], width=2),
                fill="tozeroy" if show_fill else "none",
                fillcolor=TEMP_COLORS[t].replace("#", "rgba(").replace("ff", "ff,0.08)") if show_fill else None,
                hovertemplate=f"<b>{t} °C</b><br>Strain: %{{x:.4f}} %<br>|Stress|: %{{y:.2f}} MPa<extra></extra>",
            ))
            if show_marks:
                idx_uts = sub["stress_abs"].idxmax()
                fig.add_trace(go.Scatter(
                    x=[sub.loc[idx_uts, "strain"] * 100],
                    y=[sub.loc[idx_uts, "stress_abs"]],
                    mode="markers",
                    marker=dict(symbol="star", size=12, color=TEMP_COLORS[t], line=dict(color="white", width=1)),
                    name=f"UTS {t}°C",
                    showlegend=False,
                    hovertemplate=f"<b>UTS @ {t} °C</b><br>Strain: %{{x:.4f}} %<br>Stress: %{{y:.2f}} MPa<extra></extra>",
                ))

        fig.update_layout(
            title="Stress–Strain Curves (All Temperatures)",
            xaxis_title="Strain (%)",
            yaxis_title="|Stress| (MPa)",
            height=520, showlegend=True,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">🧮 Polynomial Curve Fitting</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_b:
            fit_temp   = st.selectbox("Temperature (°C)", selected_temps, key="fit_temp")
            poly_deg   = st.slider("Polynomial degree", 2, 8, 3)
            show_resid = st.checkbox("Show residuals", True)

        sub = get_temp_df(fit_temp).sort_values("strain")
        xs  = sub["strain"].values
        ys  = sub["stress_abs"].values

        pf  = PolynomialFeatures(degree=poly_deg, include_bias=False)
        X_p = pf.fit_transform(xs.reshape(-1, 1))
        reg = LinearRegression().fit(X_p, ys)
        y_pred = reg.predict(X_p)
        r2 = r2_score(ys, y_pred)
        residuals = ys - y_pred

        with col_a:
            fig2 = make_subplots(
                rows=2 if show_resid else 1, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3] if show_resid else [1],
                vertical_spacing=0.05,
            )
            fig2.add_trace(go.Scatter(
                x=xs * 100, y=ys, mode="markers",
                name="Data",
                marker=dict(color=TEMP_COLORS[fit_temp], size=3, opacity=0.5),
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=xs * 100, y=y_pred, mode="lines",
                name=f"Poly fit (deg {poly_deg})",
                line=dict(color="white", width=2.5),
            ), row=1, col=1)
            if show_resid:
                fig2.add_trace(go.Scatter(
                    x=xs * 100, y=residuals, mode="markers",
                    name="Residuals",
                    marker=dict(color="#f0883e", size=2, opacity=0.4),
                ), row=2, col=1)
                fig2.add_hline(y=0, line_dash="dash", line_color="#8b949e", row=2, col=1)
            fig2.update_layout(title=f"Curve Fit @ {fit_temp} °C  |  R² = {r2:.5f}", height=450)
            apply_template(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        coef_df = pd.DataFrame({
            "Degree": range(1, poly_deg + 1),
            "Coefficient": reg.coef_,
        })
        st.markdown(f"**R² = {r2:.6f}** &nbsp;|&nbsp; **Intercept = {reg.intercept_:.4f}**")
        st.dataframe(coef_df.style.format({"Coefficient": "{:.6e}"}), use_container_width=True, height=220)

    with tab3:
        st.markdown('<div class="section-header">📐 Toughness (Area under Curve)</div>', unsafe_allow_html=True)
        toughness_data = []
        for t in selected_temps:
            sub = get_temp_df(t).sort_values("strain")
            area = np.trapz(sub["stress_abs"].values, sub["strain"].values)
            toughness_data.append({"Temperature (°C)": t, "Toughness (MPa·%)": round(area, 4)})

        tough_df = pd.DataFrame(toughness_data)
        fig3 = go.Figure([go.Bar(
            x=tough_df["Temperature (°C)"].astype(str),
            y=tough_df["Toughness (MPa·%)"],
            marker=dict(
                color=[TEMP_COLORS[t] for t in tough_df["Temperature (°C)"]],
                line=dict(color="rgba(255,255,255,0.1)", width=1),
            ),
            text=tough_df["Toughness (MPa·%)"].round(2),
            textposition="outside", textfont=dict(color="#c9d1d9"),
            hovertemplate="<b>%{x}</b><br>Toughness: %{y:.2f} MPa·%<extra></extra>",
        )])
        fig3.update_layout(
            title="Material Toughness by Temperature",
            xaxis_title="Temperature (°C)", yaxis_title="Toughness (MPa·%)",
            height=380,
        )
        apply_template(fig3)
        c_t1, c_t2 = st.columns([2, 1])
        with c_t1:
            st.plotly_chart(fig3, use_container_width=True)
        with c_t2:
            st.dataframe(
                tough_df.set_index("Temperature (°C)").style.background_gradient(cmap="Blues"),
                use_container_width=True, height=280,
            )


# ══════════════════════════════════════════════════════════════════
# MODULE 3 — TIME SERIES EXPLORER
# ══════════════════════════════════════════════════════════════════
elif module == "⏱️  Time Series Explorer":
    st.markdown('<div class="module-title">⏱️ Time Series Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">Explore how stress, strain, and resistance evolve over time for each temperature condition.</div>', unsafe_allow_html=True)

    tab_ts1, tab_ts2, tab_ts3 = st.tabs(["🔀 Multi-Parameter", "🔎 Single Temperature Deep Dive", "📊 Phase Plots"])

    with tab_ts1:
        params = st.multiselect(
            "Parameters to display",
            ["Stress (MPa)", "Strain (%)", "Resistance (Ω)"],
            default=["Stress (MPa)", "Strain (%)"],
        )
        n_rows = len(params)
        if n_rows == 0:
            st.info("Select at least one parameter.")
        else:
            fig = make_subplots(
                rows=n_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=params,
            )
            param_map = {
                "Stress (MPa)":  ("stress_abs", "MPa"),
                "Strain (%)":    ("strain",     "%"),
                "Resistance (Ω)":("resistance", "Ω"),
            }
            for row_i, param in enumerate(params, start=1):
                col_key, unit = param_map[param]
                for t in selected_temps:
                    sub = get_temp_df(t)
                    y = sub[col_key].values
                    if smooth_data:
                        y = smooth(y)
                    fig.add_trace(go.Scatter(
                        x=sub["time"], y=y,
                        mode="lines", name=f"{t} °C",
                        line=dict(color=TEMP_COLORS[t], width=1.5),
                        showlegend=(row_i == 1),
                    ), row=row_i, col=1)

            fig.update_layout(height=200 + n_rows * 200, title="Multi-Parameter Time Series")
            apply_template(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab_ts2:
        deep_temp = st.selectbox("Select Temperature", selected_temps, key="deep_ts")
        sub = get_temp_df(deep_temp)

        time_range = st.slider(
            "Time Window (s)",
            float(sub["time"].min()), float(sub["time"].max()),
            (float(sub["time"].min()), float(sub["time"].max())),
            step=0.5,
        )
        sub_w = sub[(sub["time"] >= time_range[0]) & (sub["time"] <= time_range[1])]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=["Stress (MPa)", "Strain (%)", "Resistance (Ω)"],
        )
        for row_i, (col_key, unit) in enumerate([
            ("stress_abs", "MPa"), ("strain", "%"), ("resistance", "Ω")
        ], start=1):
            y = sub_w[col_key].values
            y_s = smooth(y) if smooth_data else y
            fig.add_trace(go.Scatter(
                x=sub_w["time"], y=y_s, mode="lines",
                line=dict(color=TEMP_COLORS[deep_temp], width=2),
                name=col_key, showlegend=False,
            ), row=row_i, col=1)
            if smooth_data and len(y) > 11:
                fig.add_trace(go.Scatter(
                    x=sub_w["time"], y=y, mode="lines",
                    line=dict(color="#30363d", width=1, dash="dot"),
                    showlegend=False, opacity=0.5,
                ), row=row_i, col=1)

        fig.update_layout(height=620, title=f"Deep Dive @ {deep_temp} °C")
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Rolling stats
        st.markdown('<div class="section-header">📊 Rolling Statistics</div>', unsafe_allow_html=True)
        win = st.slider("Rolling window (points)", 5, 100, 20)
        roll = sub_w[["stress_abs", "strain", "resistance"]].rolling(win, center=True)
        roll_df = pd.DataFrame({
            "time":     sub_w["time"].values,
            "stress_mean": roll["stress_abs"].mean().values,
            "stress_std":  roll["stress_abs"].std().values,
            "strain_mean": roll["strain"].mean().values,
        })
        fig_r = make_subplots(rows=1, cols=2)
        fig_r.add_trace(go.Scatter(x=roll_df["time"], y=roll_df["stress_mean"],
            mode="lines", name="Rolling Mean Stress",
            line=dict(color=TEMP_COLORS[deep_temp], width=2)), row=1, col=1)
        fig_r.add_trace(go.Scatter(x=roll_df["time"], y=roll_df["stress_std"],
            mode="lines", name="Rolling Std Stress",
            line=dict(color="#f0883e", width=2)), row=1, col=2)
        fig_r.update_layout(height=260, title=f"Rolling Statistics (window={win})")
        apply_template(fig_r)
        st.plotly_chart(fig_r, use_container_width=True)

    with tab_ts3:
        st.markdown('<div class="section-header">🌀 Phase Plots (Stress vs Strain over Time)</div>', unsafe_allow_html=True)
        fig_ph = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            fig_ph.add_trace(go.Scatter(
                x=sub["strain"] * 100,
                y=sub["stress_abs"],
                mode="lines+markers",
                marker=dict(size=2, color=sub["time"], colorscale="Viridis", showscale=False),
                line=dict(color=TEMP_COLORS[t], width=1),
                name=f"{t} °C",
                hovertemplate=f"<b>{t} °C</b><br>Strain: %{{x:.4f}} %<br>Stress: %{{y:.2f}} MPa<extra></extra>",
            ))
        fig_ph.update_layout(
            title="Stress–Strain Phase Trajectories (colored by time)",
            xaxis_title="Strain (%)", yaxis_title="|Stress| (MPa)",
            height=480,
        )
        apply_template(fig_ph)
        st.plotly_chart(fig_ph, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# MODULE 4 — TEMPERATURE EFFECTS
# ══════════════════════════════════════════════════════════════════
elif module == "🌡️  Temperature Effects":
    st.markdown('<div class="module-title">🌡️ Temperature Effects</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">How temperature influences peak stress, stiffness, resistance, and energy absorption.</div>', unsafe_allow_html=True)

    # Aggregate per temperature
    agg_data = []
    for t in selected_temps:
        sub = get_temp_df(t)
        agg_data.append({
            "Temperature":     t,
            "Peak Stress":     sub["stress_abs"].max(),
            "Mean Stress":     sub["stress_abs"].mean(),
            "Max Strain":      sub["strain"].max() * 100,
            "Mean Strain":     sub["strain"].mean() * 100,
            "Peak Resistance": sub["resistance"].max(),
            "Min Resistance":  sub["resistance"].min(),
            "Δ Resistance":    sub["resistance"].max() - sub["resistance"].min(),
            "Toughness":       np.trapz(sub["stress_abs"].sort_values().values,
                                        sub.sort_values("strain")["strain"].values),
        })
    agg = pd.DataFrame(agg_data)

    tab_t1, tab_t2, tab_t3 = st.tabs(["📈 Trend Analysis", "🗺️ Heatmap", "🔄 3D Surface"])

    with tab_t1:
        c1, c2 = st.columns(2)
        for col_chart, y_col, title, clr in [
            (c1, "Peak Stress",   "Peak Stress vs Temperature",   "#58a6ff"),
            (c2, "Δ Resistance",  "ΔResistance vs Temperature",   "#f0883e"),
        ]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=agg["Temperature"], y=agg[y_col],
                mode="lines+markers",
                marker=dict(size=8, color=clr, line=dict(color="white", width=1.5)),
                line=dict(color=clr, width=2.5),
                hovertemplate="<b>%{x} °C</b><br>" + y_col + ": %{y:.3f}<extra></extra>",
            ))
            # Trend line
            slope, intercept, r, _, _ = stats.linregress(agg["Temperature"], agg[y_col])
            x_trend = np.linspace(agg["Temperature"].min(), agg["Temperature"].max(), 100)
            y_trend = slope * x_trend + intercept
            fig.add_trace(go.Scatter(
                x=x_trend, y=y_trend, mode="lines",
                line=dict(color="#8b949e", dash="dash", width=1.5),
                name=f"Linear fit (r={r:.3f})",
            ))
            fig.update_layout(title=title, xaxis_title="Temperature (°C)", height=320)
            apply_template(fig)
            col_chart.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        for col_chart, y_col, title, clr in [
            (c3, "Max Strain",    "Max Strain vs Temperature",    "#3fb950"),
            (c4, "Toughness",     "Toughness vs Temperature",     "#bc8cff"),
        ]:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=agg["Temperature"].astype(str),
                y=agg[y_col],
                marker=dict(
                    color=agg[y_col],
                    colorscale="Blues",
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                text=agg[y_col].round(3), textposition="outside",
                textfont=dict(color="#c9d1d9"),
            ))
            fig.update_layout(title=title, xaxis_title="Temperature (°C)", height=320, showlegend=False)
            apply_template(fig)
            col_chart.plotly_chart(fig, use_container_width=True)

    with tab_t2:
        st.markdown('<div class="section-header">🗺️ Parameter Correlation Heatmap across Temperatures</div>', unsafe_allow_html=True)
        pivot_cols = ["Peak Stress", "Mean Stress", "Max Strain", "Mean Strain",
                      "Peak Resistance", "Δ Resistance", "Toughness"]
        pivot_df = agg[["Temperature"] + pivot_cols].set_index("Temperature")
        # Normalize 0-1
        norm_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min())

        fig = px.imshow(
            norm_df.T,
            labels=dict(x="Temperature (°C)", y="Parameter", color="Normalized Value"),
            color_continuous_scale="Blues",
            aspect="auto",
            text_auto=".2f",
        )
        fig.update_layout(height=400, title="Normalized Parameter Heatmap")
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab_t3:
        st.markdown('<div class="section-header">🔄 3D Stress–Strain–Temperature Surface</div>', unsafe_allow_html=True)
        # Downsample for performance
        sub_3d = dff[::5].copy()
        fig3d = go.Figure(data=[go.Scatter3d(
            x=sub_3d["temperature"],
            y=sub_3d["strain"] * 100,
            z=sub_3d["stress_abs"],
            mode="markers",
            marker=dict(
                size=2,
                color=sub_3d["stress_abs"],
                colorscale="Viridis",
                colorbar=dict(title="|Stress| (MPa)", tickfont=dict(color="#c9d1d9")),
                opacity=0.7,
            ),
            hovertemplate="Temp: %{x} °C<br>Strain: %{y:.4f} %<br>Stress: %{z:.2f} MPa<extra></extra>",
        )])
        fig3d.update_layout(
            scene=dict(
                xaxis=dict(title="Temperature (°C)", backgroundcolor="#161b22", gridcolor="#30363d", color="#c9d1d9"),
                yaxis=dict(title="Strain (%)",        backgroundcolor="#161b22", gridcolor="#30363d", color="#c9d1d9"),
                zaxis=dict(title="|Stress| (MPa)",    backgroundcolor="#161b22", gridcolor="#30363d", color="#c9d1d9"),
                bgcolor="#161b22",
            ),
            height=560,
            title="3D Stress–Strain–Temperature Cloud",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
        )
        st.plotly_chart(fig3d, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# MODULE 5 — RESISTANCE ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif module == "⚡  Resistance Analysis":
    st.markdown('<div class="module-title">⚡ Resistance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">Detailed analysis of electrical resistance behavior under mechanical loading and temperature variation.</div>', unsafe_allow_html=True)

    tab_r1, tab_r2, tab_r3 = st.tabs(["📉 Resistance Profiles", "🔗 Resistance vs Stress/Strain", "🌡️ Thermal Sensitivity"])

    with tab_r1:
        fig = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            y = smooth(sub["resistance"].values) if smooth_data else sub["resistance"].values
            fig.add_trace(go.Scatter(
                x=sub["time"], y=y,
                mode="lines", name=f"{t} °C",
                line=dict(color=TEMP_COLORS[t], width=2),
                hovertemplate=f"<b>{t} °C</b><br>Time: %{{x:.1f}} s<br>R: %{{y:.4f}} Ω<extra></extra>",
            ))
        fig.update_layout(
            title="Resistance vs Time by Temperature",
            xaxis_title="Time (s)", yaxis_title="Resistance (Ω)",
            height=400,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Resistance stats
        r_stats = []
        for t in selected_temps:
            sub = get_temp_df(t)
            r_stats.append({
                "Temp (°C)": t,
                "Min (Ω)":   round(sub["resistance"].min(), 4),
                "Max (Ω)":   round(sub["resistance"].max(), 4),
                "Mean (Ω)":  round(sub["resistance"].mean(), 4),
                "Std (Ω)":   round(sub["resistance"].std(), 4),
                "Δ (Ω)":     round(sub["resistance"].max() - sub["resistance"].min(), 4),
                "CV (%)":    round(sub["resistance"].std() / sub["resistance"].mean() * 100, 2),
            })
        r_df = pd.DataFrame(r_stats).set_index("Temp (°C)")
        st.dataframe(r_df.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

    with tab_r2:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for t in selected_temps:
                sub = get_temp_df(t)
                fig.add_trace(go.Scatter(
                    x=sub["stress_abs"], y=sub["resistance"],
                    mode="markers", name=f"{t} °C",
                    marker=dict(size=3, color=TEMP_COLORS[t], opacity=0.4),
                    hovertemplate=f"<b>{t} °C</b><br>Stress: %{{x:.2f}} MPa<br>R: %{{y:.4f}} Ω<extra></extra>",
                ))
            fig.update_layout(
                title="Resistance vs Stress",
                xaxis_title="|Stress| (MPa)", yaxis_title="Resistance (Ω)",
                height=380,
            )
            apply_template(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            for t in selected_temps:
                sub = get_temp_df(t)
                fig.add_trace(go.Scatter(
                    x=sub["strain"] * 100, y=sub["resistance"],
                    mode="markers", name=f"{t} °C",
                    marker=dict(size=3, color=TEMP_COLORS[t], opacity=0.4),
                    hovertemplate=f"<b>{t} °C</b><br>Strain: %{{x:.4f}} %<br>R: %{{y:.4f}} Ω<extra></extra>",
                ))
            fig.update_layout(
                title="Resistance vs Strain",
                xaxis_title="Strain (%)", yaxis_title="Resistance (Ω)",
                height=380,
            )
            apply_template(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab_r3:
        temps_list = selected_temps
        mean_r = [get_temp_df(t)["resistance"].mean() for t in temps_list]
        max_r  = [get_temp_df(t)["resistance"].max()  for t in temps_list]
        min_r  = [get_temp_df(t)["resistance"].min()  for t in temps_list]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=temps_list, y=max_r, mode="lines+markers",
            name="Max R", line=dict(color="#f85149", width=2),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=temps_list, y=mean_r, mode="lines+markers",
            name="Mean R", line=dict(color="#58a6ff", width=2),
            marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=temps_list, y=min_r, mode="lines+markers",
            name="Min R", line=dict(color="#3fb950", width=2),
            marker=dict(size=7),
        ))
        fig.add_traces([
            go.Scatter(x=temps_list + temps_list[::-1],
                       y=max_r + min_r[::-1],
                       fill="toself", fillcolor="rgba(88,166,255,0.07)",
                       line=dict(color="rgba(0,0,0,0)"),
                       showlegend=False, hoverinfo="skip"),
        ])
        fig.update_layout(
            title="Resistance Thermal Sensitivity",
            xaxis_title="Temperature (°C)", yaxis_title="Resistance (Ω)",
            height=420,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

        # TCR estimate
        if len(temps_list) >= 2:
            slope_r, inter_r, r_r, _, _ = stats.linregress(temps_list, mean_r)
            ref_r = inter_r + slope_r * temps_list[0]
            tcr = slope_r / ref_r * 1000 if ref_r != 0 else np.nan
            st.markdown(f"""
<div class="info-box">
<b>Estimated Temperature Coefficient of Resistance (TCR):</b><br>
Slope: {slope_r:.5f} Ω/°C &nbsp;|&nbsp;
TCR ≈ <span style="color:#58a6ff;font-size:1.1rem;font-weight:700;">{tcr:.4f} mΩ/(Ω·°C)</span><br>
Pearson r = {r_r:.4f}
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MODULE 6 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif module == "📊  Statistical Analysis":
    st.markdown('<div class="module-title">📊 Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">Distributions, correlations, normality tests, and pairwise relationships across all variables.</div>', unsafe_allow_html=True)

    tab_s1, tab_s2, tab_s3, tab_s4 = st.tabs(["📦 Distributions", "🔗 Correlations", "🧪 Normality Tests", "🔀 Pair Plots"])

    with tab_s1:
        param_choice = st.selectbox(
            "Parameter",
            ["stress_abs", "strain", "resistance"],
            format_func=lambda x: {"stress_abs": "|Stress| (MPa)", "strain": "Strain (%)", "resistance": "Resistance (Ω)"}[x],
        )
        fig = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            vals = sub[param_choice].values
            fig.add_trace(go.Violin(
                y=vals, name=f"{t} °C",
                box_visible=True, meanline_visible=True,
                fillcolor=TEMP_COLORS[t].replace("#", "rgba(")[:-2] + "44)",
                line_color=TEMP_COLORS[t],
                opacity=0.8,
            ))
        fig.update_layout(
            title=f"Distribution of {param_choice} by Temperature",
            yaxis_title=param_choice, height=430,
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

        # KDE overlay
        fig2 = go.Figure()
        for t in selected_temps:
            sub = get_temp_df(t)
            vals = sub[param_choice].values
            kde_x = np.linspace(vals.min(), vals.max(), 300)
            kde = stats.gaussian_kde(vals)
            fig2.add_trace(go.Scatter(
                x=kde_x, y=kde(kde_x),
                mode="lines", name=f"{t} °C",
                line=dict(color=TEMP_COLORS[t], width=2),
                fill="tozeroy",
                fillcolor=TEMP_COLORS[t].replace("#", "rgba(")[:-2] + "10)",
            ))
        fig2.update_layout(
            title=f"KDE — {param_choice}", xaxis_title=param_choice,
            yaxis_title="Density", height=300,
        )
        apply_template(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_s2:
        corr_df = dff[["stress_abs", "strain", "resistance", "temperature", "time"]].rename(columns={
            "stress_abs": "|Stress|", "strain": "Strain",
            "resistance": "Resistance", "temperature": "Temperature", "time": "Time"
        }).corr()

        fig = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".3f",
            aspect="auto",
        )
        fig.update_layout(
            title="Pearson Correlation Matrix",
            height=450,
            coloraxis_colorbar=dict(tickfont=dict(color="#c9d1d9")),
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab_s3:
        st.markdown('<div class="section-header">🧪 Shapiro-Wilk Normality Test (sample per temperature)</div>', unsafe_allow_html=True)
        norm_results = []
        for t in selected_temps:
            sub = get_temp_df(t)
            for col, label in [("stress_abs", "|Stress|"), ("strain", "Strain"), ("resistance", "Resistance")]:
                sample = sub[col].sample(min(100, len(sub)), random_state=42).values
                stat, pval = stats.shapiro(sample)
                norm_results.append({
                    "Temp (°C)": t,
                    "Variable": label,
                    "W statistic": round(stat, 5),
                    "p-value": f"{pval:.4e}",
                    "Normal (p>0.05)": "✅ Yes" if pval > 0.05 else "❌ No",
                })
        norm_df = pd.DataFrame(norm_results)
        st.dataframe(norm_df, use_container_width=True, height=400)

        # Q-Q plots
        st.markdown('<div class="section-header">📐 Q-Q Plots</div>', unsafe_allow_html=True)
        qq_temp = st.selectbox("Temperature for Q-Q", selected_temps, key="qq_temp")
        qq_col  = st.selectbox("Variable", ["stress_abs", "strain", "resistance"], key="qq_col")
        sub_qq  = get_temp_df(qq_temp)
        vals_qq = sub_qq[qq_col].values
        (osm, osr), (slope, intercept, r) = stats.probplot(vals_qq)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(size=3, color=TEMP_COLORS[qq_temp], opacity=0.5),
            name="Data",
        ))
        fig_qq.add_trace(go.Scatter(
            x=[min(osm), max(osm)],
            y=[slope * min(osm) + intercept, slope * max(osm) + intercept],
            mode="lines", line=dict(color="white", dash="dash", width=2),
            name=f"Normal line (r={r:.4f})",
        ))
        fig_qq.update_layout(
            title=f"Q-Q Plot — {qq_col} @ {qq_temp} °C",
            xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
            height=380,
        )
        apply_template(fig_qq)
        st.plotly_chart(fig_qq, use_container_width=True)

    with tab_s4:
        st.markdown('<div class="section-header">🔀 Pairwise Scatter Matrix</div>', unsafe_allow_html=True)
        sample_df = dff[["stress_abs", "strain", "resistance", "temperature"]].sample(
            min(2000, len(dff)), random_state=42
        ).rename(columns={"stress_abs": "|Stress|", "strain": "Strain",
                           "resistance": "Resistance", "temperature": "Temp"})
        fig_pair = px.scatter_matrix(
            sample_df,
            dimensions=["|Stress|", "Strain", "Resistance"],
            color="Temp",
            color_continuous_scale="Viridis",
            opacity=0.4,
        )
        fig_pair.update_traces(marker=dict(size=2))
        fig_pair.update_layout(
            height=550, title="Scatter Matrix (sampled)",
            font=dict(color="#c9d1d9"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pair, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# MODULE 7 — MATERIAL PROPERTIES
# ══════════════════════════════════════════════════════════════════
elif module == "🔧  Material Properties":
    st.markdown('<div class="module-title">🔧 Material Properties Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">Engineering properties derived from the test data: stiffness, UTS, toughness, and more.</div>', unsafe_allow_html=True)

    # Full table
    st.markdown('<div class="section-header">📋 Computed Properties per Temperature</div>', unsafe_allow_html=True)
    display_mat = mat_table.loc[mat_table.index.isin(selected_temps)]
    st.dataframe(
        display_mat.style.background_gradient(cmap="Blues", axis=0).format(
            {c: "{:.3f}" for c in display_mat.select_dtypes(include=float).columns}
        ),
        use_container_width=True, height=300,
    )

    st.markdown('<div class="section-header">📈 Property Trends</div>', unsafe_allow_html=True)

    props_to_plot = [
        ("UTS (MPa)", "#58a6ff", "Ultimate Tensile Strength"),
        ("ΔResistance (Ω)", "#f0883e", "Resistance Change"),
        ("Max Strain (%)", "#3fb950", "Maximum Strain"),
    ]

    cols = st.columns(len(props_to_plot))
    for col_chart, (prop, clr, title) in zip(cols, props_to_plot):
        t_vals   = [t for t in selected_temps if t in display_mat.index]
        y_vals   = [float(display_mat.loc[t, prop]) if display_mat.loc[t, prop] != "—" else np.nan for t in t_vals]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_vals, y=y_vals,
            mode="lines+markers",
            marker=dict(size=9, color=clr, line=dict(color="white", width=1.5)),
            line=dict(color=clr, width=2.5),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Temp (°C)", yaxis_title=prop,
            height=280, margin=dict(l=50, r=20, t=45, b=50),
        )
        apply_template(fig)
        col_chart.plotly_chart(fig, use_container_width=True)

    # Gauge cards for selected temperature
    st.markdown('<div class="section-header">🎯 Property Inspector</div>', unsafe_allow_html=True)
    insp_temp = st.select_slider("Select Temperature", options=[t for t in selected_temps], value=selected_temps[0])
    sub_insp  = get_temp_df(insp_temp)
    props_i   = compute_material_props(sub_insp)

    cols_g = st.columns(4)
    gauge_items = [
        ("UTS (MPa)",           props_i["UTS (MPa)"],            "#58a6ff", 0, df["stress_abs"].max()),
        ("Max Strain (%)",      props_i["Max Strain (%)"],        "#3fb950", 0, df["strain"].max() * 100),
        ("ΔResistance (Ω)",     props_i["ΔResistance (Ω)"],      "#f0883e", 0, 5),
        ("Mean Resistance (Ω)", props_i["Mean Resistance (Ω)"],  "#bc8cff", 10, 18),
    ]
    for col_g, (label, val, clr, mn, mx) in zip(cols_g, gauge_items):
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(val) if val != "—" else 0,
            title={"text": label, "font": {"color": "#c9d1d9", "size": 12}},
            gauge=dict(
                axis=dict(range=[mn, mx], tickcolor="#8b949e", tickfont=dict(color="#8b949e")),
                bar=dict(color=clr),
                bgcolor="#21262d",
                bordercolor="#30363d",
                steps=[dict(range=[mn, mx], color="#161b22")],
                threshold=dict(
                    line=dict(color="white", width=2),
                    thickness=0.75,
                    value=float(val) if val != "—" else 0,
                ),
            ),
            number={"font": {"color": clr, "size": 20}},
        ))
        fig_g.update_layout(
            height=220, margin=dict(l=15, r=15, t=45, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
        )
        col_g.plotly_chart(fig_g, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# MODULE 8 — RAW DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
elif module == "🔍  Raw Data Explorer":
    st.markdown('<div class="module-title">🔍 Raw Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="module-subtitle">Browse, filter, and export the underlying dataset.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        time_f = st.slider("Time range (s)", 0.0, 69.9, (0.0, 69.9), 0.1)
    with c2:
        stress_f = st.slider(
            "|Stress| range (MPa)",
            float(df["stress_abs"].min()), float(df["stress_abs"].max()),
            (float(df["stress_abs"].min()), float(df["stress_abs"].max())),
        )
    with c3:
        sort_col = st.selectbox("Sort by", ["time", "stress_abs", "strain", "resistance", "temperature"])

    filtered = dff[
        (dff["time"] >= time_f[0]) & (dff["time"] <= time_f[1]) &
        (dff["stress_abs"] >= stress_f[0]) & (dff["stress_abs"] <= stress_f[1])
    ].sort_values(sort_col)

    st.markdown(f"**{len(filtered):,}** rows matching filters")

    display_cols = ["time", "stress_abs", "strain", "resistance", "temperature"]
    show_df = filtered[display_cols].rename(columns={
        "time": "Time (s)", "stress_abs": "|Stress| (MPa)",
        "strain": "Strain (%)", "resistance": "Resistance (Ω)",
        "temperature": "Temperature (°C)",
    })
    st.dataframe(
        show_df.style.background_gradient(cmap="Blues", subset=["|Stress| (MPa)"]),
        use_container_width=True, height=450,
    )

    # Download
    csv_bytes = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_material_data.csv",
        mime="text/csv",
    )

    # Quick stats
    st.markdown('<div class="section-header">📊 Quick Statistics</div>', unsafe_allow_html=True)
    st.dataframe(show_df.describe().T.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)
