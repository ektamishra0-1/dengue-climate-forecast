"""
DengueCast: Brazilian Dengue Climate Forecasting & Analytics Platform
Interactive research demonstration dashboard for Phase 1 & Phase 2 project outputs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ensure dashboard and root packages are resolvable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_loader import (
    load_eda_dataset,
    load_national_timeseries,
    load_all_metrics,
    load_prediction_series,
    LOCATION_MAP
)

# -----------------------------------------------------------------------------
# Page Configuration & Dark Navy / Teal Visual Identity
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DengueCast | Brazilian Dengue Climate Forecasting",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Navy & Teal CSS Theme
CUSTOM_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #0A192F;
        color: #E6F1FF;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #020C1B;
        border-right: 1px solid #1E2D4A;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #112240;
        border: 1px solid #233554;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8892B0;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #64FFDA;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-caption {
        color: #A8B2D1;
        font-size: 0.80rem;
    }
    
    /* Scientific Disclaimer Box */
    .disclaimer-box {
        background-color: #0D2847;
        border-left: 4px solid #00B4D8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
        font-size: 0.90rem;
        color: #CCD6F6;
    }
    
    /* Takeaway Box */
    .takeaway-box {
        background-color: #172A45;
        border-left: 4px solid #64FFDA;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 12px;
        font-size: 0.88rem;
        color: #E6F1FF;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #E6F1FF !important;
        font-weight: 700;
    }
    
    /* Tables styling */
    .dataframe {
        background-color: #112240 !important;
        color: #E6F1FF !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Plotly Dark Theme Factory
# -----------------------------------------------------------------------------
def get_plotly_layout(**kwargs) -> go.Layout:
    """Standardizes Plotly figures with dark navy / teal styling."""
    base = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(10, 25, 47, 0)",
        "plot_bgcolor": "rgba(17, 34, 64, 0.4)",
        "font": dict(color="#CCD6F6", family="sans-serif", size=12),
        "margin": dict(l=40, r=40, t=50, b=40),
        "xaxis": dict(
            gridcolor="#1F2D4A",
            zerolinecolor="#233554",
            showline=True,
            linecolor="#233554"
        ),
        "yaxis": dict(
            gridcolor="#1F2D4A",
            zerolinecolor="#233554",
            showline=True,
            linecolor="#233554"
        ),
        "legend": dict(
            bgcolor="rgba(17, 34, 64, 0.7)",
            bordercolor="#233554",
            borderwidth=1,
            font=dict(size=11)
        ),
    }
    for k, v in kwargs.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return go.Layout(**base)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style='text-align: center; padding: 10px 0;'>
    <h2 style='color: #64FFDA; margin: 0; font-size: 1.6rem;'>🦟 DengueCast</h2>
    <p style='color: #8892B0; font-size: 0.80rem; margin: 0;'>Brazilian Climate Forecasting Platform</p>
</div>
<hr style='border: 0.5px solid #1E2D4A; margin: 10px 0 20px 0;'>
""", unsafe_allow_html=True)

nav_section = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview",
        "2. Exploratory Data Analysis",
        "3. Climate Analysis",
        "4. Stationarity & Diagnostics",
        "5. Model Comparison",
        "6. Forecast Explorer"
    ],
    index=0
)

st.sidebar.markdown("""
<hr style='border: 0.5px solid #1E2D4A; margin: 25px 0 15px 0;'>
<div style='font-size: 0.75rem; color: #8892B0;'>
    <b style='color: #64FFDA;'>Research Scope:</b><br>
    • 558 Brazilian Microregions (IBGE)<br>
    • 2000–2019 Monthly Panel (240 Mo)<br>
    • Target: log1p(Incidence per 100k)<br>
    • Partitions: Train (02–15), Val (16–17), Test (18–19)<br>
    • Status: Phase 2 Evaluation Complete
</div>
""", unsafe_allow_html=True)

# Pre-load shared datasets
eda_df = load_eda_dataset()
nat_df = load_national_timeseries()
stat_df, val_metrics_df, test_metrics_df = load_all_metrics()

# =============================================================================
# SECTION 1: OVERVIEW
# =============================================================================
if nav_section == "1. Overview":
    st.title("DengueCast: Platform Overview")
    st.markdown("**Brazilian Dengue Climate Forecasting & Analytics Platform**")
    
    st.markdown("""
    Welcome to **DengueCast**, an interactive research analytics platform investigating the complex relationship
    between meteorological extremes (temperatures, Palmer Drought Severity Index) and dengue fever incidence across Brazil.
    This platform consolidates empirical findings and benchmarking outputs from Phase 1 (EDA & Feature Engineering) and
    Phase 2 (Classical Time-Series Forecasting).
    """)
    
    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Monitored Microregions</div>
            <div class="metric-value">558</div>
            <div class="metric-caption">All 27 Brazilian States & 5 Regions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Space-Time Panel</div>
            <div class="metric-value">133,920</div>
            <div class="metric-caption">2000–2019 (240 Balanced Months)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Total Dengue Cases</div>
            <div class="metric-value">12.9M</div>
            <div class="metric-caption">Mean: 101.4 | Max: 88,821 / mo</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Champion Model (Test)</div>
            <div class="metric-value">SARIMA</div>
            <div class="metric-caption">72% Error Reduction vs Baseline</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### National Longitudinal Trajectory (2001–2019)")
    
    # Interactive National Case Series with Outbreak Highlights
    fig_nat = go.Figure()
    fig_nat.add_trace(go.Scatter(
        x=nat_df["date"],
        y=nat_df["total_cases"] / 1000.0,
        mode="lines",
        name="Total Monthly Cases (Thousands)",
        line=dict(color="#FF4B4B", width=2.2),
        hovertemplate="%{x|%b %Y}: <b>%{y:.1f}k cases</b><extra></extra>"
    ))
    fig_nat.add_trace(go.Scatter(
        x=nat_df["date"],
        y=nat_df["cases_per_100k"],
        mode="lines",
        name="Mean Incidence (per 100k pop)",
        yaxis="y2",
        line=dict(color="#64FFDA", width=1.8, dash="dot"),
        hovertemplate="%{x|%b %Y}: <b>%{y:.1f} per 100k</b><extra></extra>"
    ))
    
    # Add epidemic wave shaded areas
    epidemic_years = [2002, 2008, 2010, 2013, 2015, 2016, 2019]
    for yr in epidemic_years:
        fig_nat.add_vrect(
            x0=f"{yr}-01-01", x1=f"{yr}-12-31",
            fillcolor="#FFB703", opacity=0.12, line_width=0,
            annotation_text=f"{yr}", annotation_position="top left",
            annotation_font=dict(color="#FFB703", size=10)
        )
        
    layout = get_plotly_layout(
        title="Continental Brazil Monthly Dengue Volume & Incidence Rate (Major Outbreak Waves Shaded)",
        yaxis=dict(title="Dengue Cases (Thousands)", color="#FF4B4B"),
        yaxis2=dict(
            title="Incidence per 100k Population",
            color="#64FFDA",
            overlaying="y",
            side="right"
        )
    )
    fig_nat.update_layout(layout)
    st.plotly_chart(fig_nat, width="stretch")
    
    st.markdown("""
    <div class="takeaway-box">
        <b>Epidemiological Observation:</b> Dengue in Brazil exhibits a dual temporal rhythm:
        a regular 12-month solar transmission cycle peaking in Southern Hemisphere autumn (April–May),
        superimposed upon multi-annual epidemic surges (2002, 2008, 2010, 2013, 2015–2016, 2019) driven
        by periodic viral serotype replacement (DENV-1 through DENV-4) and post-epidemic herd immunity depletion.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Core Research Takeaways")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        * **Population Normalization is Critical**: Evaluating raw counts introduces extreme metropolitan bias. Normalizing into `cases_per_100k` and variance-stabilizing via $\\ln(1 + \\text{incidence})$ uncovers massive per-capita attack rates in small and mid-sized microregions.
        * **Thermal Phase Lag**: Summer temperatures (`tmax`, `tmin`) peak during December–February, whereas dengue incidence peaks 2 to 3 months later (April–May), matching the biological extrinsic incubation period inside the mosquito.
        """)
    with colB:
        st.markdown("""
        * **The PDSI Drought Paradox**: Both extreme wetness and severe drought trigger transmission spikes. Water rationing during severe droughts forces domestic open-container water storage, creating abundant artificial breeding sites for *Aedes aegypti*.
        * **The Aggregation Smoothing Fallacy**: National aggregate models achieve artificially low errors (sMAPE ~4.6%) by averaging out spatial variance. Microregion panel forecasting is indispensable for municipal public health deployment.
        """)

# =============================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
elif nav_section == "2. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("Detailed investigation of incidence distributions, seasonal profiles, zero-inflation, and spatial disparities.")
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Seasonal Rhythms", "Distribution & Zero-Inflation", "Regional & Biome Heterogeneity"])
    
    with eda_tab1:
        st.subheader("Intra-Annual Seasonality: Monthly Transmission Curves")
        
        # Monthly aggregation across all years
        obs_df = eda_df.dropna(subset=["dengue_cases", "population"])
        month_agg = obs_df.groupby("month").agg(
            mean_inc=("cases_per_100k", "mean"),
            median_inc=("cases_per_100k", "median"),
            p75_inc=("cases_per_100k", lambda x: np.percentile(x, 75)),
            total_k_cases=("dengue_cases", lambda x: x.sum() / 1000.0)
        ).reset_index()
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_agg["month_name"] = month_names
        
        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(
            x=month_agg["month_name"],
            y=month_agg["total_k_cases"],
            name="Cumulative Cases (Thousands)",
            marker_color="#00B4D8",
            opacity=0.75
        ))
        fig_season.add_trace(go.Scatter(
            x=month_agg["month_name"],
            y=month_agg["mean_inc"],
            name="Mean Incidence per 100k",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#FFD166", width=2.5),
            marker=dict(size=8, color="#FFD166")
        ))
        
        fig_season.update_layout(get_plotly_layout(
            title="Monthly Dengue Transmission Profile in Brazil (Aggregated 2001–2019)",
            xaxis=dict(title="Calendar Month"),
            yaxis=dict(title="Cumulative Cases (Thousands)", color="#00B4D8"),
            yaxis2=dict(title="Mean Incidence (per 100k)", color="#FFD166", overlaying="y", side="right")
        ))
        st.plotly_chart(fig_season, width="stretch")
        
        st.markdown("""
        <div class="takeaway-box">
            <b>Seasonal Dynamics:</b> Transmission accelerates during January–February, reaches an acute peak in
            <b>April and May</b>, and collapses to an annual minimum between August and October during the southern dry winter.
        </div>
        """, unsafe_allow_html=True)
        
    with eda_tab2:
        st.subheader("Target Distribution, Skewness & Zero-Inflation")
        
        zc1, zc2, zc3, zc4 = st.columns(4)
        with zc1:
            st.metric("Zero-Case Observations", "34,031", "26.75% of observed panel")
        with zc2:
            st.metric("Distribution Skewness", "49.51", "Severe positive skew")
        with zc3:
            st.metric("Median Incidence", "4.0 / 100k", "Mean is 101.4 / 100k")
        with zc4:
            st.metric("Single-Month Max", "8,340 / 100k", "Localized epidemic spike")
            
        st.markdown("#### Log1p Transformation Effect on Target Distribution")
        
        fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["Raw Incidence per 100k (Positive Only, Log X)", "log1p(Incidence) Full Cohort"])
        
        pos_inc = obs_df.loc[obs_df["cases_per_100k"] > 0, "cases_per_100k"]
        fig_dist.add_trace(
            go.Histogram(x=pos_inc, nbinsx=60, marker_color="#00B4D8", name="Raw Incidence"),
            row=1, col=1
        )
        fig_dist.update_xaxes(type="log", row=1, col=1, title="cases_per_100k (log scale)")
        
        fig_dist.add_trace(
            go.Histogram(x=obs_df["log1p_incidence"], nbinsx=60, marker_color="#64FFDA", name="log1p(Incidence)"),
            row=1, col=2
        )
        fig_dist.update_xaxes(title="log1p(cases_per_100k)", row=1, col=2)
        
        fig_dist.update_layout(get_plotly_layout(
            title="Comparison of Raw Positive Incidence vs. log1p Transformed Distribution",
            showlegend=False
        ))
        st.plotly_chart(fig_dist, width="stretch")
        
        st.markdown("""
        <div class="takeaway-box">
            <b>Statistical Rationale:</b> Modeling raw dengue case counts directly leads to catastrophic variance instability.
            Applying the logarithmic transformation $\\ln(1 + y)$ successfully compresses positive skewness,
            allowing classical linear and autoregressive state-space models to converge reliably.
        </div>
        """, unsafe_allow_html=True)
        
    with eda_tab3:
        st.subheader("Geographical & Biome Heterogeneity")
        
        reg_df = obs_df.groupby("region_name")["cases_per_100k"].agg(["mean", "median", "max"]).reset_index()
        biome_df = obs_df.groupby("biome_name")["cases_per_100k"].agg(["mean", "median", "max"]).reset_index()
        
        colR1, colR2 = st.columns(2)
        with colR1:
            fig_reg = px.bar(
                reg_df.sort_values("mean", ascending=False),
                x="region_name", y="mean",
                title="Mean Incidence per 100k by Macro-Region",
                labels={"region_name": "Macro-Region", "mean": "Mean Incidence (per 100k)"},
                color="mean",
                color_continuous_scale="Teal"
            )
            fig_reg.update_layout(get_plotly_layout())
            st.plotly_chart(fig_reg, width="stretch")
            
        with colR2:
            fig_bio = px.bar(
                biome_df.sort_values("mean", ascending=False),
                x="biome_name", y="mean",
                title="Mean Incidence per 100k by Ecological Biome",
                labels={"biome_name": "Biome", "mean": "Mean Incidence (per 100k)"},
                color="mean",
                color_continuous_scale="Viridis"
            )
            fig_bio.update_layout(get_plotly_layout())
            st.plotly_chart(fig_bio, width="stretch")
            
        st.markdown("""
        <div class="takeaway-box">
            <b>Spatial Regimes:</b> The <b>Centre-West (Cerrado)</b> and <b>Southeast (Atlantic Rainforest)</b>
            sustain the highest incidence in Brazil, whereas the temperate <b>South (Pampa)</b> exhibits near-zero
            transmission due to cold winter frosts that suppress vector survival.
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# SECTION 3: CLIMATE ANALYSIS
# =============================================================================
elif nav_section == "3. Climate Analysis":
    st.title("Climate Drivers & Environmental Analysis")
    st.markdown("Empirical investigation of temperatures, Palmer Drought Severity Index (PDSI), and distributed lag relationships.")
    
    st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ Scientific Disclaimer on Causality:</b><br>
        All relationships shown in this section represent <i>empirical statistical correlations</i> rather than proof of direct causality.
        Meteorological variables define the <b>permissive physiological boundary</b> for <i>Aedes aegypti</i> survival and viral replication,
        but whether transmission manifests as an outbreak depends on host herd immunity, serotype replacements, and public health vector-control interventions.
    </div>
    """, unsafe_allow_html=True)
    
    c_tab1, c_tab2, c_tab3 = st.tabs(["Thermal Suitability", "Drought Patterns (PDSI)", "Lagged Cross-Correlations"])
    
    with c_tab1:
        st.subheader("Thermal Limits of Dengue Transmission")
        
        # Temperature binning analysis
        obs_df = eda_df.dropna(subset=["cases_per_100k", "tmin", "tmax"]).copy()
        obs_df["tmin_bin"] = pd.cut(obs_df["tmin"], bins=np.arange(5, 27, 2))
        obs_df["tmax_bin"] = pd.cut(obs_df["tmax"], bins=np.arange(15, 39, 2))
        
        tmin_mean = obs_df.groupby("tmin_bin", observed=False)["cases_per_100k"].mean().reset_index()
        tmin_mean["mid"] = [b.mid for b in tmin_mean["tmin_bin"]]
        
        tmax_mean = obs_df.groupby("tmax_bin", observed=False)["cases_per_100k"].mean().reset_index()
        tmax_mean["mid"] = [b.mid for b in tmax_mean["tmax_bin"]]
        
        fig_temp = make_subplots(rows=1, cols=2, subplot_titles=["Incidence vs Minimum Temperature (tmin)", "Incidence vs Maximum Temperature (tmax)"])
        
        fig_temp.add_trace(
            go.Scatter(x=tmin_mean["mid"], y=tmin_mean["cases_per_100k"], mode="lines+markers", line=dict(color="#00B4D8", width=2.5), name="tmin"),
            row=1, col=1
        )
        fig_temp.add_vline(x=15, line_dash="dash", line_color="#FFB703", annotation_text="15°C Feeding Cessation", row=1, col=1)
        fig_temp.update_xaxes(title="Minimum Temperature (°C)", row=1, col=1)
        fig_temp.update_yaxes(title="Mean Incidence per 100k", row=1, col=1)
        
        fig_temp.add_trace(
            go.Scatter(x=tmax_mean["mid"], y=tmax_mean["cases_per_100k"], mode="lines+markers", line=dict(color="#FF4B4B", width=2.5), name="tmax"),
            row=1, col=2
        )
        fig_temp.add_vline(x=32, line_dash="dash", line_color="#FFB703", annotation_text="32°C Thermal Stress", row=1, col=2)
        fig_temp.update_xaxes(title="Maximum Temperature (°C)", row=1, col=2)
        fig_temp.update_yaxes(title="Mean Incidence per 100k", row=1, col=2)
        
        fig_temp.update_layout(get_plotly_layout(title="Non-Linear Thermal Constraints on Dengue Incidence in Brazil", showlegend=False))
        st.plotly_chart(fig_temp, width="stretch")
        
        st.markdown("""
        <div class="takeaway-box">
            <b>Biological Interpretation:</b> Incidence collapses below 15°C because <i>Aedes aegypti</i> feeding stops and the extrinsic incubation period of the virus ceases.
            Between 24°C and 30°C, viral replication peaks. Above 32°C–34°C, adult mosquito mortality spikes, attenuating transmission.
        </div>
        """, unsafe_allow_html=True)
        
    with c_tab2:
        st.subheader("The PDSI Drought Paradox")
        
        def cat_pdsi(v):
            if pd.isna(v): return np.nan
            if v <= -4.0: return "1. Extreme Drought (≤-4)"
            elif v <= -3.0: return "2. Severe Drought (-4 to -3)"
            elif v <= -2.0: return "3. Moderate Drought (-3 to -2)"
            elif v < 2.0: return "4. Near Normal (-2 to +2)"
            elif v < 4.0: return "5. Moderately Wet (+2 to +4)"
            else: return "6. Extremely Wet (≥+4)"
            
        obs_df["pdsi_tier"] = obs_df["pdsi"].apply(cat_pdsi)
        pdsi_agg = obs_df.dropna(subset=["pdsi_tier"]).groupby("pdsi_tier")["cases_per_100k"].agg(["mean", "count"]).reset_index()
        
        fig_pdsi = px.bar(
            pdsi_agg,
            x="pdsi_tier", y="mean",
            title="Mean Dengue Incidence Across Palmer Drought Severity Index (PDSI) Categories",
            labels={"pdsi_tier": "PDSI Moisture / Drought Category", "mean": "Mean Incidence (per 100k)"},
            color="mean",
            color_continuous_scale="Teal"
        )
        fig_pdsi.update_layout(get_plotly_layout())
        st.plotly_chart(fig_pdsi, width="stretch")
        
        st.markdown("""
        <div class="takeaway-box">
            <b>The Drought Paradox Explained:</b> Counterintuitively, severe and extreme droughts exhibit elevated dengue incidence.
            During severe droughts in Brazil, municipal water rationing forces households to store water in open buckets, cisterns, and barrels,
            creating abundant artificial breeding sites for domestic <i>Aedes aegypti</i> mosquitoes.
        </div>
        """, unsafe_allow_html=True)
        
    with c_tab3:
        st.subheader("Distributed Lag Cross-Correlations (0 to 6 Months)")
        
        lags = list(range(0, 7))
        tmax_corr, tmin_corr, pdsi_corr = [], [], []
        
        for l in lags:
            if l == 0:
                tmax_corr.append(nat_df["mean_tmax"].corr(nat_df["mean_inc"] if "mean_inc" in nat_df else nat_df["cases_per_100k"]))
                tmin_corr.append(nat_df["mean_tmin"].corr(nat_df["mean_inc"] if "mean_inc" in nat_df else nat_df["cases_per_100k"]))
                pdsi_corr.append(nat_df["mean_pdsi"].corr(nat_df["mean_inc"] if "mean_inc" in nat_df else nat_df["cases_per_100k"]))
            else:
                tmax_corr.append(nat_df["mean_tmax"].iloc[:-l].corr(nat_df["cases_per_100k"].iloc[l:]))
                tmin_corr.append(nat_df["mean_tmin"].iloc[:-l].corr(nat_df["cases_per_100k"].iloc[l:]))
                pdsi_corr.append(nat_df["mean_pdsi"].iloc[:-l].corr(nat_df["cases_per_100k"].iloc[l:]))
                
        fig_ccf = go.Figure()
        fig_ccf.add_trace(go.Scatter(x=lags, y=tmax_corr, mode="lines+markers", name="Tmax (Lead Horizon)", line=dict(color="#FF4B4B", width=2.5)))
        fig_ccf.add_trace(go.Scatter(x=lags, y=tmin_corr, mode="lines+markers", name="Tmin (Lead Horizon)", line=dict(color="#00B4D8", width=2.5)))
        fig_ccf.add_trace(go.Scatter(x=lags, y=pdsi_corr, mode="lines+markers", name="PDSI Drought (Lead Horizon)", line=dict(color="#64FFDA", width=2.0, dash="dash")))
        fig_ccf.add_hline(y=0, line_color="#8892B0", line_width=1)
        
        fig_ccf.update_layout(get_plotly_layout(
            title="Cross-Correlation Function: Climate Variables Leading Dengue Incidence by k Months",
            xaxis=dict(title="Climate Lag (Months Lead)", dtick=1),
            yaxis=dict(title="Pearson Correlation with Incidence")
        ))
        st.plotly_chart(fig_ccf, width="stretch")
        
        st.markdown("""
        <div class="takeaway-box">
            <b>Empirical Lead:</b> Maximum and minimum temperatures reach their peak correlation with dengue incidence at <b>Lags of 1 to 2 months</b> ($r \\approx 0.38 - 0.48$).
            By lag 4 to 5, the correlation decays, directly informing the $X_{t-2}$ regressors used in the SARIMAX model.
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# SECTION 4: STATIONARITY & TIME-SERIES DIAGNOSTICS
# =============================================================================
elif nav_section == "4. Stationarity & Diagnostics":
    st.title("Stationarity & Time-Series Diagnostics")
    st.markdown("Formal statistical stationarity testing, differencing evaluation, and autocorrelation analysis.")
    
    st.markdown("""
    #### Plain-Language Statistical Guide
    * **Augmented Dickey-Fuller (ADF) Test**:
      * *Null Hypothesis ($H_0$)*: The series has a **unit root** (it is non-stationary and drifts).
      * *Decision Rule*: If $p < 0.05$, reject $H_0 \\implies$ the series is **stationary**.
    * **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test**:
      * *Null Hypothesis ($H_0$)*: The series is **stationary** around a level.
      * *Decision Rule*: If $p < 0.05$, reject $H_0 \\implies$ the series is **non-stationary**.
    * **Why Tests Often Disagree on Dengue Series**: Raw dengue incidence oscillates in persistent 12-month waves. ADF detects mean-reverting properties over 20 years ($p < 0.05$), but KPSS flags the massive seasonal swings as non-stationarity ($p < 0.05$).
    """)
    
    # Filter stationarity audit table by location
    loc_names = stat_df["Location"].unique().tolist()
    sel_loc = st.selectbox("Select Location to Inspect Differencing Diagnostics:", loc_names, index=0)
    
    sub_stat = stat_df[stat_df["Location"] == sel_loc][
        ["Transformation", "ADF Stat", "ADF p-val", "ADF Decision", "KPSS Stat", "KPSS p-val", "KPSS Decision", "Joint Status", "Series Variance"]
    ]
    st.dataframe(sub_stat.style.format({
        "ADF Stat": "{:.4f}", "ADF p-val": "{:.4f}",
        "KPSS Stat": "{:.4f}", "KPSS p-val": "{:.4f}",
        "Series Variance": "{:.4f}"
    }), width="stretch")
    
    st.markdown("### Rolling Statistics (12-Month Window)")
    
    # Compute rolling stats for the selected location
    # Map selection back to code
    loc_to_key = {
        "Belo Horizonte": "Belo Horizonte",
        "Goiânia": "Goiania",
        "Goiania": "Goiania",
        "Fortaleza": "Fortaleza",
        "Manaus": "Manaus",
        "Maringá": "Maringa",
        "Maringa": "Maringa",
        "National Aggregate (Brazil)": "National_Aggregate",
        "National_Aggregate": "National_Aggregate"
    }
    sel_code = loc_to_key.get(sel_loc, "Belo Horizonte")
    
    # Extract training series (2002-2015)
    if sel_code == "National_Aggregate":
        s_target = nat_df.set_index("date")["log1p_incidence"].loc["2002-01-01":"2015-12-01"]
    else:
        # Load microregion series
        c_int = LOCATION_MAP[sel_code]["code"]
        sub_s = eda_df[eda_df["micro_code"] == c_int].sort_values("date").set_index("date")["log1p_incidence"].loc["2002-01-01":"2015-12-01"]
        s_target = sub_s
        
    roll_m = s_target.rolling(12).mean()
    roll_s = s_target.rolling(12).std()
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=s_target.index, y=s_target.values, mode="lines", name="Observed log1p", line=dict(color="#8892B0", width=1.2)))
    fig_roll.add_trace(go.Scatter(x=roll_m.index, y=roll_m.values, mode="lines", name="12-Mo Rolling Mean", line=dict(color="#FF4B4B", width=2.5)))
    fig_roll.add_trace(go.Scatter(x=roll_s.index, y=roll_s.values, mode="lines", name="12-Mo Rolling Std", line=dict(color="#64FFDA", width=2.0, dash="dot")))
    
    fig_roll.update_layout(get_plotly_layout(
        title=f"12-Month Rolling Mean & Standard Deviation on Training Set ({sel_loc})",
        xaxis=dict(title="Date"),
        yaxis=dict(title="log1p(Incidence)")
    ))
    st.plotly_chart(fig_roll, width="stretch")
    
    st.markdown("""
    <div class="takeaway-box">
        <b>Differencing Justification:</b> Applying seasonal differencing $\\Delta_{12} y_t = y_t - y_{t-12}$
        successfully eliminates the 12-month cyclical wave, bringing the KPSS test statistic into the stationary range ($p > 0.10$)
        across all locations. This justifies the inclusion of a seasonal integration order $D = 1$ in SARIMA.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SECTION 5: MODEL COMPARISON
# =============================================================================
elif nav_section == "5. Model Comparison":
    st.title("Model Comparison & Benchmark Rankings")
    st.markdown("Rigorous benchmark evaluation of Seasonal Naive, ARIMA, SARIMA, Holt-Winters, and SARIMAX Climate Extension.")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        eval_split = st.selectbox("Select Evaluation Partition:", ["Test Set (2018–2019)", "Validation Set (2016–2017)"])
    with m_col2:
        loc_filter = st.selectbox("Filter Location:", ["All Locations (Mean)", "Belo Horizonte", "Goiânia", "Fortaleza", "Manaus", "Maringá", "National Aggregate (Brazil)"])
    with m_col3:
        target_scale_choice = st.selectbox("Display Metrics Scale:", ["Primary Scale (log1p_incidence)", "Original Scale (cases per 100k)"])
        
    curr_metrics_df = test_metrics_df if "Test" in eval_split else val_metrics_df
    
    if loc_filter == "All Locations (Mean)":
        display_df = curr_metrics_df.groupby("Model")[
            ["MAE (log1p)", "RMSE (log1p)", "sMAPE% (log1p)", "MASE (log1p)", "MAE (orig)", "RMSE (orig)", "sMAPE% (orig)"]
        ].mean().reset_index().sort_values("MAE (log1p)")
    else:
        display_df = curr_metrics_df[curr_metrics_df["Location"] == loc_filter][
            ["Model", "MAE (log1p)", "RMSE (log1p)", "sMAPE% (log1p)", "MASE (log1p)", "MAE (orig)", "RMSE (orig)", "sMAPE% (orig)", "MAPE% (orig, >=1)"]
        ].sort_values("MAE (log1p)")
        
    # Metric KPI Highlights for Champion
    champ_row = display_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Top Ranked Model", champ_row["Model"], "Lowest MAE")
    with c2:
        val_mae = champ_row["MAE (log1p)"] if "Primary" in target_scale_choice else champ_row["MAE (orig)"]
        st.metric("MAE", f"{val_mae:.4f}" if "Primary" in target_scale_choice else f"{val_mae:.2f}", "Champion error")
    with c3:
        val_rmse = champ_row["RMSE (log1p)"] if "Primary" in target_scale_choice else champ_row["RMSE (orig)"]
        st.metric("RMSE", f"{val_rmse:.4f}" if "Primary" in target_scale_choice else f"{val_rmse:.2f}", "Root Mean Sq Error")
    with c4:
        val_smape = champ_row["sMAPE% (log1p)"] if "Primary" in target_scale_choice else champ_row["sMAPE% (orig)"]
        st.metric("sMAPE", f"{val_smape:.2f}%", "Bounded [0-200%]")
        
    st.markdown("### Comparative Performance Bar Chart")
    
    metric_plot_col = "MAE (log1p)" if "Primary" in target_scale_choice else "MAE (orig)"
    fig_comp = px.bar(
        display_df.sort_values(metric_plot_col, ascending=True),
        x="Model", y=metric_plot_col,
        color="Model",
        title=f"Model Comparison: {metric_plot_col} on {eval_split} ({loc_filter})",
        labels={metric_plot_col: metric_plot_col},
        color_discrete_sequence=["#64FFDA", "#00B4D8", "#48CAE4", "#90E0EF", "#FF4B4B"]
    )
    fig_comp.update_layout(get_plotly_layout(showlegend=False))
    st.plotly_chart(fig_comp, width="stretch")
    
    st.markdown("### Comprehensive Evaluation Metrics Table")
    st.dataframe(display_df.style.format({
        "MAE (log1p)": "{:.4f}", "RMSE (log1p)": "{:.4f}", "sMAPE% (log1p)": "{:.2f}%",
        "MASE (log1p)": "{:.4f}", "MAE (orig)": "{:.2f}", "RMSE (orig)": "{:.2f}",
        "sMAPE% (orig)": "{:.2f}%", "MAPE% (orig, >=1)": "{:.2f}%"
    }), width="stretch")
    
    st.markdown("""
    <div class="takeaway-box">
        <b>Key Evaluation Findings:</b><br>
        1. <b>SARIMA</b> is the definitive champion on the final test set, achieving a <b>72% error reduction</b> over the Seasonal Naive benchmark.<br>
        2. <b>Non-Seasonal ARIMA</b> achieves reasonable log-scale errors but severely underestimates explosive peak months on the original scale (RMSE 62.71 vs 34.56 for SARIMA).<br>
        3. <b>SARIMAX Climate</b> matches or slightly edges SARIMA for the National Aggregate (MAE 0.1427 vs 0.1435), but univariate seasonal persistence dominates microregion 1-step forecasts.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# SECTION 6: FORECAST EXPLORER
# =============================================================================
elif nav_section == "6. Forecast Explorer":
    st.title("Interactive Forecast Explorer")
    st.markdown("Explore rolling one-step-ahead out-of-sample forecasts against actual observations across locations and models.")
    
    fe_col1, fe_col2, fe_col3, fe_col4 = st.columns(4)
    with fe_col1:
        loc_key = st.selectbox(
            "Select Microregion / Benchmark:",
            list(LOCATION_MAP.keys()),
            format_func=lambda k: LOCATION_MAP[k]["display_name"],
            index=0
        )
    with fe_col2:
        period_choice = st.selectbox("Select Evaluation Horizon:", ["Test Period (2018–2019)", "Validation Period (2016–2017)"])
    with fe_col3:
        scale_choice = st.selectbox("Select Prediction Scale:", ["Primary: log1p(Incidence)", "Original: cases per 100k"])
    with fe_col4:
        period_code = "test" if "Test" in period_choice else "val"
        clean_loc = LOCATION_MAP[loc_key]["clean_name"]
        pred_df = load_prediction_series(period_code, clean_loc)
        
        available_models = ["SARIMA", "ARIMA", "Holt_Winters", "SARIMAX_Climate", "Seasonal_Naive"]
        sel_models = st.multiselect("Models to Display:", available_models, default=["SARIMA", "Seasonal_Naive"])
        
    if pred_df is None or pred_df.empty:
        st.error(f"No prediction file found for {loc_key} in {period_choice}.")
    else:
        # Determine actual series based on scale
        is_log = "Primary" in scale_choice
        actual_col = "actual_log1p" if is_log else "actual_orig"
        y_actual = pred_df[actual_col]
        
        fig_fore = go.Figure()
        # Actual trace
        fig_fore.add_trace(go.Scatter(
            x=pred_df.index, y=y_actual.values,
            mode="lines+markers",
            name="Actual Observed",
            line=dict(color="#FFFFFF", width=3.0),
            marker=dict(size=6, color="#FFFFFF"),
            zorder=5
        ))
        
        palette = {
            "SARIMA": "#64FFDA",
            "ARIMA": "#00B4D8",
            "Holt_Winters": "#FFD166",
            "SARIMAX_Climate": "#FF758F",
            "Seasonal_Naive": "#8892B0"
        }
        
        for m in sel_models:
            if m in pred_df.columns:
                p_series = pred_df[m]
                if not is_log:
                    # Convert log predictions to original scale via max(0, expm1(y))
                    p_series = np.maximum(0.0, np.expm1(p_series))
                    
                fig_fore.add_trace(go.Scatter(
                    x=pred_df.index, y=p_series.values,
                    mode="lines+markers",
                    name=m,
                    line=dict(color=palette.get(m, "#48CAE4"), width=2.0, dash="dash" if m != "SARIMA" else "solid"),
                    marker=dict(size=4)
                ))
                
        fig_fore.update_layout(get_plotly_layout(
            title=f"Actual vs 1-Step Rolling Forecasts: {LOCATION_MAP[loc_key]['display_name']} ({period_choice})",
            xaxis=dict(title="Forecast Month"),
            yaxis=dict(title="log1p(Incidence)" if is_log else "Incidence per 100k Population")
        ))
        st.plotly_chart(fig_fore, width="stretch")
        
        # Residuals Plot
        st.subheader("Forecast Residuals ($e_t = y_t - \\hat{y}_t$)")
        fig_err = go.Figure()
        for m in sel_models:
            if m in pred_df.columns:
                p_series = pred_df[m] if is_log else np.maximum(0.0, np.expm1(pred_df[m]))
                err = y_actual - p_series
                fig_err.add_trace(go.Scatter(
                    x=pred_df.index, y=err.values,
                    mode="lines+markers",
                    name=f"Error ({m})",
                    line=dict(color=palette.get(m, "#48CAE4"), width=1.8)
                ))
        fig_err.add_hline(y=0, line_color="#8892B0", line_width=1)
        fig_err.update_layout(get_plotly_layout(
            title="One-Step Out-of-Sample Forecast Residuals Over Time",
            xaxis=dict(title="Forecast Month"),
            yaxis=dict(title="Residual Error (e_t)")
        ))
        st.plotly_chart(fig_err, width="stretch")
        
        st.markdown(f"""
        <div class="takeaway-box">
            <b>Location Context ({LOCATION_MAP[loc_key]['clean_name']}):</b> {LOCATION_MAP[loc_key]['desc']}
            <br>Notice how SARIMA adapts smoothly to seasonal turns without lagging behind the epidemic wave peak.
        </div>
        """, unsafe_allow_html=True)

