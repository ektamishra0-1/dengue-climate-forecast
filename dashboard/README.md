# DengueCast: Brazilian Dengue Climate Forecasting & Analytics Platform

An interactive scientific demonstration dashboard for the **Brazilian Dengue Climate Forecasting** research project. **DengueCast** showcases exploratory data analysis across 558 Brazilian microregions (2000–2019), climate-dengue spatiotemporal relationships, rigorous stationarity diagnostics, classical time-series benchmark models, and out-of-sample forecast evaluations.

---

## 🔬 Key Research Capabilities & Sections

1. **Executive Overview & Key Insights**
   - Continental coverage: 558 microregions across 20 years (133,920 records $\times$ 95 features).
   - High-level KPIs, model rankings, and summary of empirical findings.
   - Highlights SARIMA's champion performance ($\text{MAE} = 0.3039$, 72.2% error reduction over Seasonal Naive).

2. **Exploratory Data Analysis (EDA)**
   - Monthly and annual incidence trajectories on national and microregion scales.
   - Seasonal transmission curves peaking in March–April (epidemiological weeks 10–18).
   - Zero-inflation metrics (19.06% zero months nationally) and extreme skewness (kurtosis = 142.1) justifying $\log(1+x)$ transformation.
   - Macro-regional comparisons across North, Northeast, Central-West, Southeast, and South.

3. **Climate & Environmental Drivers**
   - Spatiotemporal heatmaps of monthly temperature ($T_{\max}$, $T_{\min}$), diurnal temperature range (DTR), and Palmer Drought Severity Index (PDSI).
   - Lagged cross-correlation curves (lags 0 to 6 months) demonstrating optimal climate associations at 1–3 months prior to peak transmission.
   - **Epidemiological Note:** Prominent distinction between statistical correlation and mechanistic causality.

4. **Stationarity & Time-Series Diagnostics**
   - Interactive dual hypothesis testing: Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS).
   - Rolling mean and rolling standard deviation charts showing variance non-stationarity in raw incidence vs. stabilized log1p series.
   - Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.
   - Seasonal and first-order differencing analysis justifying $(p, d, q) \times (P, D, Q)_{12}$ specifications.

5. **Classical Model Comparison & Benchmarks**
   - Performance evaluation across 5 models:
     1. **Seasonal Naive** (Baseline benchmark)
     2. **ARIMA** (Non-seasonal autoregressive integrated moving average)
     3. **SARIMA** (Seasonal autoregressive integrated moving average — Champion)
     4. **Holt-Winters** (Triple exponential smoothing with additive seasonality)
     5. **SARIMAX** (Investigative extension incorporating climate exogenous variables)
   - Comprehensive error metrics: MAE, RMSE, sMAPE, MASE on both $\log(1+x)$ transformed scale and back-transformed incidence per 100k.
   - Validation (2016–2017) vs. Out-of-Sample Test (2018–2019) comparisons.

6. **Interactive Forecast Explorer**
   - Location selector: 5 representative biomes (Belo Horizonte, Goiânia, Fortaleza, Manaus, Maringá) plus National Aggregate.
   - Multi-model toggle: overlay single or multiple model predictions simultaneously against actual observations.
   - Scale switcher: toggle between $\log(1+x)$ scale and original incidence per 100k.
   - Period switcher: Validation (2016–2017) or Test (2018–2019).
   - Dynamic point-by-point forecast error tracking.

---

## 🛠️ Tech Stack

- **Streamlit** (v1.35+) — Interactive scientific dashboard framework
- **Plotly** (v5.20+) — Interactive vector and time-series charting
- **Pandas** & **NumPy** — Data manipulation and vectorized transformation
- **Statsmodels** & **SciPy** — Classical time-series modeling and diagnostics

---

## 🚀 Installation & Launch

### 1. Environment Setup

Ensure the project virtual environment is active:

```bash
# From the repository root:
source .venv/bin/activate
pip install -r dashboard/requirements.txt
```

### 2. Launching DengueCast

Run the Streamlit application from the repository root:

```bash
streamlit run dashboard/app.py
```

Or using the virtual environment interpreter directly:

```bash
.venv/bin/streamlit run dashboard/app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🧪 Testing & Verification

Comprehensive unit tests verify data loader integrity, schema validity, stationarity tables, and prediction consistency:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_dashboard.py
```

To run the entire test suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

---

## 🛡️ Data Integrity & Reproducibility Guarantees

- **No Synthetic Data:** Every chart, metric card, and prediction line is loaded from actual precomputed pipeline outputs (`data/dengue_features.csv`, `results/metrics/`, `results/predictions/`).
- **No Data Leakage:** Preprocessing transformations and stationarity decisions were determined strictly on training partitions (2002–2015).
- **Strict Chronological Order:** Validation (2016–2017) and Test (2018–2019) partitions were evaluated sequentially without lookahead bias.
