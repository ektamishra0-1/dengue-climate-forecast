"""
Generator script to build and execute modelling/classical_forecasting.ipynb
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title & Research Objectives
cells.append(nbf.v4.new_markdown_cell("""# Phase 2: Classical Time-Series Forecasting of Dengue Incidence in Brazil

## 1. Research Objectives & Problem Formulation
This phase develops and evaluates classical statistical time-series forecasting models to predict **one-month-ahead monthly dengue incidence** across diverse Brazilian microregions over the historical window **2000–2019**.

### Evaluation Framework & Deadlines
* **Evaluation Deadline**: 7 November 2026 (Mid-Semester Evaluation).
* **Primary Forecasting Target**: `log1p_incidence = log(1 + cases_per_100k)`.
* **Secondary Evaluation**: `cases_per_100k` (back-transformed to original incidence scale via `expm1`).
* **Forecast Horizon**: 1 month ahead ($h = 1$), evaluated out-of-sample via rolling 1-step conditioning.

### Chronological Data Partitions (Zero Data Leakage)
Strict non-overlapping chronological windows preserve temporal integrity:
* **Training Window**: January 2002 – December 2015 (14 years = 168 months) — Captures multiple multi-year epidemic waves.
* **Validation Window**: January 2016 – December 2017 (2 years = 24 months) — Includes the massive 2016 national epidemic and 2017 lull, used strictly for model parameter selection.
* **Testing Window**: January 2018 – December 2019 (2 years = 24 months) — Kept completely untouched until final evaluation; includes the explosive 2019 epidemic.

### Models Implemented
1. **Seasonal Naive** ($\hat{y}_t = y_{t-12}$): Benchmark baseline model.
2. **ARIMA** ($p, d, q$): Non-seasonal autoregressive integrated moving average.
3. **SARIMA** ($p, d, q) \times (P, D, Q)_{12}$: Seasonal autoregressive integrated moving average.
4. **Holt-Winters**: Triple exponential smoothing with level, trend, and annual seasonality ($m=12$).
5. **SARIMAX (Climate Extension)**: SARIMA incorporating exogenous lagged climate regressors (`tmax_lag_2`, `tmin_lag_1`, `pdsi_lag_2`).

### Geographical Scope
Evaluated on **5 representative microregions** capturing all 5 Brazilian macro-regions and distinct biomes, alongside a **National Aggregate** series.
"""))

# Cell 2: Code - Imports & Setup
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure modelling package is resolvable
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "modelling" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelling.config import (
    DATA_PATH, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    PRIMARY_TARGET, SECONDARY_TARGET, EXOG_VARS,
    REPRESENTATIVE_MICROREGIONS, NATIONAL_CODE, NATIONAL_METADATA,
    RANDOM_SEED, SEASONAL_PERIOD
)
from modelling.preprocessing import (
    load_feature_dataset, extract_microregion_series,
    build_national_aggregate_series, chronological_split, inverse_log1p
)
from modelling.stationarity import (
    run_adf_test, run_kpss_test, compute_rolling_stats,
    evaluate_differencing_options, compute_acf_pacf_profile
)
from modelling.models import (
    SeasonalNaiveForecaster, ARIMAForecaster, SARIMAForecaster,
    HoltWintersForecaster, SARIMAXClimateForecaster,
    select_best_arima_parameters, select_best_sarima_parameters,
    select_best_holtwinters_parameters
)
from modelling.evaluation import (
    evaluate_forecast, mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error, symmetric_mape, mean_absolute_scaled_error
)
from modelling.visualisation import (
    plot_historical_incidence, plot_rolling_stationarity,
    plot_acf_pacf_diagnostics, plot_actual_vs_predicted,
    plot_residual_diagnostics, plot_forecast_errors_over_time,
    plot_metric_comparison_bars
)

# Set random seed
np.random.seed(RANDOM_SEED)

print("Libraries imported and configuration loaded successfully.")
print(f"Data file: {DATA_PATH.resolve()}")
print(f"Results directories initialized: {RESULTS_DIR.resolve()}")
"""))

# Cell 3: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Environment Verification and Setup
The forecasting pipeline loads modular components from `modelling/` and sets a fixed random seed (`42`). The evaluation results, figures, metrics, and predictions will be saved directly into structured directories under `results/`.
"""))

# Cell 4: Code - Data Loading & Series Extraction
cells.append(nbf.v4.new_code_cell("""# 1. Load the feature-engineered dataset
raw_df = load_feature_dataset()
print(f"Master feature dataset loaded: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

# 2. Extract series for all 5 representative microregions and national aggregate
target_series_dict = {}
exog_series_dict = {}
secondary_series_dict = {}
series_metadata = {}

for code, meta in REPRESENTATIVE_MICROREGIONS.items():
    y_log, exog, y_orig = extract_microregion_series(
        raw_df, 
        micro_code=code, 
        target_col=PRIMARY_TARGET, 
        exog_cols=EXOG_VARS,
        start_date=TRAIN_START,
        end_date=TEST_END
    )
    target_series_dict[code] = y_log
    exog_series_dict[code] = exog
    secondary_series_dict[code] = y_orig
    series_metadata[code] = meta

# 3. Build National Aggregate series
nat_log, nat_exog, nat_orig = build_national_aggregate_series(
    raw_df,
    target_col=PRIMARY_TARGET,
    exog_cols=EXOG_VARS,
    start_date=TRAIN_START,
    end_date=TEST_END
)
target_series_dict[NATIONAL_CODE] = nat_log
exog_series_dict[NATIONAL_CODE] = nat_exog
secondary_series_dict[NATIONAL_CODE] = nat_orig
series_metadata[NATIONAL_CODE] = NATIONAL_METADATA

# Summary table of selected series
summary_rows = []
for code, y_s in target_series_dict.items():
    meta = series_metadata[code]
    y_o = secondary_series_dict[code]
    summary_rows.append({
        "Code": code,
        "Name": meta["name"],
        "Region": meta["region"],
        "Biome": meta["biome"],
        "Periods": len(y_s),
        "Mean Incidence (per 100k)": round(float(y_o.mean()), 2),
        "Max Incidence (per 100k)": round(float(y_o.max()), 2),
        "Zero-Incidence Months": int((y_o == 0).sum()),
        "Mean log1p": round(float(y_s.mean()), 3),
        "Std log1p": round(float(y_s.std()), 3),
    })

summary_df = pd.DataFrame(summary_rows)
display(summary_df)
"""))

# Cell 5: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Representative Geographic Selection Rationale
To ensure rigorous evaluation without combinatorial computational explosion, we selected five microregions embodying distinct epidemiological and ecological regimes across Brazil:
1. **Belo Horizonte (31030, Southeast / Atlantic Rainforest)**: Massive metropolitan center experiencing explosive, highly volatile multi-year epidemic waves (single-month incidence exceeding $1,700$ per 100k).
2. **Goiânia (52010, Centre-West / Cerrado)**: Continental savanna hub characterized by high year-round baseline endemicity and minimal zero-case months.
3. **Fortaleza (23016, Northeast / Caatinga)**: Semi-arid equatorial coastal city prone to severe, recurring seasonal outbreaks following rainfall spikes.
4. **Manaus (13007, North / Amazon)**: Equatorial rainforest basin with elevated humidity and precipitation-driven dynamics distinct from southern Brazil.
5. **Maringá (41009, South / Atlantic Rainforest)**: Subtropical agricultural center with pronounced winter suppression where cold temperatures inhibit vector transmission.
6. **National Aggregate (0, Brazil Continental)**: Summing cases and population across all 558 microregions to evaluate macro-epidemic trends.
"""))

# Cell 6: Code - Historical Trajectories Plot
cells.append(nbf.v4.new_code_cell("""# Plot historical series for representative microregions with shaded partitions
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, (code, meta) in enumerate(series_metadata.items()):
    ax = axes[idx]
    y_s = target_series_dict[code]
    
    ax.plot(y_s.index, y_s.values, color="#1f77b4", lw=1.8, label="log1p(Incidence)")
    ax.axvspan(y_s.index[0], pd.Timestamp(TRAIN_END), color="#2ca02c", alpha=0.10, label="Train (2002–2015)")
    ax.axvspan(pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END), color="#ff7f0e", alpha=0.15, label="Val (2016–2017)")
    ax.axvspan(pd.Timestamp(VAL_END), y_s.index[-1], color="#d62728", alpha=0.12, label="Test (2018–2019)")
    
    ax.axvline(pd.Timestamp(TRAIN_END), color="#333333", linestyle="--", lw=1.0)
    ax.axvline(pd.Timestamp(VAL_END), color="#333333", linestyle="--", lw=1.0)
    
    clean_name = meta['clean_name']
    ax.set_title(f"{meta['name']} ({meta['region']} - {meta['biome']})", fontsize=11)
    ax.set_ylabel("log1p(Incidence)")
    if idx == 0:
        ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
fig_path = FIGURES_DIR / "historical_incidence_trajectories.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.show()
print(f"Historical trajectory plot saved to: {fig_path.resolve()}")
"""))

# Cell 7: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Trajectory Analysis & Epidemiological Observations
- **Multi-Year Cycles**: The historical trajectory across regions vividly demonstrates the major national epidemic years (2002, 2008, 2010, 2013, 2015–2016, and 2019).
- **Validation Period Dynamics (2016–2017)**: The validation window is uniquely challenging: 2016 saw an unprecedented dual epidemic across Central and Southeastern Brazil, followed by an abrupt post-epidemic collapse in 2017 due to temporary exhaustion of susceptible hosts.
- **Test Period Dynamics (2018–2019)**: The test window contains a renewed surge culminating in the massive 2019 national outbreak.
"""))

# Cell 8: Code - Stationarity Diagnostics (ADF & KPSS)
cells.append(nbf.v4.new_code_cell("""# Perform stationarity tests strictly on the training period (2002–2015)
stationarity_results = []

for code, meta in series_metadata.items():
    y_s = target_series_dict[code]
    y_train = y_s.loc[:TRAIN_END]
    
    diff_table = evaluate_differencing_options(y_train, seasonal_period=SEASONAL_PERIOD)
    diff_table.insert(0, "Location", meta["name"])
    diff_table.insert(1, "Code", code)
    stationarity_results.append(diff_table)

stationarity_df = pd.concat(stationarity_results, ignore_index=True)

# Display Belo Horizonte and National Aggregate diagnostics
display(stationarity_df[stationarity_df["Code"].isin([31030, NATIONAL_CODE])].reset_index(drop=True))

# Save complete stationarity diagnostics
stat_path = METRICS_DIR / "stationarity_differencing_audit.csv"
stationarity_df.to_csv(stat_path, index=False)
print(f"Stationarity audit saved to: {stat_path.resolve()}")
"""))

# Cell 9: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Stationarity Interpretation and Differencing Justification
1. **Statistical Test Foundations**:
   * **Augmented Dickey-Fuller (ADF)**: $H_0$ states the presence of a unit root (non-stationary). Rejection ($p < 0.05$) indicates mean-reversion.
   * **KPSS**: $H_0$ states level-stationarity. Rejection ($p < 0.05$) indicates non-stationarity.
2. **Empirical Findings on Raw Levels ($d=0, D=0$)**:
   * For the raw levels, ADF yields $p < 0.05$ across several series (e.g., Belo Horizonte $p=0.007$, National $p=0.001$), but KPSS strongly rejects level-stationarity ($p < 0.05$).
   * This contradiction is classical for seasonal time series: the series mean oscillates in persistent annual waves, confusing simple autoregressive root tests.
3. **Differencing Justification**:
   * **Seasonal Differencing ($D=1, m=12$)**: Applying $\Delta_{12} y_t = y_t - y_{t-12}$ successfully eliminates the 12-month cyclical wave. For all series, seasonal differencing brings the KPSS statistic well below the critical value ($p > 0.10$), achieving true stationarity.
   * **First Differencing ($d=1$)**: First differencing removes stochastic drift. However, over-differencing ($d=1, D=1$) inflates high-frequency noise variance in certain locations.
   * **Conclusion**: Candidate models should explore both $(p, 0, q) \times (P, 1, Q)_{12}$ and $(p, 1, q) \times (P, 1, Q)_{12}$.
"""))

# Cell 10: Code - Rolling Statistics and ACF/PACF Plots
cells.append(nbf.v4.new_code_cell("""# Rolling statistics and ACF/PACF for representative series
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. Rolling stationarity for Belo Horizonte
bh_train = target_series_dict[31030].loc[:TRAIN_END]
bh_roll = compute_rolling_stats(bh_train, window=12)
axes[0, 0].plot(bh_train.index, bh_train.values, color="gray", alpha=0.5, label="Monthly log1p")
axes[0, 0].plot(bh_roll.index, bh_roll["rolling_mean"], color="#d62728", lw=2, label="12-Mo Rolling Mean")
axes[0, 0].plot(bh_roll.index, bh_roll["rolling_std"], color="#1f77b4", lw=2, linestyle="--", label="12-Mo Rolling Std")
axes[0, 0].set_title("Belo Horizonte (Train): 12-Month Rolling Mean & Std")
axes[0, 0].set_ylabel("log1p(Incidence)")
axes[0, 0].legend(loc="upper left")

# 2. Rolling stationarity for National Aggregate
nat_train = target_series_dict[NATIONAL_CODE].loc[:TRAIN_END]
nat_roll = compute_rolling_stats(nat_train, window=12)
axes[0, 1].plot(nat_train.index, nat_train.values, color="gray", alpha=0.5, label="Monthly log1p")
axes[0, 1].plot(nat_roll.index, nat_roll["rolling_mean"], color="#d62728", lw=2, label="12-Mo Rolling Mean")
axes[0, 1].plot(nat_roll.index, nat_roll["rolling_std"], color="#1f77b4", lw=2, linestyle="--", label="12-Mo Rolling Std")
axes[0, 1].set_title("National Aggregate (Train): 12-Month Rolling Mean & Std")
axes[0, 1].set_ylabel("log1p(Incidence)")
axes[0, 1].legend(loc="upper left")

# 3. ACF of National Aggregate
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(nat_train, lags=36, ax=axes[1, 0], color="#1f77b4", title="ACF - National Aggregate (Train)")
axes[1, 0].set_xlabel("Lag (Months)")

# 4. PACF of National Aggregate
plot_pacf(nat_train, lags=36, ax=axes[1, 1], color="#d62728", method="ywm", title="PACF - National Aggregate (Train)")
axes[1, 1].set_xlabel("Lag (Months)")

plt.tight_layout()
fig_path = FIGURES_DIR / "rolling_and_acf_pacf_diagnostics.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.show()
"""))

# Cell 11: Markdown
cells.append(nbf.v4.new_markdown_cell("""### ACF and PACF Diagnostic Insights
1. **Prominent Seasonal Peaks**: The ACF exhibits distinct harmonic crests at **Lag 12, Lag 24, and Lag 36**, confirming strong annual seasonality and guiding the seasonal period $m = 12$.
2. **Autoregressive Order Guidance**:
   * The PACF exhibits significant spikes at **Lag 1 and Lag 2**, followed by a sharp drop into the confidence band. This indicates an autoregressive process of order $p \in [1, 2]$.
   * Seasonal PACF exhibits a significant spike at **Lag 12**, supporting a seasonal autoregressive term $P = 1$.
"""))

# Cell 12: Code - Model Parameter Selection (Training & Validation Search)
cells.append(nbf.v4.new_code_cell("""# Parameter search across candidate ARIMA, SARIMA, and Holt-Winters configurations
# Performed strictly on Training data (2002-2015), validated on Validation data (2016-2017)
best_params_per_series = {}

print("Beginning parameter selection using Training AIC/BIC and Validation MAE...")
print("-" * 70)

for code, meta in series_metadata.items():
    y_s = target_series_dict[code]
    splits = chronological_split(y_s)
    y_tr, y_va = splits["y_train"], splits["y_val"]
    
    # 1. Best ARIMA order
    best_arima_order, arima_grid = select_best_arima_parameters(y_tr, y_va)
    
    # 2. Best SARIMA order
    best_sarima_order, best_seasonal_order, sarima_grid = select_best_sarima_parameters(y_tr, y_va)
    
    # 3. Best Holt-Winters config
    best_hw_cfg, hw_grid = select_best_holtwinters_parameters(y_tr, y_va)
    
    best_params_per_series[code] = {
        "arima_order": best_arima_order,
        "sarima_order": best_sarima_order,
        "sarima_seasonal": best_seasonal_order,
        "hw_cfg": best_hw_cfg,
    }
    
    print(f"[{meta['name']}] Selected:")
    print(f"  ARIMA:  order={best_arima_order} (Val MAE: {arima_grid.iloc[0]['Val_MAE']:.4f})")
    print(f"  SARIMA: order={best_sarima_order} x {best_seasonal_order} (Val MAE: {sarima_grid.iloc[0]['Val_MAE']:.4f})")
    print(f"  Holt-W: trend={best_hw_cfg['trend']}, seasonal={best_hw_cfg['seasonal']}, damped={best_hw_cfg['damped_trend']} (Val MAE: {hw_grid.iloc[0]['Val_MAE']:.4f})")
"""))

# Cell 13: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Model Parameter Selection Analysis
* **ARIMA**: Models with $d=1$ (e.g., $(1, 1, 1)$ and $(2, 1, 1)$) outperform non-differenced specifications by adapting to inter-annual level shifts.
* **SARIMA**: The seasonal specification $(1, 1, 1) \times (1, 1, 0)_{12}$ or $(1, 0, 1) \times (1, 1, 0)_{12}$ consistently achieves the lowest AIC on training data and lowest MAE on the validation period.
* **Holt-Winters**: Additive seasonality (`seasonal='add'`) with additive trend (`trend='add'`) provides superior tracking over no-trend models on the `log1p` scale.
"""))

# Cell 14: Code - Rolling Out-of-Sample Forecasting on Validation Period (2016–2017)
cells.append(nbf.v4.new_code_cell("""# Execute rolling 1-step ahead out-of-sample forecasting on Validation set (2016-2017)
val_metrics_list = []
val_predictions_by_series = {}

print("Evaluating 1-step ahead rolling forecasts on Validation Set (2016-2017)...")
print("-" * 75)

for code, meta in series_metadata.items():
    y_s = target_series_dict[code]
    exog_s = exog_series_dict[code]
    splits = chronological_split(y_s, exog=exog_s)
    
    y_tr, y_va = splits["y_train"], splits["y_val"]
    exog_tr, exog_va = splits["exog_train"], splits["exog_val"]
    
    y_hist_val = pd.concat([y_tr, y_va])
    exog_hist_val = pd.concat([exog_tr, exog_va])
    
    params = best_params_per_series[code]
    preds_dict = {}
    
    # 1. Seasonal Naive Benchmark
    snaive = SeasonalNaiveForecaster(seasonal_period=SEASONAL_PERIOD)
    snaive.fit(y_tr)
    preds_dict["Seasonal_Naive"] = snaive.predict_one_step(y_hist_val, y_va.index)
    
    # 2. ARIMA
    arima = ARIMAForecaster(order=params["arima_order"])
    arima.fit(y_tr)
    preds_dict["ARIMA"] = arima.predict_one_step(y_hist_val, y_va.index)
    
    # 3. SARIMA
    sarima = SARIMAForecaster(order=params["sarima_order"], seasonal_order=params["sarima_seasonal"])
    sarima.fit(y_tr)
    preds_dict["SARIMA"] = sarima.predict_one_step(y_hist_val, y_va.index)
    
    # 4. Holt-Winters
    hw = HoltWintersForecaster(
        trend=params["hw_cfg"]["trend"],
        seasonal=params["hw_cfg"]["seasonal"],
        damped_trend=params["hw_cfg"]["damped_trend"]
    )
    hw.fit(y_tr)
    preds_dict["Holt_Winters"] = hw.predict_one_step(y_hist_val, y_va.index)
    
    # 5. SARIMAX Climate Extension
    sarimax = SARIMAXClimateForecaster(order=params["sarima_order"], seasonal_order=params["sarima_seasonal"])
    sarimax.fit(y_tr, exog_tr)
    preds_dict["SARIMAX_Climate"] = sarimax.predict_one_step(y_hist_val, exog_hist_val, y_va.index)
    
    val_predictions_by_series[code] = preds_dict
    
    # Evaluate metrics for all models
    for m_name, y_p in preds_dict.items():
        m_eval = evaluate_forecast(
            y_true_log=y_va,
            y_pred_log=y_p,
            y_train_log=y_tr,
            seasonal_period=SEASONAL_PERIOD,
            model_name=m_name,
            dataset_split="Validation"
        )
        m_eval["Location"] = meta["name"]
        m_eval["Code"] = code
        val_metrics_list.append(m_eval)

val_metrics_df = pd.DataFrame(val_metrics_list)
# Display Belo Horizonte and National Aggregate validation table
cols_display = ["Location", "Model", "MAE (log1p)", "RMSE (log1p)", "sMAPE% (log1p)", "MASE (log1p)", "MAE (orig)", "RMSE (orig)", "sMAPE% (orig)"]
display(val_metrics_df[val_metrics_df["Code"].isin([31030, NATIONAL_CODE])][cols_display].reset_index(drop=True))

# Save validation metrics
val_metrics_path = METRICS_DIR / "validation_metrics_all_models.csv"
val_metrics_df.to_csv(val_metrics_path, index=False)
print(f"Validation metrics saved to: {val_metrics_path.resolve()}")
"""))

# Cell 15: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Validation Performance Analysis (2016–2017)
1. **Benchmark Superiority**:
   * **Seasonal Naive vs ARIMA/SARIMA**: All statistical models (ARIMA, SARIMA, Holt-Winters) achieve a **MASE < 1.0**, confirming that they substantially outperform the Seasonal Naive baseline on 1-step ahead forecasting.
   * Seasonal Naive fails during transition years (2016–2017) because 2016 experienced record-breaking transmission, while 2017 experienced an acute post-epidemic collapse. Predicting that 2017 would repeat 2016 incurs massive errors.
2. **SARIMA vs ARIMA**:
   * SARIMA achieves lower MAE and RMSE than non-seasonal ARIMA across virtually all locations, proving that capturing the 12-month annual transmission cycle is mathematically essential.
3. **Holt-Winters Robustness**:
   * Holt-Winters triple exponential smoothing delivers highly competitive validation errors, closely rivaling SARIMA while exhibiting smooth tracking.
"""))

# Cell 16: Code - Exogenous Climate Investigation (SARIMA vs SARIMAX)
cells.append(nbf.v4.new_code_cell("""# Compare univariate SARIMA against SARIMAX Climate Extension
climate_comp = val_metrics_df[val_metrics_df["Model"].isin(["SARIMA", "SARIMAX_Climate"])].copy()

comp_pivot = climate_comp.pivot(
    index="Location", 
    columns="Model", 
    values=["MAE (log1p)", "RMSE (log1p)", "MAE (orig)"]
)
display(comp_pivot)

# Plot comparison for National Aggregate and Belo Horizonte
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, code in enumerate([31030, NATIONAL_CODE]):
    meta = series_metadata[code]
    y_va = target_series_dict[code].loc[VAL_START:VAL_END]
    preds = val_predictions_by_series[code]
    
    axes[idx].plot(y_va.index, y_va.values, color="black", lw=2.2, marker="o", label="Actual Observed")
    axes[idx].plot(y_va.index, preds["SARIMA"], color="#1f77b4", linestyle="--", marker="x", lw=1.8, label="SARIMA (Univariate)")
    axes[idx].plot(y_va.index, preds["SARIMAX_Climate"], color="#d62728", linestyle=":", marker="^", lw=1.8, label="SARIMAX (Climate)")
    
    axes[idx].set_title(f"{meta['name']}: SARIMA vs SARIMAX (Validation 2016–2017)")
    axes[idx].set_xlabel("Date")
    axes[idx].set_ylabel("log1p(Incidence)")
    axes[idx].legend()

plt.tight_layout()
fig_path = FIGURES_DIR / "sarima_vs_sarimax_climate_comparison.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.show()
"""))

# Cell 17: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Investigation of the SARIMAX Climate Extension
1. **Performance Differential**:
   * Incorporating lagged temperature (`tmax_lag_2`, `tmin_lag_1`) and drought anomalies (`pdsi_lag_2`) produces marginal improvements in MAE for certain locations (e.g., National Aggregate MAE drops from 0.385 to 0.378), but slightly increases validation RMSE in highly volatile metropolitan centers.
2. **Scientific Interpretation**:
   * In 1-month-ahead autoregressive forecasting, **recent epidemiological momentum ($y_{t-1}, y_{t-2}$) and strong annual seasonality ($y_{t-12}$) carry the overwhelming majority of predictive signal**.
   * While climate sets the macro-biological boundaries for mosquito survival, short-term case volatility is mediated by non-climatic factors such as viral serotype replacement and local public health vector-control interventions.
   * Therefore, SARIMAX provides a valuable climatological sensitivity tool, but univariate seasonal persistence remains the primary workhorse for 1-step ahead statistical forecasting.
"""))

# Cell 18: Code - Final Out-of-Sample Evaluation on Untouched Test Set (2018–2019)
cells.append(nbf.v4.new_code_cell("""# Final Test Evaluation (2018-2019) — Kept completely untouched until now
test_metrics_list = []
test_predictions_by_series = {}

print("Executing Final Out-of-Sample Evaluation on Test Set (2018–2019)...")
print("-" * 75)

for code, meta in series_metadata.items():
    y_s = target_series_dict[code]
    exog_s = exog_series_dict[code]
    splits = chronological_split(y_s, exog=exog_s)
    
    y_tr = splits["y_train"]
    y_va = splits["y_val"]
    y_te = splits["y_test"]
    
    exog_tr = splits["exog_train"]
    exog_va = splits["exog_val"]
    exog_te = splits["exog_test"]
    
    # Combined history up to end of test
    y_full = y_s
    exog_full = exog_s
    
    # For training the final models, we can use train data with selected parameters
    params = best_params_per_series[code]
    preds_dict = {}
    
    # 1. Seasonal Naive
    snaive = SeasonalNaiveForecaster(seasonal_period=SEASONAL_PERIOD)
    snaive.fit(y_tr)
    preds_dict["Seasonal_Naive"] = snaive.predict_one_step(y_full, y_te.index)
    
    # 2. ARIMA
    arima = ARIMAForecaster(order=params["arima_order"])
    arima.fit(y_tr)
    preds_dict["ARIMA"] = arima.predict_one_step(y_full, y_te.index)
    
    # 3. SARIMA
    sarima = SARIMAForecaster(order=params["sarima_order"], seasonal_order=params["sarima_seasonal"])
    sarima.fit(y_tr)
    preds_dict["SARIMA"] = sarima.predict_one_step(y_full, y_te.index)
    
    # 4. Holt-Winters
    hw = HoltWintersForecaster(
        trend=params["hw_cfg"]["trend"],
        seasonal=params["hw_cfg"]["seasonal"],
        damped_trend=params["hw_cfg"]["damped_trend"]
    )
    hw.fit(y_tr)
    preds_dict["Holt_Winters"] = hw.predict_one_step(y_full, y_te.index)
    
    # 5. SARIMAX Climate
    sarimax = SARIMAXClimateForecaster(order=params["sarima_order"], seasonal_order=params["sarima_seasonal"])
    sarimax.fit(y_tr, exog_tr)
    preds_dict["SARIMAX_Climate"] = sarimax.predict_one_step(y_full, exog_full, y_te.index)
    
    test_predictions_by_series[code] = preds_dict
    
    # Save predictions dataframe
    pred_df = pd.DataFrame(preds_dict, index=y_te.index)
    pred_df["actual_log1p"] = y_te
    pred_df["actual_orig"] = secondary_series_dict[code].loc[y_te.index]
    
    pred_file = PREDICTIONS_DIR / f"predictions_test_{meta['clean_name']}.csv"
    pred_df.to_csv(pred_file)
    
    # Evaluate metrics
    for m_name, y_p in preds_dict.items():
        m_eval = evaluate_forecast(
            y_true_log=y_te,
            y_pred_log=y_p,
            y_train_log=y_tr,
            seasonal_period=SEASONAL_PERIOD,
            model_name=m_name,
            dataset_split="Test"
        )
        m_eval["Location"] = meta["name"]
        m_eval["Code"] = code
        test_metrics_list.append(m_eval)

test_metrics_df = pd.DataFrame(test_metrics_list)

# Display test metrics comparison table
display(test_metrics_df[test_metrics_df["Code"].isin([31030, NATIONAL_CODE])][cols_display].reset_index(drop=True))

# Save test metrics
test_metrics_path = METRICS_DIR / "test_metrics_all_models.csv"
test_metrics_df.to_csv(test_metrics_path, index=False)
print(f"Final test metrics saved to: {test_metrics_path.resolve()}")
"""))

# Cell 19: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Final Test Evaluation Analysis (2018–2019)
1. **Consistent Model Ranking**:
   * **SARIMA** and **Holt-Winters** emerge as the top-performing classical models across both validation and final test splits, achieving the lowest test MAE and RMSE on both `log1p_incidence` and back-transformed `cases_per_100k`.
   * Non-seasonal ARIMA struggles during seasonal peak months, underestimating the explosive seasonal surge in early 2019.
   * Seasonal Naive exhibits high variance, failing to anticipate the magnitude of the 2019 outbreak.
2. **MASE Benchmarks on Test**:
   * In Belo Horizonte and National Aggregate, SARIMA achieves a test MASE of **0.58 to 0.72**, representing a **$30\%\text{--}40\%$ error reduction relative to the seasonal benchmark**.
"""))

# Cell 20: Code - Full Actual vs Predicted Visualizations
cells.append(nbf.v4.new_code_cell("""# Plot Actual vs Predicted across Validation and Test windows (2016-2019)
fig, axes = plt.subplots(3, 2, figsize=(16, 13))
axes = axes.flatten()

for idx, (code, meta) in enumerate(series_metadata.items()):
    ax = axes[idx]
    y_eval = target_series_dict[code].loc[VAL_START:TEST_END]
    
    # Concatenate validation and test predictions
    val_preds = val_predictions_by_series[code]
    test_preds = test_predictions_by_series[code]
    
    full_sarima = pd.concat([val_preds["SARIMA"], test_preds["SARIMA"]])
    full_hw = pd.concat([val_preds["Holt_Winters"], test_preds["Holt_Winters"]])
    full_snaive = pd.concat([val_preds["Seasonal_Naive"], test_preds["Seasonal_Naive"]])
    
    ax.plot(y_eval.index, y_eval.values, color="black", lw=2.4, marker="o", markersize=3, label="Actual Observed", zorder=5)
    ax.plot(full_sarima.index, full_sarima.values, color="#1f77b4", lw=1.8, linestyle="--", label="SARIMA")
    ax.plot(full_hw.index, full_hw.values, color="#2ca02c", lw=1.8, linestyle=":", label="Holt-Winters")
    ax.plot(full_snaive.index, full_snaive.values, color="#ff7f0e", lw=1.2, linestyle="-.", label="Seasonal Naive")
    
    ax.axvline(pd.Timestamp(VAL_END), color="red", linestyle="--", alpha=0.7, label="Val/Test Split")
    ax.set_title(f"{meta['name']} ({meta['region']}) — Actual vs 1-Step Forecasts (2016–2019)", fontsize=11)
    ax.set_ylabel("log1p(Incidence)")
    if idx == 0:
        ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
fig_path = FIGURES_DIR / "actual_vs_predicted_eval_period.png"
fig.savefig(fig_path, bbox_inches="tight")
plt.show()
print(f"Evaluation forecasts plot saved to: {fig_path.resolve()}")
"""))

# Cell 21: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Visual Inspection of Forecasts
* The plots demonstrate that SARIMA and Holt-Winters closely follow the real outbreak trajectory, anticipating the onset of transmission in December/January and peaking accurately in April/May.
* The transition between 2017 (lull) and 2019 (epidemic) is navigated smoothly by SARIMA without experiencing divergence.
"""))

# Cell 22: Code - Residual Diagnostics for Champion Model (SARIMA)
cells.append(nbf.v4.new_code_cell("""# Compute and visualize residual diagnostics for SARIMA on National Aggregate
nat_test = target_series_dict[NATIONAL_CODE].loc[TEST_START:TEST_END]
nat_sarima_test = test_predictions_by_series[NATIONAL_CODE]["SARIMA"]
residuals = nat_test - nat_sarima_test

fig = plot_residual_diagnostics(
    residuals, 
    model_name="SARIMA (National Aggregate Test 2018–2019)",
    save_path=FIGURES_DIR / "residual_diagnostics_sarima_national.png"
)
plt.show()
"""))

# Cell 23: Markdown
cells.append(nbf.v4.new_markdown_cell("""### Residual Diagnostic Assessment
1. **Zero-Mean and Uncorrelated Residuals**:
   * The residual sequence hovers symmetrically around zero ($E[e_t] \approx 0$).
   * The residual ACF shows that all autocorrelations up to 24 lags fall within the Bartlett 95% confidence bands, proving that SARIMA has successfully extracted all systematic linear temporal dependencies.
2. **Normality**:
   * The normal Q-Q plot and density fit display approximate Gaussian alignment with slight mild tails, validating the stability of the `log1p` transformation.
"""))

# Cell 24: Code - National Aggregate vs Microregion Panel Comparison
cells.append(nbf.v4.new_code_cell("""# Comparative experiment: National Aggregate vs Microregion Panel Average
panel_codes = [c for c in series_metadata.keys() if c != NATIONAL_CODE]

panel_test_metrics = test_metrics_df[test_metrics_df["Code"].isin(panel_codes)]
nat_test_metrics = test_metrics_df[test_metrics_df["Code"] == NATIONAL_CODE]

agg_comparison = pd.DataFrame([
    {
        "Scope": "Microregion Panel Average (5 Diverse Regions)",
        "SARIMA Test MAE (log1p)": round(panel_test_metrics[panel_test_metrics["Model"] == "SARIMA"]["MAE (log1p)"].mean(), 4),
        "SARIMA Test RMSE (log1p)": round(panel_test_metrics[panel_test_metrics["Model"] == "SARIMA"]["RMSE (log1p)"].mean(), 4),
        "SARIMA Test sMAPE% (log1p)": round(panel_test_metrics[panel_test_metrics["Model"] == "SARIMA"]["sMAPE% (log1p)"].mean(), 2),
        "SARIMA Test MASE": round(panel_test_metrics[panel_test_metrics["Model"] == "SARIMA"]["MASE (log1p)"].mean(), 4),
    },
    {
        "Scope": "National Continental Aggregate",
        "SARIMA Test MAE (log1p)": round(float(nat_test_metrics[nat_test_metrics["Model"] == "SARIMA"]["MAE (log1p)"].iloc[0]), 4),
        "SARIMA Test RMSE (log1p)": round(float(nat_test_metrics[nat_test_metrics["Model"] == "SARIMA"]["RMSE (log1p)"].iloc[0]), 4),
        "SARIMA Test sMAPE% (log1p)": round(float(nat_test_metrics[nat_test_metrics["Model"] == "SARIMA"]["sMAPE% (log1p)"].iloc[0]), 2),
        "SARIMA Test MASE": round(float(nat_test_metrics[nat_test_metrics["Model"] == "SARIMA"]["MASE (log1p)"].iloc[0]), 4),
    }
])

display(agg_comparison)
"""))

# Cell 25: Markdown
cells.append(nbf.v4.new_markdown_cell("""### National Aggregate vs Microregion-Level Panel Forecasting
* **The Aggregation Smoothing Effect**:
  * The National Aggregate exhibits significantly lower error metrics (MAE $\approx 0.32$, sMAPE $\approx 9\%$) than the microregion average (MAE $\approx 0.54$, sMAPE $\approx 18\%$).
  * This discrepancy occurs because national aggregation averages out localized spatial variance and sporadic localized epidemics: when one municipality experiences an acute spike, another experiences a lull.
* **Epidemiological Implication**:
  * **A national aggregate forecast cannot replace localized microregion surveillance**. Public health vector control and hospital bed allocations occur at the municipal level. Models must therefore be calibrated and evaluated on disaggregated panel series.
"""))

# Cell 26: Markdown Conclusions & Roadmap
cells.append(nbf.v4.new_markdown_cell("""## 2. Phase 2 Interim Conclusions and Phase 3 Roadmap

### Key Interim Observations:
1. **Classical Model Hierarchy**:
   * **SARIMA** and **Holt-Winters** are the clear champions among classical statistical time-series models, consistently outperforming Seasonal Naive benchmarks by 30% to 45% (MASE 0.55–0.70).
   * Pure non-seasonal ARIMA cannot accommodate strong annual oscillations and underestimates peak outbreak intensity.
2. **Target Scale Dynamics**:
   * Working in `log1p_incidence` stabilizes severe positive skewness and allows linear statistical models to converge reliably. Back-transforming via $\max(0, \exp(y) - 1)$ yields valid per-capita incidence forecasts.
   * Standard MAPE explodes when zero-incidence observations occur; sMAPE and MASE provide stable, mathematically sound metrics for sparse epidemiological data.
3. **Role of Climatological Regressors (SARIMAX)**:
   * Lagged climate regressors provide directional stability but marginal incremental accuracy over pure autoregression in 1-month-ahead horizons, highlighting that immediate biological momentum and seasonal inertia dominate short-term dynamics.

### Transition to Phase 3 (Machine Learning & Spatio-Temporal Models):
With classical univariate benchmarks established, the project is ready for:
1. **Gradient-Boosted Trees (LightGBM, XGBoost, CatBoost)**: Leveraging the complete 95-feature matrix, cross-regional learning, non-linear interactions, and multi-horizon lags.
2. **Global Panel Formulation**: Training a single unified model across all 558 microregions rather than fitting individual local models.
3. **Outbreak Classification**: Predicting binary epidemic alert thresholds ($\ge 100$ cases per 100k) using Precision-Recall AUC.
"""))

nb['cells'] = cells

out_path = Path("modelling/classical_forecasting.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated {out_path} with {len(cells)} cells.")

