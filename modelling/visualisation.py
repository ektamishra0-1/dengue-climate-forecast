"""
Visualization module for classical time-series forecasting.
Generates publication-quality figures saved directly into results/figures/.
"""
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

from modelling.config import FIGURES_DIR, TRAIN_END, VAL_END, TEST_END

# Global plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11


def plot_historical_incidence(
    series: pd.Series,
    title: str,
    save_path: Optional[Path] = None,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END
) -> plt.Figure:
    """
    Plot historical incidence trajectory with shaded chronological partition boundaries.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, color="#1f77b4", lw=1.8, label="Observed Target (log1p_incidence)")
    
    # Highlight splits
    ax.axvspan(series.index[0], pd.Timestamp(train_end), color="#2ca02c", alpha=0.10, label="Training (2002–2015)")
    ax.axvspan(pd.Timestamp(train_end), pd.Timestamp(val_end), color="#ff7f0e", alpha=0.15, label="Validation (2016–2017)")
    ax.axvspan(pd.Timestamp(val_end), series.index[-1], color="#d62728", alpha=0.12, label="Test (2018–2019)")
    
    ax.axvline(pd.Timestamp(train_end), color="#333333", linestyle="--", lw=1.2)
    ax.axvline(pd.Timestamp(val_end), color="#333333", linestyle="--", lw=1.2)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("log1p(Incidence per 100k)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_rolling_stationarity(
    series: pd.Series,
    title: str,
    window: int = 12,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot level series alongside 12-month rolling mean and standard deviation.
    """
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series.index, series.values, color="#7f7f7f", alpha=0.6, lw=1.2, label="Monthly Observations")
    ax.plot(roll_mean.index, roll_mean.values, color="#d62728", lw=2.2, label=f"{window}-Month Rolling Mean")
    ax.plot(roll_std.index, roll_std.values, color="#1f77b4", lw=2.0, linestyle="--", label=f"{window}-Month Rolling Std")
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("log1p_incidence")
    ax.legend(loc="upper left")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_acf_pacf_diagnostics(
    series: pd.Series,
    title: str,
    lags: int = 36,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate side-by-side ACF and PACF plots.
    """
    clean_series = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    
    plot_acf(clean_series, lags=lags, ax=axes[0], color="#1f77b4", title=f"ACF - {title}")
    plot_pacf(clean_series, lags=lags, ax=axes[1], color="#d62728", method="ywm", title=f"PACF - {title}")
    
    axes[0].set_xlabel("Lag (Months)")
    axes[1].set_xlabel("Lag (Months)")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(
    actual: pd.Series,
    predictions: Dict[str, pd.Series],
    title: str,
    val_end: str = VAL_END,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot actual observations against 1-step ahead out-of-sample forecasts across all models.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(actual.index, actual.values, color="black", lw=2.5, marker="o", markersize=4, label="Actual Observed", zorder=5)
    
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#e377c2", "#8c564b"]
    for i, (name, pred_series) in enumerate(predictions.items()):
        color = palette[i % len(palette)]
        ax.plot(pred_series.index, pred_series.values, lw=1.8, linestyle="--", marker="x", markersize=3, label=name, color=color)
        
    if pd.Timestamp(val_end) in actual.index:
        ax.axvline(pd.Timestamp(val_end), color="red", linestyle=":", lw=1.5, label="Validation / Test Split")
        
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("log1p(Incidence)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_residual_diagnostics(
    residuals: pd.Series,
    model_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    4-panel diagnostic plot for forecast residuals:
    1. Residual sequence over time
    2. Residual histogram + KDE vs standard normal
    3. Normal Q-Q probability plot
    4. Residual ACF
    """
    clean_res = residuals.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    # 1. Residuals over time
    axes[0, 0].plot(clean_res.index, clean_res.values, color="#1f77b4", lw=1.5)
    axes[0, 0].axhline(0, color="red", linestyle="--", alpha=0.8)
    axes[0, 0].set_title(f"Residuals Over Time: {model_name}")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Residual (e_t)")
    
    # 2. Histogram + KDE
    sns.histplot(clean_res, kde=True, color="#2ca02c", ax=axes[0, 1], stat="density")
    x_grid = np.linspace(clean_res.min(), clean_res.max(), 100)
    mu, sigma = clean_res.mean(), clean_res.std()
    axes[0, 1].plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), color="red", linestyle="--", label="Normal Fit")
    axes[0, 1].set_title("Residual Density vs Normal Fit")
    axes[0, 1].legend()
    
    # 3. Normal Q-Q plot
    stats.probplot(clean_res, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Normal Q-Q Plot")
    
    # 4. Residual ACF (max lags bounded by sample size)
    max_lags = min(12, len(clean_res) // 2)
    plot_acf(clean_res, lags=max_lags, ax=axes[1, 1], color="#ff7f0e", title=f"Residual Autocorrelation (ACF, {max_lags} Lags)")
    axes[1, 1].set_xlabel("Lag")
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_forecast_errors_over_time(
    actual: pd.Series,
    predictions: Dict[str, pd.Series],
    title: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot time series of prediction errors (y_true - y_pred) across models.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#e377c2"]
    
    for i, (name, pred_series) in enumerate(predictions.items()):
        error = actual.loc[pred_series.index] - pred_series
        ax.plot(error.index, error.values, lw=1.5, label=name, color=palette[i % len(palette)])
        
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Error (y - y_hat)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_metric_comparison_bars(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate horizontal bar chart comparing models by an evaluation metric.
    """
    df_sorted = metrics_df.sort_values(metric_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    bars = ax.barh(df_sorted["Model"], df_sorted[metric_col], color="#1f77b4", height=0.55)
    ax.set_title(title)
    ax.set_xlabel(metric_col)
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.02), bar.get_y() + bar.get_height()/2, 
                f"{width:.3f}", va="center", ha="left", fontsize=10)
        
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

