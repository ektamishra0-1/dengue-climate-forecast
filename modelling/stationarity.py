"""
Stationarity analysis module for classical time-series forecasting.
Performs ADF tests, KPSS tests, rolling statistics, and differencing justifications
strictly fitted on the training split to avoid data leakage.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf


def run_adf_test(series: pd.Series, maxlag: int = 12) -> Dict[str, float]:
    """
    Augmented Dickey-Fuller Test
    H0 (Null Hypothesis): The series has a unit root (is non-stationary).
    H1 (Alternative): The series is stationary.
    Interpretation: If p-value < 0.05, reject H0 in favor of stationarity.
    """
    clean_series = series.dropna()
    res = adfuller(clean_series, maxlag=maxlag, autolag="AIC")
    return {
        "adf_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags_used": int(res[2]),
        "n_obs": int(res[3]),
        "crit_1%": float(res[4]["1%"]),
        "crit_5%": float(res[4]["5%"]),
        "crit_10%": float(res[4]["10%"]),
        "is_stationary": bool(res[1] < 0.05)
    }


def run_kpss_test(series: pd.Series, regression: str = "c") -> Dict[str, float]:
    """
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
    H0 (Null Hypothesis): The series is stationary around a constant level (or trend).
    H1 (Alternative): The series possesses a unit root (is non-stationary).
    Interpretation: If p-value < 0.05, reject H0 in favor of non-stationarity.
    """
    clean_series = series.dropna()
    # statsmodels kpss returns stat, p_value, lags, crit_dict
    res = kpss(clean_series, regression=regression, nlags="auto")
    return {
        "kpss_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags_used": int(res[2]),
        "crit_1%": float(res[3]["1%"]),
        "crit_5%": float(res[3]["5%"]),
        "crit_10%": float(res[3]["10%"]),
        "is_stationary": bool(res[1] >= 0.05)
    }


def compute_rolling_stats(series: pd.Series, window: int = 12) -> pd.DataFrame:
    """
    Compute backward rolling mean and rolling standard deviation to inspect variance/mean stability.
    """
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    
    return pd.DataFrame({
        "original": series,
        "rolling_mean": roll_mean,
        "rolling_std": roll_std
    }, index=series.index)


def evaluate_differencing_options(series: pd.Series, seasonal_period: int = 12) -> pd.DataFrame:
    """
    Systematically evaluate stationarity across:
    1. Raw level series (y_t)
    2. First-differenced series (Delta y_t = y_t - y_{t-1})
    3. Seasonally differenced series (Delta_12 y_t = y_t - y_{t-12})
    4. First + Seasonally differenced series (Delta Delta_12 y_t)
    
    Returns a structured diagnostic comparison table to guide (d, D) parameter selection.
    """
    diff_first = series.diff().dropna()
    diff_seasonal = series.diff(seasonal_period).dropna()
    diff_both = series.diff().diff(seasonal_period).dropna()
    
    transformations = {
        "Original Level (d=0, D=0)": series,
        "First Difference (d=1, D=0)": diff_first,
        "Seasonal Difference (d=0, D=1)": diff_seasonal,
        "First + Seasonal Diff (d=1, D=1)": diff_both,
    }
    
    records = []
    for name, s in transformations.items():
        adf = run_adf_test(s)
        kpss_res = run_kpss_test(s)
        
        # Joint interpretation:
        # Stationary: ADF rejects H0 (p < 0.05) AND KPSS fails to reject H0 (p >= 0.05)
        # Trend/Seasonal non-stationary: ADF fails to reject OR KPSS rejects
        is_joint_stationary = adf["is_stationary"] and kpss_res["is_stationary"]
        
        status = "Stationary (Both agree)" if is_joint_stationary else (
            "Non-Stationary" if (not adf["is_stationary"] and not kpss_res["is_stationary"]) else
            "Ambiguous / Weak Stationarity"
        )
        
        records.append({
            "Transformation": name,
            "ADF Stat": round(adf["adf_stat"], 4),
            "ADF p-val": round(adf["p_value"], 4),
            "ADF Decision": "Stationary" if adf["is_stationary"] else "Unit Root",
            "KPSS Stat": round(kpss_res["kpss_stat"], 4),
            "KPSS p-val": round(kpss_res["p_value"], 4),
            "KPSS Decision": "Stationary" if kpss_res["is_stationary"] else "Non-Stationary",
            "Joint Status": status,
            "Series Variance": round(float(s.var()), 4)
        })
        
    return pd.DataFrame(records)


def compute_acf_pacf_profile(series: pd.Series, nlags: int = 36) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ACF and PACF arrays up to nlags.
    """
    clean = series.dropna()
    acf_vals = acf(clean, nlags=nlags, fft=True)
    pacf_vals = pacf(clean, nlags=nlags, method="ywm")
    return acf_vals, pacf_vals

