"""
Evaluation module for classical time-series forecasting.
Computes MAE, RMSE, MAPE (with explicit zero/near-zero handling), sMAPE, and MASE
across both log1p incidence scale and original incidence-per-100k scale.
"""
from typing import Dict, Optional
import numpy as np
import pandas as pd
from modelling.preprocessing import inverse_log1p


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)"""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    threshold: float = 1.0
) -> Dict[str, float]:
    """
    Mean Absolute Percentage Error (MAPE).
    
    CRITICAL EPIDEMIOLOGICAL NOTE:
    Standard MAPE = mean(|y - y_hat| / |y|) divides by y. When actual incidence
    is zero or near-zero, MAPE explodes to infinity or divides by zero.
    
    Rather than silently substituting zeros with arbitrary constants, this function:
    1. Evaluates MAPE exclusively on non-near-zero observations (y_true >= threshold).
    2. Reports the proportion of observations evaluated.
    """
    mask = y_true >= threshold
    n_evaluated = int(np.sum(mask))
    total_obs = len(y_true)
    
    if n_evaluated == 0:
        return {
            "mape": np.nan,
            "mape_evaluated_pct": 0.0,
            "mape_excluded_count": total_obs
        }
    
    mape_val = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0)
    return {
        "mape": mape_val,
        "mape_evaluated_pct": float(n_evaluated / total_obs * 100.0),
        "mape_excluded_count": int(total_obs - n_evaluated)
    }


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE):
    sMAPE = (100 / n) * sum( 2 * |y - y_hat| / (|y| + |y_hat| + eps) )
    Bounded strictly in [0%, 200%], highly reliable for sparse/zero-inflated dengue series.
    """
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    smape_val = float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100.0)
    return smape_val


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 12
) -> float:
    """
    Mean Absolute Scaled Error (MASE):
    Compares the forecast MAE to the in-sample Seasonal Naive benchmark MAE on y_train.
    MASE < 1 indicates the model achieves superior accuracy to a Seasonal Naive persistence model.
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    
    # In-sample seasonal naive difference on training data
    if len(y_train) <= seasonal_period:
        return np.nan
    
    naive_diffs = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    scale = np.mean(naive_diffs)
    
    if scale == 0 or np.isnan(scale):
        return np.nan
    
    return float(mae_model / scale)


def evaluate_forecast(
    y_true_log: pd.Series | np.ndarray,
    y_pred_log: pd.Series | np.ndarray,
    y_train_log: Optional[pd.Series | np.ndarray] = None,
    seasonal_period: int = 12,
    model_name: str = "Model",
    dataset_split: str = "Validation"
) -> Dict[str, float | str]:
    """
    Compute full evaluation metrics across both:
    1. Primary log1p_incidence scale
    2. Original incidence-per-100k scale (back-transformed via expm1)
    """
    yt_log = np.asarray(y_true_log, dtype=float)
    yp_log = np.asarray(y_pred_log, dtype=float)
    
    # 1. Primary scale (log1p_incidence)
    mae_log = mean_absolute_error(yt_log, yp_log)
    rmse_log = root_mean_squared_error(yt_log, yp_log)
    smape_log = symmetric_mape(yt_log, yp_log)
    
    mase_log = np.nan
    if y_train_log is not None:
        y_tr_log = np.asarray(y_train_log, dtype=float)
        mase_log = mean_absolute_scaled_error(yt_log, yp_log, y_tr_log, seasonal_period)
    
    # 2. Original scale (cases_per_100k)
    yt_orig = inverse_log1p(yt_log)
    yp_orig = inverse_log1p(yp_log)
    
    mae_orig = mean_absolute_error(yt_orig, yp_orig)
    rmse_orig = root_mean_squared_error(yt_orig, yp_orig)
    smape_orig = symmetric_mape(yt_orig, yp_orig)
    mape_orig_info = mean_absolute_percentage_error(yt_orig, yp_orig, threshold=1.0)
    
    mase_orig = np.nan
    if y_train_log is not None:
        y_tr_orig = inverse_log1p(y_tr_log)
        mase_orig = mean_absolute_scaled_error(yt_orig, yp_orig, y_tr_orig, seasonal_period)
        
    return {
        "Model": model_name,
        "Split": dataset_split,
        # Log1p Scale
        "MAE (log1p)": round(mae_log, 4),
        "RMSE (log1p)": round(rmse_log, 4),
        "sMAPE% (log1p)": round(smape_log, 2),
        "MASE (log1p)": round(mase_log, 4) if not np.isnan(mase_log) else np.nan,
        # Original Scale (cases per 100k)
        "MAE (orig)": round(mae_orig, 2),
        "RMSE (orig)": round(rmse_orig, 2),
        "sMAPE% (orig)": round(smape_orig, 2),
        "MAPE% (orig, >=1)": round(mape_orig_info["mape"], 2) if not np.isnan(mape_orig_info["mape"]) else np.nan,
        "MAPE_eval_pct": round(mape_orig_info["mape_evaluated_pct"], 1),
        "MASE (orig)": round(mase_orig, 4) if not np.isnan(mase_orig) else np.nan,
    }

