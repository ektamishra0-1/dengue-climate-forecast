"""
Forecasting models module for classical time-series approaches.
Implements:
1. Seasonal Naive (Benchmark)
2. ARIMA (Non-seasonal)
3. SARIMA (Seasonal)
4. Holt-Winters (Triple Exponential Smoothing)
5. SARIMAX (Seasonal with Exogenous Climate Regressors)

All models support strictly leak-free 1-step ahead rolling out-of-sample forecasting.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress known convergence warnings during grid search
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Benchmark Forecaster.
    Forecast rule: y_hat_{t} = y_{t - m}, where m is seasonal_period (default 12).
    Requires no parameters and represents the simplest seasonal persistence benchmark.
    """
    def __init__(self, seasonal_period: int = 12):
        self.seasonal_period = seasonal_period
        self.y_train: Optional[pd.Series] = None
        
    def fit(self, y_train: pd.Series) -> "SeasonalNaiveForecaster":
        self.y_train = y_train.copy()
        return self
        
    def predict_one_step(
        self, 
        y_history: pd.Series, 
        horizon_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Produce 1-step ahead forecasts across horizon_index.
        For each date t in horizon_index, prediction is the observation from t - seasonal_period.
        """
        full_series = y_history.copy()
        predictions = []
        for date in horizon_index:
            lag_date = date - pd.DateOffset(months=self.seasonal_period)
            if lag_date in full_series.index:
                pred = full_series.loc[lag_date]
            else:
                # Fallback to closest available prior seasonal lag
                pred = full_series.iloc[-self.seasonal_period]
            predictions.append(pred)
            
        return pd.Series(predictions, index=horizon_index, name="pred_seasonal_naive")


class ARIMAForecaster:
    """
    Non-Seasonal ARIMA(p, d, q) Forecaster using State Space formulation.
    """
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model_res: Optional[SARIMAXResults] = None
        self.y_train: Optional[pd.Series] = None
        
    def fit(self, y_train: pd.Series) -> "ARIMAForecaster":
        self.y_train = y_train.copy()
        model = SARIMAX(
            y_train,
            order=self.order,
            trend="c" if self.order[1] == 0 else "n",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_res = model.fit(disp=False, maxiter=200)
        return self
        
    def predict_one_step(
        self, 
        y_history: pd.Series, 
        horizon_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Generate 1-step ahead rolling forecasts over horizon_index using state space filtering.
        """
        if self.model_res is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        # Apply fitted parameters to the extended sequence
        extended_res = self.model_res.apply(y_history)
        preds = extended_res.predict(
            start=horizon_index[0], 
            end=horizon_index[-1], 
            dynamic=False
        )
        preds.name = f"pred_arima_{self.order[0]}_{self.order[1]}_{self.order[2]}"
        return preds


class SARIMAForecaster:
    """
    Seasonal ARIMA(p, d, q) x (P, D, Q)[m] Forecaster.
    """
    def __init__(
        self, 
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 12)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res: Optional[SARIMAXResults] = None
        self.y_train: Optional[pd.Series] = None
        
    def fit(self, y_train: pd.Series) -> "SARIMAForecaster":
        self.y_train = y_train.copy()
        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="c" if self.order[1] == 0 and self.seasonal_order[1] == 0 else "n",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_res = model.fit(disp=False, maxiter=200)
        return self
        
    def predict_one_step(
        self, 
        y_history: pd.Series, 
        horizon_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Generate 1-step ahead rolling forecasts over horizon_index using state space filtering.
        """
        if self.model_res is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        extended_res = self.model_res.apply(y_history)
        preds = extended_res.predict(
            start=horizon_index[0], 
            end=horizon_index[-1], 
            dynamic=False
        )
        preds.name = f"pred_sarima_{self.order}_{self.seasonal_order}"
        return preds


class HoltWintersForecaster:
    """
    Holt-Winters Triple Exponential Smoothing Forecaster.
    Models level, trend, and annual seasonality (seasonal_periods=12).
    """
    def __init__(
        self, 
        trend: Optional[str] = "add", 
        seasonal: Optional[str] = "add", 
        seasonal_periods: int = 12,
        damped_trend: bool = False
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.hw_fit: Any = None
        self.y_train: Optional[pd.Series] = None
        
    def fit(self, y_train: pd.Series) -> "HoltWintersForecaster":
        self.y_train = y_train.copy()
        
        # Check non-negativity if multiplicative
        if self.seasonal == "mul" and (y_train <= 0).any():
            # If zeros exist, shift or default to additive
            self.seasonal = "add"
            
        model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
            initialization_method="estimated"
        )
        self.hw_fit = model.fit(optimized=True)
        return self
        
    def predict_one_step(
        self, 
        y_history: pd.Series, 
        horizon_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Execute rolling 1-step ahead predictions across horizon_index
        using fixed parameters learned on training data.
        """
        if self.hw_fit is None or self.y_train is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        params = self.hw_fit.params
        train_len = len(self.y_train)
        preds = []
        
        # Rolling 1-step ahead evaluation
        for i, dt in enumerate(horizon_index):
            # Known history up to current step t-1
            current_hist = y_history.loc[:dt].iloc[:-1]
            if len(current_hist) < self.seasonal_periods * 2:
                current_hist = self.y_train
                
            step_model = ExponentialSmoothing(
                current_hist,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend,
                initialization_method="estimated"
            )
            # Filter with fixed parameters
            step_fit = step_model.fit(
                smoothing_level=params.get("smoothing_level"),
                smoothing_trend=params.get("smoothing_trend"),
                smoothing_seasonal=params.get("smoothing_seasonal"),
                damping_trend=params.get("damping_trend"),
                optimized=False
            )
            p = float(step_fit.forecast(1).iloc[0])
            preds.append(p)
            
        return pd.Series(preds, index=horizon_index, name=f"pred_hw_{self.trend}_{self.seasonal}")


class SARIMAXClimateForecaster:
    """
    SARIMAX with Exogenous Climate Variables Forecaster.
    Integrates lagged environmental features (e.g., tmax_lag_2, tmin_lag_1, pdsi_lag_2).
    """
    def __init__(
        self, 
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 12)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res: Optional[SARIMAXResults] = None
        self.y_train: Optional[pd.Series] = None
        self.exog_train: Optional[pd.DataFrame] = None
        
    def fit(self, y_train: pd.Series, exog_train: pd.DataFrame) -> "SARIMAXClimateForecaster":
        self.y_train = y_train.copy()
        self.exog_train = exog_train.copy()
        
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="c" if self.order[1] == 0 and self.seasonal_order[1] == 0 else "n",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_res = model.fit(disp=False, maxiter=200)
        return self
        
    def predict_one_step(
        self, 
        y_history: pd.Series, 
        exog_history: pd.DataFrame,
        horizon_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Generate 1-step ahead rolling forecasts incorporating observed exogenous climate indicators.
        """
        if self.model_res is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        extended_res = self.model_res.apply(y_history, exog=exog_history)
        preds = extended_res.predict(
            start=horizon_index[0], 
            end=horizon_index[-1], 
            exog=exog_history.loc[horizon_index[0]:horizon_index[-1]],
            dynamic=False
        )
        preds.name = f"pred_sarimax_climate_{self.order}_{self.seasonal_order}"
        return preds


def select_best_arima_parameters(
    y_train: pd.Series, 
    y_val: pd.Series,
    candidate_orders: Optional[List[Tuple[int, int, int]]] = None
) -> Tuple[Tuple[int, int, int], pd.DataFrame]:
    """
    Evaluate candidate ARIMA orders on training data and validation performance.
    """
    if candidate_orders is None:
        candidate_orders = [
            (1, 0, 0), (1, 0, 1), (2, 0, 1),
            (1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2)
        ]
        
    results = []
    best_order = candidate_orders[0]
    best_val_mae = float("inf")
    
    y_hist_val = pd.concat([y_train, y_val])
    
    for order in candidate_orders:
        try:
            model = ARIMAForecaster(order=order)
            model.fit(y_train)
            aic = float(model.model_res.aic)
            bic = float(model.model_res.bic)
            
            # Predict 1-step ahead on validation
            val_preds = model.predict_one_step(y_hist_val, y_val.index)
            val_mae = float(np.mean(np.abs(y_val - val_preds)))
            
            results.append({
                "order": str(order),
                "AIC": round(aic, 2),
                "BIC": round(bic, 2),
                "Val_MAE": round(val_mae, 4)
            })
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_order = order
        except Exception:
            continue
            
    res_df = pd.DataFrame(results).sort_values("Val_MAE").reset_index(drop=True)
    return best_order, res_df


def select_best_sarima_parameters(
    y_train: pd.Series, 
    y_val: pd.Series,
    candidate_configs: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]] = None
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], pd.DataFrame]:
    """
    Evaluate candidate SARIMA orders on training data and validation performance.
    """
    if candidate_configs is None:
        candidate_configs = [
            ((1, 0, 1), (1, 1, 0, 12)),
            ((1, 1, 1), (1, 1, 0, 12)),
            ((1, 1, 1), (0, 1, 1, 12)),
            ((2, 1, 1), (1, 1, 0, 12)),
            ((1, 0, 1), (0, 1, 1, 12)),
            ((2, 0, 1), (1, 1, 0, 12)),
        ]
        
    results = []
    best_cfg = candidate_configs[0]
    best_val_mae = float("inf")
    y_hist_val = pd.concat([y_train, y_val])
    
    for order, s_order in candidate_configs:
        try:
            model = SARIMAForecaster(order=order, seasonal_order=s_order)
            model.fit(y_train)
            aic = float(model.model_res.aic)
            bic = float(model.model_res.bic)
            
            val_preds = model.predict_one_step(y_hist_val, y_val.index)
            val_mae = float(np.mean(np.abs(y_val - val_preds)))
            
            results.append({
                "order": str(order),
                "seasonal_order": str(s_order),
                "AIC": round(aic, 2),
                "BIC": round(bic, 2),
                "Val_MAE": round(val_mae, 4)
            })
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_cfg = (order, s_order)
        except Exception:
            continue
            
    res_df = pd.DataFrame(results).sort_values("Val_MAE").reset_index(drop=True)
    return best_cfg[0], best_cfg[1], res_df


def select_best_holtwinters_parameters(
    y_train: pd.Series, 
    y_val: pd.Series
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate Holt-Winters options (additive vs multiplicative, damped trend) on validation set.
    """
    configs = [
        {"trend": "add", "seasonal": "add", "damped_trend": False, "name": "HW_Add_Add"},
        {"trend": "add", "seasonal": "add", "damped_trend": True, "name": "HW_Add_Damped_Add"},
        {"trend": None, "seasonal": "add", "damped_trend": False, "name": "HW_NoTrend_Add"},
    ]
    
    results = []
    best_cfg = configs[0]
    best_val_mae = float("inf")
    y_hist_val = pd.concat([y_train, y_val])
    
    for cfg in configs:
        try:
            model = HoltWintersForecaster(
                trend=cfg["trend"], 
                seasonal=cfg["seasonal"], 
                damped_trend=cfg["damped_trend"]
            )
            model.fit(y_train)
            val_preds = model.predict_one_step(y_hist_val, y_val.index)
            val_mae = float(np.mean(np.abs(y_val - val_preds)))
            
            results.append({
                "Configuration": cfg["name"],
                "Trend": str(cfg["trend"]),
                "Seasonal": str(cfg["seasonal"]),
                "Damped": str(cfg["damped_trend"]),
                "Val_MAE": round(val_mae, 4)
            })
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_cfg = cfg
        except Exception:
            continue
            
    res_df = pd.DataFrame(results).sort_values("Val_MAE").reset_index(drop=True)
    return best_cfg, res_df

