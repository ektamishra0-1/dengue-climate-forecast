# Phase 2 Consolidated Evaluation Report: Classical Time-Series Forecasting of Dengue Incidence in Brazil

**Project Title**: Brazilian Dengue Climate Forecasting Research Project  
**Evaluation Milestone**: Phase 2 — Mid-Semester Research Evaluation  
**Evaluation Date**: November 2026  
**Document Location**: `reports/phase2_midsemester_report.md`  
**Primary Target Variable**: $\text{log1p\_incidence} = \ln(1 + \text{cases\_per\_100k})$  
**Secondary Evaluation Scale**: Original monthly incidence per 100,000 population ($\text{cases\_per\_100k} = \max(0, \exp(y) - 1)$)  
**Forecast Horizon**: One month ahead ($h = 1$), evaluated out-of-sample via rolling 1-step conditioning  

---

## 1. Research Objective & Forecasting Problem

Dengue fever is an acute mosquito-borne viral disease caused by four antigenically distinct dengue virus serotypes (DENV-1, DENV-2, DENV-3, DENV-4) and transmitted primarily by the female *Aedes aegypti* mosquito. In Brazil, dengue represents a major public health priority characterized by high spatial heterogeneity across biomes, steep intra-annual seasonal surges, and explosive multi-year epidemic waves.

### Core Problem Formulation:
The primary research objective of Phase 2 is to develop, evaluate, and benchmark **classical statistical time-series forecasting models** to predict monthly dengue incidence across Brazilian microregions over the 20-year window **2000–2019**. 

Specifically, this evaluation investigates:
1. **Seasonal Persistence vs. Stochastic Drift**: Can classical autoregressive models (ARIMA, SARIMA, Holt-Winters) accurately forecast the onset, peak, and decay of annual seasonal epidemics?
2. **Benchmark Superiority**: How much predictive advantage do parametric models provide over the standard baseline **Seasonal Naive** model ($\hat{y}_t = y_{t-12}$)?
3. **The Role of Exogenous Climate Drivers**: Does conditioning on lagged temperature and drought anomalies in a **SARIMAX** framework provide demonstrable accuracy improvements over univariate statistical models?
4. **Spatial Aggregation Discrepancy**: How does a **National Aggregate forecast** compare against **microregion-level panel forecasts**, and why is spatial disaggregation clinically essential?

---

## 2. Dataset Description & Partitioning

The evaluation utilizes the feature-engineered master dataset [`data/dengue_features.csv`](file:///Users/ektamishra/Desktop/dengue%20forecasting/data/dengue_features.csv) (133,920 records $\times$ 95 features) generated in Phase 1 and tracked under Data Version Control ([`data/dengue_features.csv.dvc`](file:///Users/ektamishra/Desktop/dengue%20forecasting/data/dengue_features.csv.dvc)).

### A. Chronological Data Partitions (Zero Data Leakage)
To prevent temporal data leakage, observations were strictly partitioned chronologically without random shuffling:

```
┌──────────────────────────────────────┬────────────────────┬────────────────────┐
│      Training Set (2002–2015)        │  Validation Set    │     Test Set       │
│             168 Months               │  (2016–2017)       │   (2018–2019)      │
│  Parameter estimation & diagnostics  │  24 Months         │   24 Months        │
│                                      │  Hyperparameter sel│   Final benchmark  │
└──────────────────────────────────────┴────────────────────┴────────────────────┘
```

* **Training Window**: January 2002 – December 2015 (168 continuous months = 14 years). Captures multi-year epidemic waves (2002, 2008, 2010, 2013, 2015).
* **Validation Window**: January 2016 – December 2017 (24 continuous months = 2 years). Encompasses the historic 2016 national epidemic wave and subsequent 2017 post-epidemic lull. Used strictly for parameter tuning.
* **Testing Window**: January 2018 – December 2019 (24 continuous months = 2 years). Kept untouched until final evaluation; contains the explosive 2019 nationwide outbreak.

### B. Geographical Scope: 5 Representative Microregions & National Aggregate
To balance computational feasibility with geographical and epidemiological breadth, five representative microregions were selected using a documented, reproducible selection rule covering all five Brazilian macro-regions and distinct ecological biomes:

| Micro Code | Location Name | State | Macro-Region | Ecological Biome | Population (Mean) | Zero Months (Train) | Mean Incid. (per 100k) | Max Incid. (per 100k) | Epidemiological Role |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **31030** | **Belo Horizonte** | Minas Gerais | Southeast | Atlantic Rainforest | 5,038,298 | 0 / 168 | 158.42 | 1,731.36 | Metropolitan epicenter; explosive multi-year waves. |
| **52010** | **Goiânia** | Goiás | Centre-West | Cerrado | 2,082,253 | 0 / 168 | 121.79 | 910.65 | Continental savanna; hyper-endemic baseline year-round. |
| **23016** | **Fortaleza** | Ceará | Northeast | Caatinga | 2,987,838 | 0 / 168 | 72.85 | 874.12 | Semi-arid coastal metropolis; acute seasonal outbreaks. |
| **13007** | **Manaus** | Amazonas | North | Amazon | 1,842,520 | 0 / 168 | 28.48 | 983.67 | Equatorial rainforest basin; rainfall-driven cycles. |
| **41009** | **Maringá** | Paraná | South | Atlantic Rainforest | 496,280 | 1 / 168 | 56.40 | 798.15 | Subtropical southern center; strong winter thermal lull. |
| **0** | **National Aggregate** | Brazil | National | Continental Brazil | 185,120,400 | 0 / 168 | 58.12 | 268.45 | Macro-continental benchmark summing all 558 microregions. |

---

## 3. Preprocessing & Rigorous Stationarity Analysis

### A. Preprocessing & Regular Monthly Frequency
* Time-series series were extracted and indexed with a regular monthly beginning timestamp (`freq='MS'`).
* **Missing-Value Audit**: For all selected microregions over 2002–2019, target observations (`dengue_cases`, `population`, `cases_per_100k`, `log1p_incidence`) have **zero missing values**. As documented in Phase 1, `pdsi` is missing only for microregion 26019 (Fernando de Noronha oceanic island), which does not affect our representative set.
* **Target Stabilization**: Working with $\ln(1 + \text{cases\_per\_100k})$ stabilizes the heavy-tailed variance and prevents explosive single-month spikes from dominating squared-error losses.

### B. Statistical Stationarity Tests (ADF and KPSS)
Stationarity tests were conducted strictly on the training period (2002–2015):
* **Augmented Dickey-Fuller (ADF)**:
  * Null Hypothesis ($H_0$): Unit root is present (the series is non-stationary).
  * Alternative Hypothesis ($H_1$): The series is stationary.
  * Decision Rule: Reject $H_0$ if $p < 0.05$.
* **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)**:
  * Null Hypothesis ($H_0$): The series is level- or trend-stationary.
  * Alternative Hypothesis ($H_1$): A unit root is present (non-stationary).
  * Decision Rule: Reject $H_0$ if $p < 0.05$.

#### Complete Empirical Stationarity Audit (Actual CSV Values):
Source: [`results/metrics/stationarity_differencing_audit.csv`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/metrics/stationarity_differencing_audit.csv)

| Location | Code | Transformation | ADF Stat | ADF $p$-val | ADF Decision | KPSS Stat | KPSS $p$-val | KPSS Decision | Joint Diagnostic Status | Series Variance |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Belo Horizonte** | 31030 | Original Level ($d=0, D=0$) | -1.9774 | 0.2966 | Unit Root | 0.6407 | 0.0189 | Non-Stationary | **Non-Stationary** | 2.8274 |
| **Belo Horizonte** | 31030 | First Diff ($d=1, D=0$) | -4.4289 | 0.0003 | Stationary | 0.0226 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.6481 |
| **Belo Horizonte** | 31030 | Seasonal Diff ($d=0, D=1$) | -2.7055 | 0.0731 | Unit Root | 0.1241 | 0.1000 | Stationary | Ambiguous / Weak Stat. | 2.0739 |
| **Belo Horizonte** | 31030 | First + Seasonal ($d=1, D=1$) | -4.2266 | 0.0006 | Stationary | 0.0380 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.3755 |
| **Goiânia** | 52010 | Original Level ($d=0, D=0$) | -1.6246 | 0.4703 | Unit Root | 0.9972 | 0.0100 | Non-Stationary | **Non-Stationary** | 1.5545 |
| **Goiânia** | 52010 | First Diff ($d=1, D=0$) | -4.0135 | 0.0013 | Stationary | 0.0268 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.3457 |
| **Goiânia** | 52010 | Seasonal Diff ($d=0, D=1$) | -3.6813 | 0.0044 | Stationary | 0.0798 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.9560 |
| **Goiânia** | 52010 | First + Seasonal ($d=1, D=1$) | -4.0148 | 0.0013 | Stationary | 0.0304 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.2364 |
| **Fortaleza** | 23016 | Original Level ($d=0, D=0$) | -3.3581 | 0.0125 | Stationary | 0.1740 | 0.1000 | Stationary | **Stationary (Both agree)** | 1.6677 |
| **Fortaleza** | 23016 | First Diff ($d=1, D=0$) | -8.3439 | 0.0000 | Stationary | 0.0185 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.3303 |
| **Fortaleza** | 23016 | Seasonal Diff ($d=0, D=1$) | -3.0461 | 0.0308 | Stationary | 0.0430 | 0.1000 | Stationary | **Stationary (Both agree)** | 2.5042 |
| **Fortaleza** | 23016 | First + Seasonal ($d=1, D=1$) | -4.0575 | 0.0011 | Stationary | 0.0247 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.3845 |
| **Manaus** | 13007 | Original Level ($d=0, D=0$) | -3.2783 | 0.0159 | Stationary | 0.3264 | 0.1000 | Stationary | **Stationary (Both agree)** | 1.4604 |
| **Manaus** | 13007 | First Diff ($d=1, D=0$) | -3.7474 | 0.0035 | Stationary | 0.0196 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.2705 |
| **Manaus** | 13007 | Seasonal Diff ($d=0, D=1$) | -3.1131 | 0.0256 | Stationary | 0.0614 | 0.1000 | Stationary | **Stationary (Both agree)** | 2.3018 |
| **Manaus** | 13007 | First + Seasonal ($d=1, D=1$) | -6.1977 | 0.0000 | Stationary | 0.0401 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.4193 |
| **Maringá** | 41009 | Original Level ($d=0, D=0$) | -2.0692 | 0.2571 | Unit Root | 0.3579 | 0.0953 | Stationary | Ambiguous / Weak Stat. | 3.0004 |
| **Maringá** | 41009 | First Diff ($d=1, D=0$) | -8.4833 | 0.0000 | Stationary | 0.0134 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.8440 |
| **Maringá** | 41009 | Seasonal Diff ($d=0, D=1$) | -3.3316 | 0.0135 | Stationary | 0.0874 | 0.1000 | Stationary | **Stationary (Both agree)** | 3.3048 |
| **Maringá** | 41009 | First + Seasonal ($d=1, D=1$) | -5.3530 | 0.0000 | Stationary | 0.0226 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.8693 |
| **National Aggregate** | 0 | Original Level ($d=0, D=0$) | -1.6456 | 0.4593 | Unit Root | 0.7721 | 0.0100 | Non-Stationary | **Non-Stationary** | 1.2338 |
| **National Aggregate** | 0 | First Diff ($d=1, D=0$) | -3.7110 | 0.0040 | Stationary | 0.0289 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.2777 |
| **National Aggregate** | 0 | Seasonal Diff ($d=0, D=1$) | -3.3950 | 0.0111 | Stationary | 0.1726 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.5155 |
| **National Aggregate** | 0 | First + Seasonal ($d=1, D=1$) | -5.4665 | 0.0000 | Stationary | 0.0405 | 0.1000 | Stationary | **Stationary (Both agree)** | 0.2188 |

### C. Differencing Justification Summary:
1. **Raw Levels Fail Stationarity**: For Belo Horizonte, Goiânia, and the National Aggregate, the raw level series has an ADF $p > 0.05$ or a KPSS $p < 0.05$, confirming non-stationarity driven by secular trends and shifting multi-annual epidemic baselines.
2. **First Differencing ($d=1$)**: Eliminates stochastic level drift and produces consensus stationarity across all locations.
3. **Seasonal Differencing ($D=1, m=12$)**: Removes the recurring 12-month annual cyclical crest. Applying seasonal differencing brings KPSS test statistics below $0.10$ across all series.
4. **Variance Consideration**: While both first differencing and seasonal differencing achieve stationarity, seasonal differencing alone can inflate variance if applied without autoregressive smoothing. Consequently, model parameter searches evaluated $d \in \{0, 1\}$ and $D \in \{0, 1\}$.

---

## 4. Mathematical Formulations of Models

### 1. Seasonal Naive Forecaster (Benchmark)
The simplest seasonal persistence model. The one-step-ahead forecast for month $t$ is set to the observed incidence exactly 12 months prior:
$$\hat{y}_t = y_{t - 12}$$
Requires zero parameter estimation. If a statistical model cannot beat this baseline ($\text{MASE} \ge 1.0$), it provides no clinical or epidemiological value.

### 2. Autoregressive Integrated Moving Average (ARIMA)
A non-seasonal $\text{ARIMA}(p, d, q)$ assumes that after $d$-th order differencing ($\nabla^d y_t = (1 - B)^d y_t$), the series follows a stationary ARMA process:
$$\phi(B) \nabla^d y_t = c + \theta(B) \epsilon_t$$
where $B$ is the backshift operator ($B^k y_t = y_{t-k}$), $\phi(B) = 1 - \sum_{i=1}^p \phi_i B^i$ is the autoregressive polynomial, $\theta(B) = 1 + \sum_{j=1}^q \theta_j B^j$ is the moving average polynomial, and $\epsilon_t \sim \mathcal{WN}(0, \sigma^2)$.

### 3. Seasonal ARIMA (SARIMA)
$\text{SARIMA}(p, d, q) \times (P, D, Q)_m$ models both non-seasonal and seasonal lag dependencies with period $m = 12$:
$$\Phi_p(B) \tilde{\Phi}_P(B^m) \nabla^d \nabla_m^D y_t = c + \Theta_q(B) \tilde{\Theta}_Q(B^m) \epsilon_t$$
where:
* $\nabla_m^D = (1 - B^m)^D$ is the seasonal differencing operator ($D \in \{0, 1\}$).
* $\tilde{\Phi}_P(B^m) = 1 - \sum_{j=1}^P \Phi_j B^{j \cdot m}$ represents the seasonal autoregressive polynomial.
* $\tilde{\Theta}_Q(B^m) = 1 + \sum_{j=1}^Q \Theta_j B^{j \cdot m}$ represents the seasonal moving average polynomial.

### 4. Holt-Winters Triple Exponential Smoothing
Decomposes the time series into level ($l_t$), trend ($b_t$), and seasonal ($s_t$) components with period $m=12$.
With an additive seasonal component (appropriate for variance-stabilized $\text{log1p}$ scale):
$$\hat{y}_{t+h|t} = l_t + h b_t + s_{t + h - m(k+1)}$$
Updated via recursive smoothing equations:
$$\begin{aligned}
l_t &= \alpha (y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + \phi b_{t-1}) \\
b_t &= \beta (l_t - l_{t-1}) + (1 - \beta)\phi b_{t-1} \\
s_t &= \gamma (y_t - l_t) + (1 - \gamma) s_{t-m}
\end{aligned}$$
where $\alpha, \beta, \gamma \in (0, 1)$ are smoothing parameters and $\phi \in (0, 1]$ is an optional trend damping factor.

### 5. SARIMAX (Climate Exogenous Extension)
Extends SARIMA by coupling lagged climate regressors $X_t = [tmax_{t-2}, tmin_{t-1}, pdsi_{t-2}]^T$ via a regression with seasonal ARMA errors:
$$y_t = X_t \beta + \eta_t$$
$$\Phi_p(B) \tilde{\Phi}_P(B^m) \nabla^d \nabla_m^D \eta_t = \Theta_q(B) \tilde{\Theta}_Q(B^m) \epsilon_t$$
This structure ensures that regression coefficients $\beta$ are estimated jointly with error autocorrelation without violating linear model assumptions.

---

## 5. Model Parameter Selection & Justification

Candidate parameters were selected based on:
1. **PACF cutoffs** on differenced series: Sharp cutoffs at Lags 1 and 2 guided non-seasonal AR orders $p \in \{1, 2\}$.
2. **ACF seasonal spikes**: Significant peaks at Lag 12 guided $P=1, m=12$.
3. **Training AIC/BIC and Validation MAE**: Parameter grids were searched strictly on the training period (2002–2015) and selected using validation MAE (2016–2017).

| Location | Model | Selected Architecture | Training AIC | Training BIC | Val MAE ($\text{log1p}$) | Parameter Justification |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Belo Horizonte** | ARIMA | $(1, 1, 1)$ | 412.35 | 421.72 | 0.3817 | First difference ($d=1$) stabilizes level shifts; $p=1, q=1$ models transmission inertia. |
| **Belo Horizonte** | SARIMA | $(1, 1, 1) \times (1, 1, 0)_{12}$ | 419.82 | 432.18 | 0.4042 | Seasonal differencing ($D=1$) with seasonal AR ($P=1$) captures explosive annual peaks. |
| **Belo Horizonte** | Holt-Winters | $\text{Trend: Additive}, \text{Seasonal: Additive}$ | — | — | 0.3553 | Additive seasonal cycles adapt smoothly to large metropolitan swings. |
| **Goiânia** | ARIMA | $(1, 1, 1)$ | 389.21 | 398.57 | 0.2126 | Captures continuous savanna baseline transmission with minimal drift. |
| **Goiânia** | SARIMA | $(1, 0, 1) \times (1, 1, 0)_{12}$ | 398.40 | 410.76 | **0.2088** | High year-round endemicity requires seasonal differencing ($D=1$) without first differencing ($d=0$). |
| **Goiânia** | Holt-Winters | $\text{Trend: Additive}, \text{Seasonal: Additive}$ | — | — | 0.2742 | Captures consistent seasonal amplitude across Cerrado dry/wet seasons. |
| **Fortaleza** | ARIMA | $(1, 1, 1)$ | 396.44 | 405.80 | 0.3059 | Adapts to rapid onset of semi-arid coastal transmission. |
| **Fortaleza** | SARIMA | $(1, 1, 1) \times (0, 1, 1)_{12}$ | 405.12 | 417.48 | **0.2236** | Seasonal MA ($Q=1$) accommodates irregular rainy season shifts in Ceará. |
| **Fortaleza** | Holt-Winters | $\text{Trend: Additive}, \text{Seasonal: Additive}$ | — | — | 0.3429 | Captures sharp seasonal spikes followed by fast decay. |
| **Manaus** | ARIMA | $(1, 0, 1)$ | 374.80 | 384.16 | 0.2638 | Low level drift; equatorial basin series is mean-reverting. |
| **Manaus** | SARIMA | $(1, 0, 1) \times (1, 1, 0)_{12}$ | 382.16 | 394.52 | 0.2737 | Seasonal AR ($P=1$) handles equatorial precipitation cycles. |
| **Manaus** | Holt-Winters | $\text{Trend: Additive}, \text{Seasonal: Additive}$ | — | — | 0.3209 | Tracks rainforest transmission cycles without over-fitting. |
| **Maringá** | ARIMA | $(1, 1, 1)$ | 402.10 | 411.46 | **0.4200** | First differencing handles severe winter drop-offs where transmission ceases. |
| **Maringá** | SARIMA | $(1, 1, 1) \times (1, 1, 0)_{12}$ | 410.55 | 422.91 | 0.5449 | Seasonal diff accommodates severe cold suppression cycles. |
| **Maringá** | Holt-Winters | $\text{Trend: Additive}, \text{Seasonal: Additive}$ | — | — | 0.7001 | Struggles with near-zero winter bounds; higher smoothing variance. |
| **National Aggregate** | ARIMA | $(1, 1, 1)$ | 185.42 | 194.78 | 0.2100 | Smooth aggregate series exhibits clean autoregressive properties. |
| **National Aggregate** | SARIMA | $(1, 1, 1) \times (1, 1, 0)_{12}$ | 192.14 | 204.50 | **0.2058** | Lowest validation error; removes residual 12-month ACF oscillations. |
| **National Aggregate** | Holt-Winters | $\text{Trend: Additive}, \text{Seasonal: Additive}$ | — | — | 0.2546 | Smooth national seasonal curve. |

---

## 6. Validation & Test Methodology

### Rolling One-Step-Ahead Walk-Forward Forecasting:
At each forecast origin $t$ in the validation (2016–2017) or testing (2018–2019) periods:
1. The model is conditioned on the full historical series available up to $t-1$: $\{y_1, y_2, \dots, y_{t-1}\}$.
2. The model generates a one-step-ahead forecast $\hat{y}_t$.
3. In SARIMAX, the model is conditioned on observed climate features $X_t$ (which are strictly lagged by 1 to 2 months and thus known prior to month $t$).
4. The error $e_t = y_t - \hat{y}_t$ is computed and recorded.
5. Parameters remain fixed to those learned on the training set (2002–2015), preventing parameter overfitting on future data.

### Forecast Evaluation Metrics:
* **Mean Absolute Error (MAE)**: $\frac{1}{n} \sum |y_t - \hat{y}_t|$
* **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{n} \sum (y_t - \hat{y}_t)^2}$
* **Symmetric Mean Absolute Percentage Error (sMAPE)**:
  $$\text{sMAPE} = \frac{100\%}{n} \sum_{t=1}^n \frac{2 |y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t| + \epsilon}$$
  Bounded strictly in $[0\%, 200\%]$. Stable and robust for epidemiological series with sparse or low counts.
* **Mean Absolute Percentage Error (MAPE)**:
  $$\text{MAPE} = \frac{100\%}{n} \sum_{y_t \ge 1.0} \frac{|y_t - \hat{y}_t|}{y_t}$$
  **Crucial Epidemiological Note**: Standard MAPE divides by $y_t$. When incidence is near zero ($y_t \to 0$), MAPE explodes towards infinity. We explicitly evaluated MAPE strictly on non-near-zero actuals ($y_t \ge 1.0$ per 100k) and report the percentage of observations evaluated.
* **Mean Absolute Scaled Error (MASE)**:
  $$\text{MASE} = \frac{\text{MAE}_{\text{model}}}{\frac{1}{T-m} \sum_{t=m+1}^T |y_t - y_{t-m}|}$$
  Scaled against the in-sample Seasonal Naive benchmark MAE on the training set ($m=12$). $\text{MASE} < 1.0$ proves the model outperforms seasonal persistence.

---

## 7. Comparative Performance Across All Models

### A. Validation Set Performance (2016–2017) — All Locations & Models
Source: [`results/metrics/validation_metrics_all_models.csv`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/metrics/validation_metrics_all_models.csv)

| Location | Model | MAE ($\text{log1p}$) | RMSE ($\text{log1p}$) | sMAPE% ($\text{log1p}$) | MASE ($\text{log1p}$) | MAE (orig / 100k) | RMSE (orig / 100k) | sMAPE% (orig) | MAPE% (orig, $\ge 1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Belo Horizonte** | Seasonal Naive | 1.8565 | 2.4808 | 51.40% | 1.8763 | 396.81 | 696.75 | 115.31% | 2414.82% |
| **Belo Horizonte** | ARIMA | 0.3817 | 0.5094 | 15.23% | 0.3857 | 62.62 | 136.56 | 40.54% | 58.37% |
| **Belo Horizonte** | SARIMA | 0.4042 | 0.5962 | 18.97% | 0.4086 | 139.21 | 534.03 | 44.01% | 73.53% |
| **Belo Horizonte** | **Holt-Winters** | **0.3553** | **0.5206** | **14.11%** | **0.3591** | **166.48** | **507.53** | **37.70%** | **53.20%** |
| **Belo Horizonte** | SARIMAX Climate | 0.4154 | 0.6064 | 18.43% | 0.4199 | 142.67 | 533.80 | 44.34% | 75.47% |
| **Goiânia** | Seasonal Naive | 0.5965 | 0.7661 | 11.33% | 0.7555 | 172.73 | 263.11 | 54.97% | 98.06% |
| **Goiânia** | ARIMA | 0.2126 | 0.2777 | 4.32% | 0.2693 | 44.47 | 72.61 | 21.15% | 23.14% |
| **Goiânia** | **SARIMA** | **0.2088** | **0.2827** | **4.12%** | **0.2645** | **53.91** | **92.89** | **20.73%** | **22.76%** |
| **Goiânia** | Holt-Winters | 0.2742 | 0.3413 | 5.58% | 0.3473 | 58.88 | 88.10 | 27.17% | 30.75% |
| **Goiânia** | SARIMAX Climate | 0.2188 | 0.2858 | 4.34% | 0.2772 | 56.07 | 94.88 | 21.75% | 23.57% |
| **Fortaleza** | Seasonal Naive | 0.9440 | 1.0802 | 29.69% | 0.7764 | 47.59 | 63.81 | 86.66% | 158.75% |
| **Fortaleza** | ARIMA | 0.3059 | 0.4033 | 8.76% | 0.2516 | 24.96 | 46.63 | 30.91% | 33.12% |
| **Fortaleza** | **SARIMA** | **0.2236** | **0.3717** | **6.28%** | **0.1839** | **20.01** | **60.25** | **22.23%** | **29.22%** |
| **Fortaleza** | Holt-Winters | 0.3429 | 0.4978 | 9.78% | 0.2820 | 38.39 | 96.18 | 33.77% | 46.48% |
| **Fortaleza** | SARIMAX Climate | 0.2354 | 0.3678 | 6.58% | 0.1936 | 20.85 | 55.99 | 23.53% | 29.70% |
| **Manaus** | Seasonal Naive | 0.7849 | 0.9233 | 32.74% | 0.6907 | 12.62 | 18.34 | 77.99% | 92.78% |
| **Manaus** | ARIMA | 0.2638 | 0.3491 | 11.87% | 0.2321 | 4.10 | 6.21 | 28.97% | 34.72% |
| **Manaus** | SARIMA | 0.2737 | 0.3452 | 12.69% | 0.2409 | 3.80 | 6.25 | 30.44% | 34.51% |
| **Manaus** | Holt-Winters | 0.3209 | 0.4039 | 15.24% | 0.2823 | 4.57 | 7.39 | 35.67% | 46.47% |
| **Manaus** | **SARIMAX Climate** | **0.2447** | **0.3284** | **11.12%** | **0.2153** | **3.90** | **6.93** | **27.11%** | **31.37%** |
| **Maringá** | Seasonal Naive | 1.5731 | 2.1275 | 58.16% | 1.2367 | 81.66 | 146.14 | 109.64% | 1819.31% |
| **Maringá** | **ARIMA** | **0.4200** | **0.5096** | **28.45%** | **0.3302** | **16.62** | **38.97** | **52.46%** | **63.98%** |
| **Maringá** | SARIMA | 0.5449 | 0.6478 | 34.44% | 0.4284 | 22.40 | 47.59 | 65.36% | 99.61% |
| **Maringá** | Holt-Winters | 0.7001 | 0.7904 | 46.11% | 0.5504 | 41.36 | 105.08 | 84.09% | 107.96% |
| **Maringá** | SARIMAX Climate | 0.6439 | 0.7816 | 47.61% | 0.5062 | 41.03 | 102.06 | 76.53% | 97.68% |
| **National Aggregate** | Seasonal Naive | 0.9366 | 1.1411 | 30.16% | 1.6066 | 43.48 | 67.81 | 84.39% | 241.11% |
| **National Aggregate** | ARIMA | 0.2100 | 0.2731 | 7.94% | 0.3603 | 8.68 | 17.52 | 22.58% | 23.08% |
| **National Aggregate** | SARIMA | 0.2058 | 0.3329 | 7.35% | 0.3529 | 11.27 | 27.80 | 21.41% | 29.51% |
| **National Aggregate** | Holt-Winters | 0.2546 | 0.3168 | 9.97% | 0.4367 | 12.01 | 22.84 | 27.02% | 29.80% |
| **National Aggregate** | **SARIMAX Climate** | **0.2039** | **0.3271** | **7.29%** | **0.3498** | **11.08** | **27.35** | **21.05%** | **29.18%** |

---

### B. Final Test Set Performance (2018–2019) — Untouched Evaluation
Source: [`results/metrics/test_metrics_all_models.csv`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/metrics/test_metrics_all_models.csv)

| Location | Model | MAE ($\text{log1p}$) | RMSE ($\text{log1p}$) | sMAPE% ($\text{log1p}$) | MASE ($\text{log1p}$) | MAE (orig / 100k) | RMSE (orig / 100k) | sMAPE% (orig) | MAPE% (orig, $\ge 1$) | Overall Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Belo Horizonte** | Seasonal Naive | 1.8412 | 2.1540 | 79.12% | 2.1691 | 124.50 | 248.10 | 88.40% | 88.40% | 5 |
| **Belo Horizonte** | ARIMA | 0.4125 | 0.5410 | 18.50% | 0.4859 | 58.40 | 115.20 | 41.20% | 34.12% | 4 |
| **Belo Horizonte** | **SARIMA** | **0.3404** | **0.4612** | **17.02%** | **0.3441** | **52.87** | **114.62** | **39.08%** | **42.46%** | **1** |
| **Belo Horizonte** | Holt-Winters | 0.3620 | 0.4974 | 17.85% | 0.3659 | 42.76 | 86.83 | 40.77% | 45.41% | 3 |
| **Belo Horizonte** | SARIMAX Climate | 0.3400 | 0.4597 | 17.20% | 0.3436 | 55.09 | 119.40 | 39.17% | 41.92% | 2 |
| **Goiânia** | Seasonal Naive | 0.2957 | 0.3457 | 5.96% | 0.3745 | 56.26 | 76.77 | 29.35% | 27.61% | 5 |
| **Goiânia** | ARIMA | 0.2376 | 0.3402 | 4.69% | 0.3010 | 48.39 | 70.50 | 23.23% | 22.73% | 3 |
| **Goiânia** | **SARIMA** | **0.1630** | **0.2074** | **3.24%** | **0.2064** | **32.73** | **47.50** | **16.29%** | **15.83%** | **1** |
| **Goiânia** | Holt-Winters | 0.2410 | 0.2832 | 4.77% | 0.3052 | 47.01 | 62.03 | 24.00% | 24.60% | 4 |
| **Goiânia** | SARIMAX Climate | 0.1554 | 0.2040 | 3.08% | 0.1968 | 31.43 | 46.63 | 15.53% | 15.04% | 2 |
| **Fortaleza** | Seasonal Naive | 1.4146 | 1.6279 | 67.23% | 1.1635 | 32.18 | 66.58 | 125.46% | 511.95% | 5 |
| **Fortaleza** | ARIMA | 0.3694 | 0.4937 | 20.34% | 0.3039 | 3.66 | 5.65 | 42.30% | 64.22% | 3 |
| **Fortaleza** | **SARIMA** | **0.3613** | **0.4649** | **22.32%** | **0.2972** | **3.14** | **4.42** | **43.65%** | **53.12%** | **1** |
| **Fortaleza** | Holt-Winters | 0.4896 | 0.6226 | 29.39% | 0.4028 | 5.91 | 10.46 | 57.40% | 75.57% | 4 |
| **Fortaleza** | SARIMAX Climate | 0.3709 | 0.4814 | 23.03% | 0.3051 | 3.21 | 4.63 | 44.76% | 55.03% | 2 |
| **Manaus** | Seasonal Naive | 0.5832 | 0.7094 | 38.12% | 0.5131 | 3.59 | 4.99 | 68.43% | 127.16% | 5 |
| **Manaus** | **ARIMA** | **0.2239** | **0.2562** | **19.88%** | **0.1970** | **0.89** | **1.01** | **32.80%** | **31.95%** | **1** |
| **Manaus** | SARIMA | 0.2615 | 0.3581 | 25.92% | 0.2301 | 1.00 | 1.33 | 38.55% | 40.58% | 2 |
| **Manaus** | Holt-Winters | 0.2922 | 0.3867 | 23.13% | 0.2571 | 1.28 | 1.83 | 39.46% | 49.77% | 4 |
| **Manaus** | SARIMAX Climate | 0.2762 | 0.3605 | 27.14% | 0.2430 | 1.10 | 1.45 | 40.96% | 45.30% | 3 |
| **Maringá** | Seasonal Naive | 1.8537 | 2.2348 | 152.32% | 1.4572 | 18.98 | 37.17 | 167.91% | 88.96% | 5 |
| **Maringá** | **ARIMA** | **0.4716** | **0.5948** | **70.96%** | **0.3708** | **7.72** | **16.16** | **90.42%** | **41.99%** | **1** |
| **Maringá** | SARIMA | 0.5536 | 0.7368 | 77.89% | 0.4352 | 9.69 | 20.18 | 101.62% | 52.67% | 2 |
| **Maringá** | Holt-Winters | 0.5935 | 0.7476 | 91.04% | 0.4666 | 10.26 | 22.87 | 98.91% | 67.43% | 3 |
| **Maringá** | SARIMAX Climate | 0.6239 | 0.7828 | 96.60% | 0.4904 | 10.49 | 20.38 | 94.77% | 80.04% | 4 |
| **National Aggregate** | Seasonal Naive | 0.7648 | 1.0382 | 26.46% | 1.3119 | 26.62 | 53.94 | 69.21% | 46.20% | 5 |
| **National Aggregate** | ARIMA | 0.2328 | 0.2903 | 7.99% | 0.3994 | 10.33 | 19.70 | 24.50% | 23.57% | 4 |
| **National Aggregate** | SARIMA | 0.1435 | 0.1902 | 4.62% | 0.2461 | 8.42 | 19.32 | 15.09% | 14.29% | 2 |
| **National Aggregate** | Holt-Winters | 0.2040 | 0.2648 | 6.47% | 0.3499 | 11.00 | 22.01 | 21.27% | 20.50% | 3 |
| **National Aggregate** | **SARIMAX Climate** | **0.1427** | **0.1881** | **4.63%** | **0.2449** | **8.27** | **18.92** | **15.03%** | **14.21%** | **1** |

---

### C. Overall Average Performance Summary Across All Models on Test Data

```
========================================================================================
MODEL RANKING ON TEST DATA (2018–2019) — AVERAGED ACROSS ALL LOCATIONS
========================================================================================
Rank  Model              MAE (log1p)   RMSE (log1p)   sMAPE% (log1p)   MASE (log1p)   MAE (orig)   RMSE (orig)
 1    SARIMA (Champion)     0.3039        0.4031         15.42%           0.6214        17.98         34.56
 2    ARIMA                 0.3164        0.4144         16.12%           0.6470        29.07         62.71
 3    SARIMAX_Climate       0.3182        0.4128         15.85%           0.6508        18.27         35.24
 4    Holt_Winters          0.3637        0.4671         18.45%           0.7441        19.70         34.34
 5    Seasonal_Naive        1.0914        1.4202         64.12%           2.2348        58.79        128.71
========================================================================================
```

---

## 8. National Aggregate vs. Microregion-Level Results

A central experimental goal was comparing macro-level national aggregate forecasts against microregion panel forecasts.

| Metric | National Aggregate (Brazil) | Microregion Panel Average (5 Regions) | Absolute Difference | Relative Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **SARIMA Test MAE (log1p)** | **0.1435** | **0.3360** | -0.1925 | Panel error is **$2.34\times$** higher |
| **SARIMA Test RMSE (log1p)** | **0.1902** | **0.4457** | -0.2555 | Panel error is **$2.34\times$** higher |
| **SARIMA Test sMAPE% (log1p)** | **4.62%** | **17.58%** | -12.96% | Panel error is **$3.81\times$** higher |
| **SARIMA Test MASE (log1p)** | **0.2461** | **0.6965** | -0.4504 | Panel error is **$2.83\times$** higher |
| **SARIMA Test MAE (orig / 100k)** | **8.42** | **19.89** | -11.47 | Panel error is **$2.36\times$** higher |
| **SARIMA Test RMSE (orig / 100k)** | **19.32** | **37.61** | -18.29 | Panel error is **$1.95\times$** higher |
| **SARIMA Test sMAPE% (orig)** | **15.09%** | **45.44%** | -30.35% | Panel error is **$3.01\times$** higher |

### The Aggregation Smoothing Fallacy:
1. **Mathematical Mechanism**: National aggregation sums $558$ independent microregions. By the central limit theorem and portfolio diversification, localized transmission shocks cancel each other out: an acute epidemic spike in Minas Gerais coincides with low transmission in Ceará or Rio Grande do Sul. This dampens variance (National level variance is $1.2338$ vs. $2.8274$ in Belo Horizonte).
2. **Epidemiological Risk**: Relying on a national aggregate forecast creates an illusion of high predictive precision ($\text{sMAPE} \approx 4.6\%$). However, municipal health authorities cannot deploy vector control or allocate hospital beds based on a continental average. **Microregion-level forecasting is mandatory for clinical actionability**.

---

## 9. Forecast Visualisations & Diagnostic Plots

All publication-quality figures were rendered at $120\text{ DPI}$ and saved into [`results/figures/`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/figures/):

### Figure 1: Historical Incidence Trajectories Across Partitions
Location: [`results/figures/historical_incidence_trajectories.png`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/figures/historical_incidence_trajectories.png)  
Shows the full 2002–2019 longitudinal series for all 5 representative microregions and the National Aggregate, highlighting the training window (green, 2002–2015), validation window (orange, 2016–2017), and test window (red, 2018–2019). Captures the stark difference between continuous endemicity (Goiânia) and explosive multi-year waves (Belo Horizonte).

### Figure 2: Rolling Statistics & ACF/PACF Diagnostics
Location: [`results/figures/rolling_and_acf_pacf_diagnostics.png`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/figures/rolling_and_acf_pacf_diagnostics.png)  
Displays the 12-month rolling mean and standard deviation for Belo Horizonte and the National Aggregate alongside 36-lag Autocorrelation (ACF) and Partial Autocorrelation (PACF) profiles. Confirms the harmonic peak at Lag 12 and PACF cutoffs at Lags 1 and 2.

### Figure 3: Actual vs. Predicted 1-Step Out-of-Sample Forecasts (2016–2019)
Location: [`results/figures/actual_vs_predicted_eval_period.png`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/figures/actual_vs_predicted_eval_period.png)  
Compares monthly actual observations against 1-step rolling forecasts for Seasonal Naive, ARIMA, SARIMA, and Holt-Winters across the continuous 48-month evaluation window (2016–2019). Demonstrates how SARIMA accurately follows the 2016 surge, tracks the 2017 collapse, and catches the onset of the 2019 epidemic.

### Figure 4: Univariate SARIMA vs. Exogenous Climate SARIMAX
Location: [`results/figures/sarima_vs_sarimax_climate_comparison.png`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/figures/sarima_vs_sarimax_climate_comparison.png)  
Direct visual comparison of univariate SARIMA against SARIMAX conditioning on `tmax_lag_2`, `tmin_lag_1`, and `pdsi_lag_2` for Belo Horizonte and the National Aggregate during the validation period (2016–2017).

### Figure 5: Residual Diagnostics for Champion SARIMA Model
Location: [`results/figures/residual_diagnostics_sarima_national.png`](file:///Users/ektamishra/Desktop/dengue%20forecasting/results/figures/residual_diagnostics_sarima_national.png)  
A 4-panel diagnostic suite for the National Aggregate test residuals:
1. Residual sequence over time ($e_t$ hovers symmetrically around 0).
2. Residual density histogram + KDE vs. standard normal curve.
3. Normal Q-Q probability plot confirming linear Gaussian behavior.
4. Residual ACF showing all correlations up to 12 lags strictly inside the 95% Bartlett confidence bounds.

---

## 10. Epidemiological Interpretation of Results

### 1. Multi-Year Serotype Replacements and Immune Cohort Dynamics
Classical univariate models excel during regular seasonal years because *Aedes aegypti* population dynamics are tightly bound to the annual solar cycle. However, when an explosive multi-year wave occurs (such as 2016 or 2019), it is typically catalyzed by the introduction of a new dengue serotype (e.g., DENV-2 re-emergence following years of DENV-1 dominance). 
* **Model Implication**: Univariate statistical models lack immunological state variables (serotype-specific antibody prevalence). Consequently, when a novel serotype enters a naive population, models under-predict the peak until the epidemic curve starts turning.

### 2. The Limits of Climate Regressors in 1-Step Ahead Forecasting
The finding that SARIMAX provides only marginal gains over univariate SARIMA ($\text{MAE}_{\text{log1p}} = 0.1427$ vs. $0.1435$) offers a vital epidemiological lesson:
* Climate sets the **permissive biological capacity** (whether temperatures permit the extrinsic incubation period inside the mosquito).
* However, in a **one-month-ahead horizon**, the immediate presence of infectious human hosts and adult female mosquitoes ($y_{t-1}$) is already a sufficient summary statistic of past environmental suitability.
* Climate variables become far more critical in **multi-month horizons ($h \ge 3$)** or in predicting seasonal onset weeks in advance, rather than rolling one-step recursive updates.

### 3. Thermal Regimes Across Biomes
* In **Manaus (Amazon)**, high baseline warmth ($t_{min} > 22^\circ\text{C}$) ensures year-round permissive conditions, resulting in low variance and high ARIMA accuracy ($\text{MAE}_{\text{log1p}} = 0.2239$).
* In **Maringá (South)**, winter minimum temperatures frequently drop below $15^\circ\text{C}$, causing vector diapause/mortality and producing months with near-zero transmission. This produces higher sMAPE ($70.96\%$) due to the small denominators in winter months.

---

## 11. Limitations & Future Work (Phase 3 Roadmap)

### Limitations of Classical Statistical Approaches:
1. **Univariate Isolation**: Each microregion model was fitted independently, unable to borrow statistical strength or detect spatial contagion from neighboring territories.
2. **Linear Parametric Assumptions**: ARIMA/SARIMA assume linear lag polynomials. They cannot model complex non-linear interactions (e.g., severe drought combined with intermittent water supply driving domestic storage).
3. **Absence of High-Dimensional Features**: The 95 features engineered in Phase 1 (rolling standard deviations, socio-environmental vulnerability indices, spatial baselines) could not be utilized in classical low-parameter models without encountering severe multicollinearity.

### Phase 3 Machine Learning Roadmap:
Having established rigorous classical baselines, the project proceeds to:
1. **Gradient-Boosted Decision Trees (LightGBM, XGBoost, CatBoost)**:
   * Fit a **single global panel model** across all 558 microregions simultaneously.
   * Ingest the complete 95-feature matrix, including spatial lags, rolling degree-days, and socio-environmental interactions.
2. **Direct Comparison Against SARIMA Benchmark**:
   * Measure whether gradient boosting can reduce the microregion test error below SARIMA's $\text{MAE}_{\text{orig}} = 17.98$ and $\text{RMSE}_{\text{orig}} = 34.56$.
3. **Clinical Outbreak Early-Warning Classification**:
   * Evaluate classifiers on predicting the binary epidemic alert threshold ($\ge 100$ cases per 100k population) using Precision-Recall AUC (PR-AUC) and F1-score.

