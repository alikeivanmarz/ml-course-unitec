# Time Series and Forecasting

Time series data — observations indexed by time — requires modelling and evaluation patterns that differ from independent-sample data. Random splits leak future information into training; standard error estimates assume independence that does not hold; and seasonality, trend, and autocorrelation must be modelled explicitly. This guide covers the structure of time series data, the classical statistical methods (ARIMA family, exponential smoothing), the machine-learning approaches built on engineered lag features, the deep-learning architectures applicable to sequential data, and the evaluation protocols that produce honest forecast accuracy estimates.

**Table of Contents**

1. [What Makes Time Series Different](#1-what-makes-time-series-different)
2. [Working with Datetime Data](#2-working-with-datetime-data)
3. [Stationarity and Differencing](#3-stationarity-and-differencing)
4. [Decomposition](#4-decomposition)
5. [ARIMA and SARIMA](#5-arima-and-sarima)
6. [Exponential Smoothing and Prophet](#6-exponential-smoothing-and-prophet)
7. [Machine Learning with Lag Features](#7-machine-learning-with-lag-features)
8. [Deep Learning for Time Series](#8-deep-learning-for-time-series)
9. [Forecast Evaluation](#9-forecast-evaluation)
10. [Common Pitfalls](#10-common-pitfalls)
11. [Resources](#11-resources)

---

## 1. What Makes Time Series Different

| Property | Implication |
|----------|-------------|
| Temporal ordering | Observations are not exchangeable; train must precede test |
| Autocorrelation | Adjacent observations are correlated; standard errors that assume independence are wrong |
| Trend | Long-term direction must be modelled or removed |
| Seasonality | Periodic patterns must be modelled or removed |
| Non-stationarity | Distribution properties (mean, variance) change over time |
| Irregular sampling | Real-world series often have gaps or uneven intervals |

These properties dictate three departures from standard ML practice:

1. **No random splits.** Validation and test sets must be temporally later than training.
2. **No standard cross-validation.** K-fold CV leaks future into past. Use rolling-origin or expanding-window splits.
3. **No standard preprocessing pipelines that fit on the full dataset.** Statistics must be computed on training data only and applied to subsequent data.

---

## 2. Working with Datetime Data

### 2.1 Pandas Datetime Operations

```python
import pandas as pd

# Parse on load
df = pd.read_csv("series.csv", parse_dates=["timestamp"], index_col="timestamp")

# Or convert after
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

# Datetime accessors
df.index.year
df.index.month
df.index.day_of_week
df.index.hour
df.index.is_month_end
```

Sorting by index is essential before any time-based operation. Many bugs in time series code originate from unordered data.

### 2.2 Resampling

Resampling converts between sampling frequencies — aggregating high-frequency data to lower frequency, or interpolating low-frequency data to higher frequency.

```python
df.resample("D").mean()       # daily means from sub-daily data
df.resample("W").sum()        # weekly totals
df.resample("M").last()       # last value of each month
df.resample("H").asfreq()     # hourly grid; introduces NaN where no observation exists
```

Common frequency aliases: `H` hourly, `D` daily, `W` weekly, `M` month-end, `MS` month-start, `Q` quarter-end, `Y` year-end.

### 2.3 Filling Gaps

```python
df = df.resample("D").asfreq()           # complete daily index; NaN in gaps
df = df.ffill()                          # forward-fill
df = df.interpolate(method="time")       # time-aware interpolation
```

The choice of fill method depends on the series' nature: forward-fill is appropriate for status-like variables; interpolation is appropriate for continuous quantities; explicit zero-fill is appropriate for count data where missingness signifies absence.

---

## 3. Stationarity and Differencing

A *stationary* series has constant statistical properties (mean, variance, autocorrelation structure) over time. Most classical time series models assume stationarity. Real-world series rarely are; transformations make them so.

### 3.1 Testing Stationarity

The Augmented Dickey–Fuller test checks the null hypothesis that a series has a unit root (is non-stationary):

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series.dropna())
adf_stat, p_value = result[0], result[1]
# p_value < 0.05 -> reject null -> series is stationary
```

The KPSS test is the complement: null hypothesis is stationarity. Using both gives more robust conclusions than either alone.

### 3.2 Differencing

First differencing removes trend by subtracting each observation from the previous:

```python
series_diff = series.diff().dropna()
```

Seasonal differencing removes seasonal patterns by differencing at the period:

```python
series_seasonal_diff = series.diff(periods=12)   # for monthly data with annual seasonality
```

Repeated differencing is rarely required; one or two passes typically suffices.

### 3.3 Other Transformations

| Transformation | Purpose |
|----------------|---------|
| Log | Stabilize variance for series with multiplicative noise |
| Box–Cox | Generalised power transform; choose λ to maximize normality |
| Square root | Reduce variance for count data |
| Standardisation | Zero mean, unit variance — useful for ML approaches |

Variance-stabilising transformations are applied before differencing.

---

## 4. Decomposition

Decomposition separates a series into trend, seasonal, and residual components. It is descriptive rather than predictive — useful for understanding structure before model selection.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(series, model="additive", period=12)
result.trend
result.seasonal
result.resid
```

| Model | Form | When to use |
|-------|------|-------------|
| Additive | y = trend + seasonal + residual | Constant seasonal amplitude |
| Multiplicative | y = trend × seasonal × residual | Seasonal amplitude grows with the level |

For non-constant seasonal patterns, STL decomposition (`from statsmodels.tsa.seasonal import STL`) is more flexible than `seasonal_decompose`.

---

## 5. ARIMA and SARIMA

ARIMA (AutoRegressive Integrated Moving Average) is the foundational classical model for univariate forecasting. The acronym describes three components:

| Letter | Stands for | Parameter |
|--------|------------|-----------|
| AR | AutoRegressive: dependence on past values | `p` |
| I | Integrated: number of differencing operations applied | `d` |
| MA | Moving Average: dependence on past forecast errors | `q` |

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(series, order=(1, 1, 1))
fitted = model.fit()
forecast = fitted.forecast(steps=12)
```

### 5.1 SARIMA

SARIMA extends ARIMA with seasonal AR, differencing, and MA components, plus a seasonal period:

```python
model = ARIMA(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fitted = model.fit()
```

`seasonal_order=(P, D, Q, s)` mirrors `(p, d, q)` at the seasonal lag `s`.

### 5.2 Order Selection

Manual order selection uses ACF (autocorrelation function) and PACF (partial autocorrelation function) plots. Automated selection (`pmdarima.auto_arima`) tries combinations and selects by AIC:

```python
# pip install pmdarima
import pmdarima as pm

model = pm.auto_arima(series, seasonal=True, m=12, suppress_warnings=True)
```

`auto_arima` can over-fit; results should be sense-checked against domain knowledge of seasonality and trend.

---

## 6. Exponential Smoothing and Prophet

### 6.1 Exponential Smoothing

Exponential smoothing weights recent observations more heavily than distant ones. The Holt–Winters method extends simple smoothing with trend and seasonality.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    series, trend="add", seasonal="add", seasonal_periods=12,
)
fitted = model.fit()
forecast = fitted.forecast(steps=12)
```

Exponential smoothing is competitive with ARIMA on many real-world series and easier to specify. It is widely used as a baseline.

### 6.2 Prophet

Prophet, developed by Meta, is designed for business time series with strong seasonality, holiday effects, and missing observations.

```python
# pip install prophet
from prophet import Prophet

df = pd.DataFrame({"ds": series.index, "y": series.values})
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

Prophet handles missing data, multiple seasonalities, and holiday effects out of the box. It is robust on series with sparse history and irregular sampling. Trade-off: less customisable than ARIMA for the cases where careful order selection matters.

---

## 7. Machine Learning with Lag Features

Standard regression and gradient-boosting models can be applied to time series after engineering lag features.

### 7.1 Lag Features

```python
def make_lag_features(series, lags):
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df.dropna()

features = make_lag_features(series, lags=[1, 2, 3, 7, 14, 30])
```

Common lag patterns:

| Lag set | Captures |
|---------|----------|
| 1, 2, 3 | Short-term momentum |
| 7, 14, 21 | Weekly patterns (daily data) |
| 12, 24, 36 | Annual patterns (monthly data) |
| Rolling means over 7, 14, 30 days | Smoothed local averages |
| Rolling std, min, max | Local variability |

### 7.2 Calendar Features

```python
df["month"] = df.index.month
df["day_of_week"] = df.index.day_of_week
df["is_weekend"] = df.index.day_of_week.isin([5, 6])
df["hour"] = df.index.hour
df["day_of_year"] = df.index.day_of_year
```

Cyclical features (month, hour) are sometimes encoded as sine/cosine pairs to give the model the cyclic structure:

```python
import numpy as np

df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
```

### 7.3 Modelling

Once features are engineered, any tabular model applies:

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=0)
model.fit(features.drop("y", axis=1), features["y"])
```

Gradient boosting (XGBoost, LightGBM, CatBoost) is a strong default for tabular time series, often competitive with or better than classical methods on real datasets.

---

## 8. Deep Learning for Time Series

Deep learning is competitive with classical and ML methods primarily on long, complex, multivariate series. For short, single-series forecasting, classical methods often win.

| Architecture | Strength |
|--------------|----------|
| LSTM, GRU | Sequential modelling with learned long-range dependencies |
| 1D CNN (Temporal Convolutional Network) | Captures local patterns; parallel-friendly |
| Transformer-based (Informer, Autoformer, PatchTST) | Long-range dependencies via attention; strong on multivariate |
| N-BEATS, NHITS | Specifically designed for forecasting; strong benchmarks |

### 8.1 LSTM Example Sketch

```python
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])     # use last timestep's hidden state
```

Inputs are typically windowed: a fixed-length history is fed in to predict one or more future steps.

### 8.2 Foundation Models

Recent work (Chronos, TimeGPT, Lag-Llama) provides pre-trained time series foundation models for zero-shot or few-shot forecasting. These are starting to compete with or exceed task-specific models on standard benchmarks.

---

## 9. Forecast Evaluation

Time series evaluation requires temporally ordered splits. Three patterns:

### 9.1 Single Hold-Out

The simplest split: train on the earliest portion, test on the latest.

```python
split_point = int(len(series) * 0.8)
train, test = series[:split_point], series[split_point:]
```

### 9.2 Expanding-Window Cross-Validation

Repeatedly train on an initial window, predict the next step(s), and expand the window forward.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(series):
    train, test = series.iloc[train_idx], series.iloc[test_idx]
    ...
```

### 9.3 Rolling-Origin Cross-Validation

Same as expanding window, but the training set is a fixed-size sliding window. Appropriate when the data-generating process changes and only recent history is informative.

### 9.4 Forecast Metrics

| Metric | Formula sketch | Notes |
|--------|----------------|-------|
| MAE | mean of \|y − ŷ\| | Robust to outliers; same units as the series |
| RMSE | sqrt(mean of (y − ŷ)²) | Penalizes large errors more |
| MAPE | mean of \|y − ŷ\| / \|y\| | Percentage; undefined when y is zero |
| sMAPE | mean of 2 \|y − ŷ\| / (\|y\| + \|ŷ\|) | Symmetric variant; bounded |
| MASE | MAE divided by MAE of naive seasonal forecast | Scale-free; allows comparison across series |

MAPE and sMAPE are common but problematic when actuals can be zero or near-zero. MASE is preferred for comparing accuracy across series of different scales.

---

## 10. Common Pitfalls

| Pitfall | Mechanism | Fix |
|---------|-----------|-----|
| Random train/test split | Future leaks into past | Use temporal split |
| Standard cross-validation | Same | Use `TimeSeriesSplit` |
| Fitting scaler on full dataset | Test-set statistics leak | Fit on training only; transform test |
| Using future-derived features | E.g., lag of leading indicator known only later | Audit each feature's availability time |
| Ignoring autocorrelation in errors | Standard errors are too small | Use HAC standard errors; check residuals |
| Forecasting differenced series without inverting | Forecasts are on the wrong scale | Apply inverse transform before evaluation |
| Comparing models on a single hold-out | Comparison is high-variance | Use rolling-origin evaluation across multiple folds |
| Naive baseline omitted | No reference for whether the model adds value | Always include a naive (last-value or seasonal-naive) baseline |
| Reporting in-sample fit instead of out-of-sample forecast accuracy | Inflated performance | Report only forecast metrics from a held-out period |

---

## 11. Resources

- [Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., 2021)](https://otexts.com/fpp3/) — open-access textbook; the standard reference.
- [`statsmodels.tsa` documentation](https://www.statsmodels.org/stable/tsa.html) — ARIMA, SARIMA, exponential smoothing, decomposition.
- [Prophet documentation](https://facebook.github.io/prophet/) — official guide and recipes.
- [`pmdarima`](https://alkaline-ml.com/pmdarima/) — automatic ARIMA order selection.
- [Makridakis et al., *The M5 Accuracy Competition Results* (2022)](https://www.sciencedirect.com/science/article/pii/S0169207021001874) — large-scale benchmark of forecasting methods.
- [Lim and Zohren, *Time-Series Forecasting with Deep Learning: A Survey* (2021)](https://arxiv.org/abs/2004.13408) — survey of deep architectures for forecasting.
- [Nixtla — `statsforecast`, `neuralforecast`, `mlforecast`](https://github.com/Nixtla) — modern Python ecosystem covering classical, ML, and DL approaches with consistent APIs.

---

[← Previous: Unsupervised Learning](18_UNSUPERVISED_LEARNING_GUIDE.md) | [Index](README.md) | [Next: Reinforcement Learning →](20_REINFORCEMENT_LEARNING_GUIDE.md)
