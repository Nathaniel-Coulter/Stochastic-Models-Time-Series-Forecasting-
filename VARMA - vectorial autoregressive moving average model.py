import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import var_order_select, adfuller
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.diagnostic.diagnostic_tests import durbin_watson

# Load data (assuming X and Y are your time series)
# X = pd.read_csv('X.csv', index_col='Date', parse_dates=True)['Value']
# Y = pd.read_csv('Y.csv', index_col='Date', parse_dates=True)['Value']

# Check for stationarity using the KPSS test
def kpss_test(timeseries):
    kpss_stat, p_value, _, _, _ = adfuller(timeseries, regression='ct')
    return p_value > 0.05

if not kpss_test(X):
    X = X.diff().dropna()
if not kpss_test(Y):
    Y = Y.diff().dropna()

# Lag order selection for VAR(p)
p = var_order_select(X, Y, ic='aic')['n_lags']

# Durbin-Watson test for residuals
def durbin_watson_test(timeseries):
    resid = np.array(timeseries)
    dw_stat, _, _, _, _ = durbin_watson(resid)
    return dw_stat

model = VARMAX(endog=[X, Y], order=(p, 0, p))
results = model.fit()

# Check for residual autocorrelation
residual_autocorrelation = durbin_watson_test(results.resid)

if residual_autocorrelation < 1.5 or residual_autocorrelation > 2.5:
    # If residual autocorrelation is present, increase q to improve the model
    q = 1
    model = VARMAX(endog=[X, Y], order=(p, 0, q))
    results = model.fit()

# Model diagnostic checks (e.g., residuals, model fit)

# Forecasting and analysis
forecast = results.forecast(steps=12)
print("Forecast:", forecast)