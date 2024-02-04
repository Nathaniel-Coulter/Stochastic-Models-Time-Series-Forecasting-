#Now in order to validate our model we need to ensure it's accuracy using a couple factors.
# 1.) Correlation in Actual vs. Forecasr
# 2.) Mean Absolute Percentage Error (MAPE)
# 3.) Mean error, Mean Absolute Error, Lag Correlation and Min-Max Error


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)
# Our Accuracy Metrics
#> mape': 0.02250131357314834,
#> me': 3.230783108990054,
#> mae': 4.548322194530069,
#> mpe': 0.016421001932706705,
#> mse': 6.373238534601827,
#> acf1': 0.5105506325288692,
#> corr': 0.9674576513924394,
#> minmax': 0.02163154777672227}

