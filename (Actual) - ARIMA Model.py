from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df.value, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

#Residual Error testing 
#residuals = pd.DataFrame(model_fit.resid)
#fig, ax = plt.subplots(1,2)
#residuals.plot(title="Residuals", ax=ax[0])
#residuals.plot(kind='kde', title='Density', ax=ax[1])
#plt.show()

#Assuming variance is uniform, plot versus fitted values
#model_fit.plot_predict(dynamic=False)
#plt.show()

#We use dynamic-False to sample lagged values for prediction

#Training Dataset / ML Stuff
from statsmodels.tsa.stattools import acf
train = df.value[:85]
test = df.value[85:]
model = ARIMA(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1)  
fc, se, conf = fitted.forecast(15, alpha=0.05)
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
#Graph
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# ACTUAL MODEL TIME !
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

fc, se, conf = fitted.forecast(15, alpha=0.05)
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k',
                 alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

