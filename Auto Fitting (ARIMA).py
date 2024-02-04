from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

df = pd.read_csv('C:\Users\hocke\.vscode\FedRate CSV\Feb 2014- Feb 2024 Treasury Data (Monthly).csv', names=['value'], header=0)

model = pm.auto_arima(df.value, start_p=1, start_q=1,
                      test='adf',       
                      max_p=3, max_q=3, 
                      m=1,              
                      d=None,           
                      seasonal=False,   
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

#Hopefully metrics are accurate and fit before this code runs as a forecast

n_periods = #CHOOSE A PERIOD
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.value), len(df.value)+n_periods)

fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

plt.plot(df.value)
plt.plot(fc_series, color='green')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Interest Rates ARIMA")
plt.show()
