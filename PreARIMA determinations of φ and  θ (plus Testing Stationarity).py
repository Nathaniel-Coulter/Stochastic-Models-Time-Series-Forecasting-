#Order Differencing 
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
#ADF: -2.72640
#φ: 0.1442193

#Graphs
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

df = pd.read_csv('C:\Users\hocke\.vscode\FedRate CSV\Feb 2014- Feb 2024 Treasury Data (Monthly).csv', names=['value'], header=0)

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])
plt.show()


#Testing and Determining Stationarity
from pmdarima.arima.utils import ndiffs
df = pd.read_csv('C:\Users\hocke\.vscode\FedRate CSV\Feb 2014- Feb 2024 Treasury Data (Monthly).csv', names=['value'], header=0)
y = df.value

ndiffs(y, test='adf') 
ndiffs(y, test='kpss')  
ndiffs(y, test='pp')  
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])
plt.show()

#Autocorrelation / Determining Order 
##The following is a plot of the first series only
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])
plt.show()

#Determining Moving Average Order (I have merged a few files onto one py file, keeping the neccesary package imports to be thorough & incase I come back and copy/paste for real world use)
#import pandas as pd
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
df = pd.read_csv('C:\Users\hocke\.vscode\FedRate CSV\Feb 2014- Feb 2024 Treasury Data (Monthly).csv')

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df.value.diff().dropna(), ax=axes[1])
plt.show()

