# I will try and include a separate text file to explaination, as this is the simplest example of an ARMA model I intend on having on github. 
#Any other ARMA models will refer to this post as a guide of what packages you should become familiar with.
#forData 
import yfinance as yf
#for manipulation
import pandas as pd
import numpy as np
#for time series 
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
#for visualisation
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
#for warnings
import warnings
warnings.filterwarnings('ignore')
#our seed
np.random.seed(2022)
#normal distribution error for our time series
errors = np.random.normal(0, 0.1, 1000)
#parameters for our AR & MA arrays, respectively 
arparams = np.array([.1, .25, .5, .75, .9, .99])
maparams = np.array([-.1, -.25, -.5, -.75, -.9, -.99])
#base dataframe
df = pd.DataFrame(data = errors, columns=['errors'])
#Our goal will now be to simulate the following... 
#AR(1) models with a range of φ coefficient values equal to (.1)-->(.99)
#MA(1) models with a range of θ coefficient values equal to (-.1)-->(-.99)
#hopefully ending up with a (1,1) ARMA model that has equal values for φ and θ of (0.3)
for i in range(1000):
    if i == 0:
        for ar in arparams:

         df[f'ARMA_1_0_0{str(ar)[2:]}_0'] = df['errors'].iloc[0]
         for ma in maparams:
            df[f'ARMA_0_1_0_0{str(ma)[3:]}'] = df['errors'].iloc[0]
            df[f'ARMA_1_1_03_03'] = df['errors'].iloc[0]

         for ar in arparams:
            df[f'ARMA_1_0_0{str(ar)[2:]}_0'].iloc[i] = ar * \
            df[f'ARMA_1_0_0{str(ar)[2:]}_0'].iloc[i-1] + df['errors'].iloc[i]

         for ma in maparams:
            df[f'ARMA_0_1_0_0{str(ma)[3:]}'].iloc[i] = df['errors'].iloc[i] + \
            ma*df['errors'].iloc[i-1]
            df[f'ARMA_1_1_03_03'] = 0.3*df[f'ARMA_1_1_03_03'].iloc[i-1] + \
            df['errors'].iloc[i] + 0.3*df['errors'].iloc[i-1]
        
         fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(
         15, 7), sharex=True, sharey=True)
         params = [1, 25, 5, 75, 9, 99]

         for param in params:
            if params.index(param) < 3:
               axs[0, params.index(param)].plot(df.index, df[f'ARMA_1_0_0{param}_0'])
               axs[0, params.index(param)].tick_params(labelsize=15)
               axs[0, params.index(param)].set_title(
                  f'ARMA(1,0) time series with phi=0.{param}', fontdict={'fontsize': 16})
         
         else:
            axs[1, (params.index(param)-3)].plot(df.index,
                                                 df[f'ARMA_1_0_0{param}_0'])
            axs[1, (params.index(param)-3)].tick_params(labelsize=15)
            axs[1, (params.index(param)-3)
                ].set_title(f'ARMA(1,0) time series with phi=0.{param}', fontdict={'fontsize': 16})