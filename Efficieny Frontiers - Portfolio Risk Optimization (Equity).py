#In Quant Finance, the Efficient Frontier is a method used for portfolio optimization to calculate the set(s) of portfolios which offer the highest expected return per level of risk.
#(or vice versa), the lowest possible risk per level of expected return

#For our model I have choosen to use Microsoft (MSFT) bc I hate Macintosh. Also our data will be from Yahoo. 

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandas_datareader as pdr

start_date = '2010-01-01'
end_date = '2023-01-01'

msft = pdr.get_data_yahoo('MSFT', start=start_date, end=end_date)

n_stocks = 100
n_dates = len(msft)

prices = pd.DataFrame(index=msft.index)
prices['Price'] = msft['Close']

portfolio_prices = pd.concat([prices] * n_stocks, ignore_index=True)

returns = portfolio_prices.pct_change().dropna()

n_assets = len(returns)

expected_returns = np.mean(returns, axis=1)
covariance_matrix = np.cov(returns.T)

def minimize_volatility(expected_returns, covariance_matrix, target_return):
    n_assets = len(expected_returns)

    args = (expected_returns, covariance_matrix)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    result = minimize(portfolio_volatility, n_assets*[1./n_assets,], args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)

    return result

n_points = 200
target_returns = np.linspace(0, 0.2, n_points)
volatilities = []

for target_return in target_returns:
    result = minimize_volatility(expected_returns, covariance_matrix, target_return)
    portfolio_volatility = np.sqrt(result.fun)
    volatilities.append(portfolio_volatility)

eff_frontier = pd.DataFrame({'Expected Return': target_returns, 'Volatility': volatilities})

#Plotting the efficient fromtier and calculating high - low and average risk metrics
plt.scatter(eff_frontier['Volatility'], eff_frontier['Expected Return'], c='r', label='Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.legend()
plt.grid()
plt.show(
    msft_returns = returns.iloc[:, 0]
msft_volatility = np.std(msft_returns)
msft_expected_return = np.mean(msft_returns)

print(f"Microsoft:")
print(f"  High Risk: {np.max(msft_returns)}")
print(f"  Low Risk: {np.min(msft_returns)}