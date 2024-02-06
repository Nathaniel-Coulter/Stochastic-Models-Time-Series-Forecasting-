import numpy as np
from scipy.optimize import minimize

#past returns of 3 assets
returns = np.array([[0.01, 0.02, 0.03],
                     [0.04, 0.05, 0.06],
                     [0.07, 0.08, 0.09]])


expected_returns = np.mean(returns, axis=1)

#covariance matrix of returns
covariance_matrix = np.cov(returns)

# Function objective for optimization
def optimize_portfolio(expected_returns, covariance_matrix, target_return=0.05):
    n_assets = len(expected_returns)
    args = (expected_returns, covariance_matrix)

    #constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    #Optimizer
    result = minimize(portfolio_volatility, n_assets*[1./n_assets,], args=args,
                       method='SLSQP', bounds=bounds, constraints=constraints)

    return result

#portfolio volatility function 
def portfolio_volatility(weights, expected_returns, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

# set optimizer bounds
bounds = tuple((0, 1) for asset in range(len(expected_returns)))

# run to return optimal portfolio
optimal_portfolio = optimize_portfolio(expected_returns, covariance_matrix)

print("Optimal weights:")
print(optimal_portfolio.x)
print("Portfolio volatility:")
print(portfolio_volatility(optimal_portfolio.x, expected_returns, covariance_matrix))

#Using our friends from Linear Algebra, vectors, set a target return

target_return = 0.05

#expected returns & covariance matrix 
expected_returns = np.mean(returns, axis=1)
covariance_matrix = np.cov(returns)

# Optimizer bounds 
bounds = tuple((0, 1) for asset in range(len(expected_returns)))

# set the constraints for our optimization problem
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# make sure to define the optimizer
result = minimize(portfolio_volatility, n_assets*[1./n_assets,], args=(expected_returns, covariance_matrix),
                   method='SLSQP', bounds=bounds, constraints=constraints)

#vouilla !
print("Optimal weights:")
print(result.x)
print("Portfolio volatility:")
print(portfolio_volatility(result.x, expected_returns, covariance_matrix))