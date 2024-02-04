import numpy as np
import scipy as sp
def gbm_simulator(S0, mu, sigma, T, N):
    
    S = np.zeros([N+1])
    S[0] = S0

   
    t = np.linspace(0, T, N+1)
    Z = np.random.normal(0, 1, N)
    
    for i in range(1, N+1):
        drift = (mu - 0.5 * sigma**2) * (t[i] - t[i-1])
        diffusion = sigma * Z[i-1] * np.sqrt(t[i] - t[i-1])
        S[i] = S[i-1] * np.exp(drift + diffusion)
    return S, t


S0 = 10  # Initial stock price
mu = 0.05  # Expected return (annualized)
sigma = 0.3  # Volatility (annualized)
T = 1  # Time  
N = 2 * 52  # Number of time steps (if daily timesteps) 
S, t = gbm_simulator(S0, mu, sigma, T, N)
# Graph
plt.plot(t, S)
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('GBM Simulation')
plt.show()