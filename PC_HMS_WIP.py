import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import scipy.optimize as opt
from datetime import datetime, timedelta

# Download stock data from yahoo finance
ticker = 'AAPL'
start_date = '2010-01-01'
end = '2015-01-01'
start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.strptime(end, '%Y-%m-%d')

raw_data = yf.download(ticker, start=start_datetime, end=end_datetime)['Adj Close']
data = raw_data.pct_change().dropna()

# Estimating Historical Mean and Std Dev
mean_return_daily = data.mean()
sigma_return_daily = data.std()

# Define a small positive constant to avoid computational issues
epsilon = 1e-8

# Log-likelihood function for the Heston model
def heston_log_likelihood(params, data):
    kappa, theta, sigma, rho = params
    dt = 1 / 252  # 252 trading days in a year
    log_likelihood = 0  # Initializing log-likelihood
    S = data.values
    N = len(S)
    v = np.zeros(N)
    v[0] = np.var(S)  # Initial variance
    for t in range(1, N):
        dS = S[t] - S[t-1]
        dv = kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(max(v[t-1], epsilon)) * np.random.normal()
        v[t] = v[t-1] + dv
        v[t] = max(v[t], epsilon)
        log_likelihood += -0.5 * (np.log(2 * np.pi * v[t]) + (dS**2) / v[t])
    return -log_likelihood  # Negative log-likelihood for minimization

# Minimization of the negative log-likelihood
from scipy.optimize import differential_evolution
bounds = [(0.01, 10), (0.01, 0.2), (0.01, 1), (-0.9, 0.9)]
result = differential_evolution(heston_log_likelihood, bounds, args=(data,))
kappa, theta, sigma, rho = result.x
print(f"Estimated Parameters: kappa={kappa}, theta={theta}, sigma={sigma}, rho={rho}")

# Heston Model Parameters
S0 = raw_data[-1]  # Initial stock price
v0 = data.var()  # Use variance of the historical returns as initial variance
T = 1  # Time horizon of 1 year
dt = 1 / 252  # Time step (daily)
N = int(T / dt)  # Number of steps
mu = mean_return_daily * 252  # Drift or expected return of asset (Annualized)

# Number of simulations
num_simulations = 1000  # Reduced number for quicker execution

# Array to store the final prices of each simulation
final_prices = np.zeros(num_simulations)
simulated_paths = np.zeros((num_simulations, N))

# Perform Monte Carlo simulations
for sim in range(num_simulations):
    S = np.zeros(N)
    v = np.zeros(N)
    S[0] = S0
    v[0] = v0
    Z1 = np.random.normal(size=N)
    Z2 = np.random.normal(size=N)
    W_S = Z1 * np.sqrt(dt)
    W_v = rho * Z1 * np.sqrt(dt) + np.sqrt(1 - rho**2) * Z2 * np.sqrt(dt)
    for t in range(1, N):
        v[t] = v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(max(v[t-1], epsilon)) * W_v[t-1]
        v[t] = max(v[t], epsilon)
        S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(max(v[t-1], epsilon)) * W_S[t-1])
    final_prices[sim] = S[-1]
    simulated_paths[sim] = S

mean_final_price = np.mean(final_prices)
median_final_price = np.median(final_prices)
print(f"Mean of simulated final prices after {num_simulations} simulations: {mean_final_price}")
print(f"Median of simulated final prices after {num_simulations} simulations: {median_final_price}")

# Plot the distribution of final prices
plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=50, alpha=0.75)
plt.title(f'Distribution of Simulated Final Prices for {ticker}')
plt.xlabel('Final Price')
plt.ylabel('Frequency')
plt.show()

# Plot sample paths and historical price data for visual inspection
plt.figure(figsize=(10, 6))
x = 365
end_change = end_datetime + timedelta(days=x)
historical_backtest = yf.download(ticker, start=end, end=end_change)['Adj Close']
historical_prices = historical_backtest.values
plt.plot(historical_prices, label='Historical Price', color='black', linewidth=2)
for i in range(35):  # Plot more simulations for better comparison
    plt.plot(simulated_paths[i], alpha=0.3, label=f'Simulated Path {i+1}')
plt.title(f'Historical vs Simulated Stock Price Paths for {ticker}')
plt.xlabel('Time (days)')
plt.ylabel('Price')
plt.legend()
plt.show()
