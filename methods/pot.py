import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Parameters
current_price = 100.00
strike_price = 95.00  # Selling a Put spread here
days_to_expire = 5
simulations = 5000
dt = 1/390 # Granularity: Minutes in a trading day (approx)
steps = int(days_to_expire * 390) 

# Volatility (Annualized)
sigma = 0.40 # 40% IV (High vol environment described in 2025)

# 2. Monte Carlo Engine
# Generate random paths using Geometric Brownian Motion
# (In the advanced version, we would sample from the KDE distribution instead of normal)
rets = np.random.normal(0, sigma * np.sqrt(1/252/390), (simulations, steps))
price_paths = current_price * np.exp(np.cumsum(rets, axis=1))

# 3. Calculate Probability of Touch (POT)
# Check if ANY point in the path drops below the strike
touches = np.any(price_paths <= strike_price, axis=1)
pot = np.sum(touches) / simulations

# Calculate Probability of ITM (Expiring below strike)
itm = price_paths[:, -1] <= strike_price
prob_itm = np.sum(itm) / simulations

# 4. Visualization
plt.figure(figsize=(12, 6))
# Plot first 50 paths
plt.plot(price_paths[:50].T, color='grey', alpha=0.3)
plt.axhline(strike_price, color='red', linewidth=2, label='Strike Price')
plt.title(f'Monte Carlo Simulation (5 Days). POT: {pot:.1%} vs P(ITM): {prob_itm:.1%}')
plt.ylabel('Price')
plt.xlabel('Trading Minutes')
plt.legend()
plt.show()

print(f"Theoretical Delta (approx P_ITM): {prob_itm:.2%}")
print(f"Real Risk (Probability of Touch): {pot:.2%}")
print(f"Ratio: {pot/prob_itm:.2f}x (POT is typically ~2x Delta)")