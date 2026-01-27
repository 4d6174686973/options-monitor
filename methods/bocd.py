import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Generate Synthetic Data (Bull Regime -> Crash Regime)
np.random.seed(42)
T = 200
# Regime 1: Steady Bull Market (Low Vol, Positive Drift)
data_1 = np.random.normal(loc=0.5, scale=0.5, size=100)
# Regime 2: Volatility Shock (High Vol, Negative Drift)
data_2 = np.random.normal(loc=-1.0, scale=2.5, size=100) 
returns = np.concatenate([data_1, data_2])
price = 100 * np.exp(np.cumsum(returns) / 100)

# 2. BOCD Implementation
def bocd(data, hazard_rate=1/30):
    T = len(data)
    # Initialize run length probabilities (matrix size T x T)
    # R[t, r] is prob that run length is r at time t
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1
    
    # Simple predictive model parameters (Gaussian)
    mu_0 = 0
    kappa_0 = 1
    alpha_0 = 1
    beta_0 = 1
    
    # Store max run length for plotting
    max_run_lengths = np.zeros(T)
    
    # Store posterior parameters to update
    mu_T = np.array([mu_0])
    kappa_T = np.array([kappa_0])
    alpha_T = np.array([alpha_0])
    beta_T = np.array([beta_0])

    for t, x in enumerate(data):
        # 1. Evaluate Predictive Probability (Student's t approx via Gaussian here for simplicity)
        # Using posterior predictive distribution
        df = 2 * alpha_T
        scale = np.sqrt(beta_T * (kappa_T + 1) / (alpha_T * kappa_T))
        pred_probs = norm.pdf(x, loc=mu_T, scale=scale)
        
        # 2. Calculate Growth Probabilities
        growth_probs = pred_probs * R[t, :t+1] * (1 - hazard_rate)
        
        # 3. Calculate Changepoint Probabilities
        cp_prob = np.sum(pred_probs * R[t, :t+1] * hazard_rate)
        
        # 4. Calculate Evidence and New Run Length Distribution
        evidence = np.sum(growth_probs) + cp_prob
        R[t+1, 1:t+2] = growth_probs / evidence
        R[t+1, 0] = cp_prob / evidence
        
        # 5. Update Sufficient Statistics for next step
        # (Updating the parameters for the new run length possibilities)
        new_mean = (kappa_T * mu_T + x) / (kappa_T + 1)
        new_kappa = kappa_T + 1
        new_alpha = alpha_T + 0.5
        new_beta = beta_T + (kappa_T * (x - mu_T)**2) / (2 * (kappa_T + 1))
        
        # Append "fresh" prior parameters for the case of a changepoint (r=0)
        mu_T = np.concatenate(([mu_0], new_mean))
        kappa_T = np.concatenate(([kappa_0], new_kappa))
        alpha_T = np.concatenate(([alpha_0], new_alpha))
        beta_T = np.concatenate(([beta_0], new_beta))
        
        # Determine most likely run length
        max_run_lengths[t] = np.argmax(R[t+1])

    return max_run_lengths, R

# Run BOCD
run_lengths, R_matrix = bocd(returns)

# 3. Visualization
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot Price
ax[0].plot(price, color='black', label='Asset Price')
ax[0].axvline(x=100, color='red', linestyle='--', label='True Regime Change')
ax[0].set_title('Asset Price (Bull Market to Vol Shock)')
ax[0].legend()

# Plot Run Length
ax[1].plot(run_lengths, color='blue', label='Inferred Run Length')
ax[1].set_title('BOCD Run Length (Drops to 0 = Regime Change Detected)')
ax[1].set_ylabel('Days since last change')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()