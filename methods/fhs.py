import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Processing & Volatility Filtering
# ==========================================

def fetch_data(ticker, start_date, end_date):
    """Fetches OHLC data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data.dropna(inplace=True)
    return data

def garman_klass_vol(data):
    """
    Calculates the Garman-Klass volatility estimator.
    More efficient than Close-to-Close because it uses High/Low/Open.
    
    Formula: 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    """
    # Avoid division by zero by handling flat candles if necessary, though log(1)=0 is fine.
    log_hl = np.log(data['High'] / data['Low'])
    log_co = np.log(data['Close'] / data['Open'])
    
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    return np.sqrt(gk_var) # Return Volatility (Standard Deviation equivalent)

def prepare_fhs_data(data):
    """
    Prepares the 'Pool of Shocks' (Z-scores) for bootstrapping.
    """
    # 1. Calculate Raw Log Returns (Close-to-Close)
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 2. Calculate Garman-Klass Volatility for every historical day
    data['GK_Vol'] = garman_klass_vol(data)
    
    # 3. Filter (De-volatilize) the returns
    # We divide the return by the volatility of THAT specific day.
    # This creates a "Standardized Shock" (Z) that is regime-independent.
    # Epsilon added to prevent division by zero
    data['Z'] = data['Log_Ret'] / (data['GK_Vol'] + 1e-9)
    
    return data.dropna()

# ==========================================
# 2. Simulation Engine (Bootstrapping)
# ==========================================

def run_fhs(data, days_to_expiry=30, n_paths=10000):
    """
    Simulates future price paths using bootstrapped shocks scaled by CURRENT volatility.
    """
    # The pool of historical standardized shocks
    shock_pool = data['Z'].values
    
    # FIX: Extract as scalar floats to avoid Pandas Series alignment errors
    current_vol = data['GK_Vol'].iloc[-1]
    if isinstance(current_vol, pd.Series):
        current_vol = current_vol.item()
        
    current_price = data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series):
        current_price = current_price.item()
    
    # Randomly sample shocks
    random_shocks = np.random.choice(shock_pool, size=(days_to_expiry, n_paths))
    
    # Re-volatilize
    simulated_daily_returns = random_shocks * current_vol
    cumulative_returns = np.sum(simulated_daily_returns, axis=0)
    
    # This multiplication will now work because current_price is a float, not a Series
    simulated_prices = current_price * np.exp(cumulative_returns)
    
    return simulated_prices, current_price

# ==========================================
# 3. Strike Selection (CVaR & Sortino)
# ==========================================

def optimize_strike(simulated_prices, current_price, risk_tolerance_cvar=0.05):
    """
    Iterates through strikes to find optimal Risk/Reward.
    """
    # Define search range: 5% OTM to 20% OTM Puts
    start_strike = int(current_price * 0.80)
    end_strike = int(current_price * 0.95)
    strikes = range(start_strike, end_strike, 1) # $1 steps
    
    results = []
    
    for K in strikes:
        # 1. Calculate Theoretical Premium (Fair Value)
        # In a real system, you would grab the REAL BID price here.
        # We simulate "Fair Value" as the average payout of the option.
        intrinsic_values = np.maximum(K - simulated_prices, 0)
        fair_value = np.mean(intrinsic_values)
        
        # Apply a "Market Edge" (Assuming we sell for 10% over fair value for the example)
        market_premium = fair_value * 1.10
        if market_premium < 0.05: continue # Skip worthless options
        
        # 2. Calculate PnL Vector
        # Profit = Premium - Loss (if ITM)
        pnl_vector = market_premium - intrinsic_values
        
        # 3. Calculate CVaR (Expected Shortfall)
        # Average loss of the worst 5% of cases
        tail_cutoff = int(len(pnl_vector) * 0.05)
        sorted_pnl = np.sort(pnl_vector)
        worst_5_percent = sorted_pnl[:tail_cutoff]
        cvar = abs(np.mean(worst_5_percent)) # Absolute value for ratio calc
        
        # 4. Calculate Win Rate (PoP)
        pop = np.sum(pnl_vector > 0) / len(pnl_vector)
        
        # 5. Sortino-like Ratio (Reward / Tail Risk)
        ratio = market_premium / cvar if cvar > 0 else 0
        
        results.append({
            'Strike': K,
            'Premium': round(market_premium, 2),
            'PoP': round(pop * 100, 1),
            'CVaR': round(cvar, 2),
            'Ratio': round(ratio, 4),
            'Pct_OTM': round((1 - K/current_price)*100, 1)
        })
        
    df_results = pd.DataFrame(results)
    return df_results.sort_values(by='Ratio', ascending=False)

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    # Settings
    TICKER = "NVDA" 
    START = "2018-01-01" # Includes 2020 crash and 2022 bear market
    END = "2023-12-30"
    
    print(f"--- 1. Fetching Data for {TICKER} ---")
    df = fetch_data(TICKER, START, END)
    
    print("--- 2. Pre-processing & De-volatilizing ---")
    df_clean = prepare_fhs_data(df)
    
    print(f"--- 3. Running Simulation (10,000 Paths) ---")
    sim_prices, spot = run_fhs(df_clean, days_to_expiry=30, n_paths=10000)
    
    print(f"Current Spot: {spot:.2f}")
    
    print("--- 4. Optimizing Strike Selection ---")
    results = optimize_strike(sim_prices, spot)
    
    # Display Top 5 Strikes by Reward-to-Risk Ratio
    print("\nTop 5 Strikes (Sorted by Premium/CVaR Ratio):")
    print(results.head(5).to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(10,5))
    plt.hist(sim_prices, bins=100, density=True, alpha=0.6, color='blue', label='FHS Distribution')
    plt.axvline(spot, color='red', linestyle='--', label='Current Price')
    plt.title(f"Filtered Historical Simulation Distribution ({TICKER})")
    plt.legend()
    plt.show()