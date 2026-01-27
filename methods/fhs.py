import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Processing (Unchanged)
# ==========================================

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    data.dropna(inplace=True)
    return data

def garman_klass_vol(data):
    log_hl = np.log(data['High'] / data['Low'])
    log_co = np.log(data['Close'] / data['Open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    return np.sqrt(gk_var)

def prepare_fhs_data(data):
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data['GK_Vol'] = garman_klass_vol(data)
    data['Z'] = data['Log_Ret'] / (data['GK_Vol'] + 1e-9)
    return data.dropna()

# ==========================================
# 2. Simulation Engine (Fixed for Pandas Errors)
# ==========================================

def run_fhs(data, days_to_expiry=7, n_paths=10000):
    shock_pool = data['Z'].values
    
    current_vol = data['GK_Vol'].iloc[-1]
    if isinstance(current_vol, pd.Series): current_vol = current_vol.item()
        
    current_price = data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series): current_price = current_price.item()
    
    random_shocks = np.random.choice(shock_pool, size=(days_to_expiry, n_paths))
    simulated_daily_returns = random_shocks * current_vol
    cumulative_returns = np.sum(simulated_daily_returns, axis=0)
    simulated_prices = current_price * np.exp(cumulative_returns)
    
    return simulated_prices, current_price

# ==========================================
# 3. Spread Optimization (New Logic)
# ==========================================

def calculate_spread_metrics(sim_prices, short_strike, long_strike, spread_type='put'):
    """
    Calculates PnL vector and risk metrics for a specific vertical spread.
    """
    # 1. Calculate Fair Value of both legs based on simulation
    if spread_type == 'put':
        # Put: Max(K - S, 0)
        short_payoff = np.maximum(short_strike - sim_prices, 0)
        long_payoff = np.maximum(long_strike - sim_prices, 0)
    else:
        # Call: Max(S - K, 0)
        short_payoff = np.maximum(sim_prices - short_strike, 0)
        long_payoff = np.maximum(sim_prices - long_strike, 0)
        
    fair_value_short = np.mean(short_payoff)
    fair_value_long = np.mean(long_payoff)
    
    # 2. Net Credit (Fair Value * Edge Factor)
    # We assume we sell slightly better than fair value and buy slightly worse
    # due to bid-ask spread. Here we apply a flat "Edge" on the net result for simplicity.
    raw_credit = fair_value_short - fair_value_long
    market_credit = raw_credit * 1.10  # 10% edge assumption
    
    if market_credit <= 0.01: return None # Skip worthless spreads

    # 3. Calculate PnL Vector for the Spread
    # PnL = Credit - (Payout_Short - Payout_Long)
    # Note: Payouts are positive values (losses to the seller)
    net_payouts = short_payoff - long_payoff
    pnl_vector = market_credit - net_payouts
    
    # 4. Calculate Risk Metrics
    
    # CVaR (Expected Shortfall): Average loss of worst 5% outcomes
    tail_cutoff = int(len(pnl_vector) * 0.05)
    sorted_pnl = np.sort(pnl_vector)
    worst_5_percent = sorted_pnl[:tail_cutoff]
    
    # Filter for actual losses in the tail to calculate downside risk
    losses = worst_5_percent[worst_5_percent < 0]
    if len(losses) == 0:
        cvar = 0 # No losses in the worst 5% (Very safe, or low vol)
    else:
        cvar = abs(np.mean(losses))
        
    # Win Rate
    pop = np.sum(pnl_vector > 0) / len(pnl_vector)
    
    # Sortino-like Ratio: Credit / CVaR
    # If CVaR is 0 (safe trade), we cap ratio to avoid infinity
    ratio = market_credit / cvar if cvar > 0.01 else 999 

    return {
        'Type': spread_type.upper(),
        'Short': short_strike,
        'Long': long_strike,
        'Credit': round(market_credit, 2),
        'Max_Loss': abs(short_strike - long_strike) - market_credit,
        'PoP': round(pop * 100, 1),
        'CVaR': round(cvar, 2),
        'Ratio': round(ratio, 3)
    }

def optimize_spreads(sim_prices, current_price, spread_width=5):
    results = []
    
    # --- PUT SPREADS (Below Market) ---
    # Scan strikes from 15% OTM to 2% OTM
    start_k = int(current_price * 0.85)
    end_k = int(current_price * 0.98)
    
    for k_short in range(start_k, end_k, 1):
        k_long = k_short - spread_width
        metrics = calculate_spread_metrics(sim_prices, k_short, k_long, 'put')
        if metrics: results.append(metrics)

    # --- CALL SPREADS (Above Market) ---
    # Scan strikes from 2% OTM to 15% OTM
    start_k = int(current_price * 1.02)
    end_k = int(current_price * 1.15)
    
    for k_short in range(start_k, end_k, 1):
        k_long = k_short + spread_width
        metrics = calculate_spread_metrics(sim_prices, k_short, k_long, 'call')
        if metrics: results.append(metrics)
        
    df = pd.DataFrame(results)
    return df

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    TICKER = "NVDA"
    START = "2019-01-01"
    END = "2023-12-30"
    SPREAD_WIDTH = 5  # $5 Wide Spreads
    
    print(f"--- FHS Optimization for {TICKER} ---")
    df = fetch_data(TICKER, START, END)
    df_clean = prepare_fhs_data(df)
    sim_prices, spot = run_fhs(df_clean)
    
    print(f"Spot Price: ${spot:.2f}")
    print(f"Spread Width: ${SPREAD_WIDTH}")
    
    df_results = optimize_spreads(sim_prices, spot, spread_width=SPREAD_WIDTH)
    
    # Separate and Display Top Puts and Calls
    puts = df_results[df_results['Type'] == 'PUT'].sort_values(by='Ratio', ascending=False).head(5)
    calls = df_results[df_results['Type'] == 'CALL'].sort_values(by='Ratio', ascending=False).head(5)
    
    print("\n--- TOP 5 PUT SPREADS (High Efficiency) ---")
    print(puts[['Short', 'Long', 'Credit', 'PoP', 'CVaR', 'Ratio']].to_string(index=False))
    
    print("\n--- TOP 5 CALL SPREADS (High Efficiency) ---")
    print(calls[['Short', 'Long', 'Credit', 'PoP', 'CVaR', 'Ratio']].to_string(index=False))

    # Optional: Plotting the "Efficiency Frontier"
    plt.figure(figsize=(10, 6))
    
    puts_all = df_results[df_results['Type'] == 'PUT']
    calls_all = df_results[df_results['Type'] == 'CALL']
    
    plt.scatter(puts_all['Short'], puts_all['Ratio'], c='red', label='Put Spreads', alpha=0.6)
    plt.scatter(calls_all['Short'], calls_all['Ratio'], c='green', label='Call Spreads', alpha=0.6)
    
    plt.axvline(spot, color='black', linestyle='--', label='Spot Price')
    plt.xlabel('Short Strike Price ($)')
    plt.ylabel('Sortino Ratio (Credit / Tail Risk)')
    plt.title(f'Strike Selection Efficiency Frontier ({TICKER})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()