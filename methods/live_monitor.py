import asyncio
import logging
import numpy as np
import pandas as pd
from ib_insync import *
from datetime import datetime, timedelta
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

# 1. SILENCE LOGGING
logging.getLogger('ib_insync').setLevel(logging.CRITICAL)

def onError(reqId, errorCode, errorString, contract):
    if errorCode in [10091, 2104, 2106, 2158]: return
    print(f"{Fore.RED}IBKR Error {errorCode}: {errorString}")

# ==========================================
# 2. DATA ENGINE (Native IBKR)
# ==========================================
async def fetch_ib_history(ib, contract, duration='3 Y'):
    """
    Fetches 3 Years of Daily candles from IBKR.
    """
    print(f"{Fore.CYAN}--- Fetching IBKR History for {contract.symbol} ---")
    
    # Check if contract is qualified, if not, qualify it
    if not contract.conId:
        await ib.qualifyContractsAsync(contract)
    
    # Request History
    # whatToShow='TRADES' is standard for generic High/Low volatility
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,        # Regular Trading Hours only (avoids bad spikes)
        formatDate=1,
        keepUpToDate=False
    )
    
    if not bars:
        raise ValueError("No historical data returned from IBKR.")

    # Convert to DataFrame
    df = util.df(bars)
    
    # Normalize Columns (IBKR uses lowercase, we need Title Case for consistency)
    df.rename(columns={
        'date': 'Date', 
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'volume': 'Volume'
    }, inplace=True)
    
    df.set_index('Date', inplace=True)
    return df

# ==========================================
# 3. MATH ENGINE (FHS Simulation)
# ==========================================
def run_fhs_simulation(df, days_to_expiry, n_paths=20000):
    """
    Runs Filtered Historical Simulation on the provided DataFrame.
    Note: n_paths increased to 20,000 to capture tails in short timeframes.
    """
    # Garman-Klass Volatility
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    df['GK_Vol'] = np.sqrt(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2)
    
    # Filter Shocks
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Z'] = df['Log_Ret'] / (df['GK_Vol'] + 1e-9)
    df.dropna(inplace=True)
    
    # Bootstrap
    shock_pool = df['Z'].values
    
    # Use the LAST KNOWN volatility and price from history
    current_vol = df['GK_Vol'].iloc[-1]
    if isinstance(current_vol, pd.Series): current_vol = current_vol.item()
    
    current_price = df['Close'].iloc[-1]
    if isinstance(current_price, pd.Series): current_price = current_price.item()

    # Simulate
    random_shocks = np.random.choice(shock_pool, size=(days_to_expiry, n_paths))
    sim_returns = random_shocks * current_vol
    cum_returns = np.sum(sim_returns, axis=0)
    sim_prices = current_price * np.exp(cum_returns)
    
    return sim_prices, current_price

# ==========================================
# 4. PRICING UTILS
# ==========================================
def get_robust_price(ticker, side):
    if side == 'sell' and ticker.bid > 0: return ticker.bid
    if side == 'buy' and ticker.ask > 0: return ticker.ask
    if ticker.close > 0: return ticker.close
    if ticker.last > 0: return ticker.last
    return 0.0

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
async def main():
    ib = IB()
    ib.errorEvent += onError
    
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=97)
    except Exception as e:
        print(f"{Fore.RED}Connection Failed: {e}")
        return

    # Use Delayed Frozen (Type 4) for testing
    ib.reqMarketDataType(4)
    print(f"{Fore.YELLOW}Connected. Data Type: Frozen/Delayed.")

    SYMBOL = 'SPY'
    contract = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(contract)
    
    # --- STEP 1: GET IBKR HISTORY ---
    try:
        history_df = await fetch_ib_history(ib, contract)
    except Exception as e:
        print(f"{Fore.RED}Failed to fetch history: {e}")
        ib.disconnect()
        return

    # --- STEP 2: CALCULATE DTE ---
    today = datetime.now().date()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0: days_until_friday = 0 # 0DTE logic (Intraday)
    
    # If today is Friday, we might want to target next week, 
    # but for now let's simulate the remaining time. 
    # If 0 days left, we simulate 1 day to be safe.
    sim_days = max(1, days_until_friday)
    
    print(f"Simulation Horizon: {sim_days} Day(s) (Matching DTE)")
    
    # --- STEP 3: RUN FHS ---
    sim_prices, spot_ref = run_fhs_simulation(history_df, days_to_expiry=sim_days)
    print(f"Ref Spot (IBKR): {spot_ref:.2f}")

    # --- STEP 4: CHAIN & STRIKES ---
    chains = await ib.reqSecDefOptParamsAsync(contract.symbol, '', contract.secType, contract.conId)
    
    this_friday = today + timedelta(days=days_until_friday)
    expiry_str = this_friday.strftime('%Y%m%d')

    target_chain = None
    for c in chains:
        if expiry_str in c.expirations:
            target_chain = c
            break
            
    if not target_chain:
        print(f"Expiry {expiry_str} not found.")
        ib.disconnect()
        return

    print(f"Targeting Expiry: {expiry_str}")

    # Filter Strikes (0.5% to 5% OTM)
    valid_strikes = sorted(list(target_chain.strikes))
    upper_bound = spot_ref * 0.995
    lower_bound = spot_ref * 0.950
    potential_shorts = [k for k in valid_strikes if lower_bound < k < upper_bound]
    
    print(f"Scanning {len(potential_shorts)} strikes ({lower_bound:.1f} - {upper_bound:.1f})")

    # --- STEP 5: BUILD & SCAN ---
    SPREAD_WIDTHS = [1, 5, 10, 15] 
    contracts_to_req = []
    spread_combos = []
    seen = set()
    unique_reqs = []

    for k_short in potential_shorts:
        for width in SPREAD_WIDTHS:
            k_long = k_short - width
            if k_long not in valid_strikes: continue

            c_short = Option(SYMBOL, expiry_str, k_short, 'P', 'SMART', tradingClass=target_chain.tradingClass)
            c_long = Option(SYMBOL, expiry_str, k_long, 'P', 'SMART', tradingClass=target_chain.tradingClass)
            
            spread_combos.append({'short': c_short, 'long': c_long, 'width': width})
            
            for c in [c_short, c_long]:
                key = (c.strike, c.right)
                if key not in seen:
                    unique_reqs.append(c)
                    seen.add(key)

    contracts_to_req = unique_reqs
    contracts_to_req = await ib.qualifyContractsAsync(*contracts_to_req)
    
    print("Requesting Snapshots...")
    tickers = {}
    for c in contracts_to_req:
        tickers[c.conId] = ib.reqMktData(c, '', True, False)
        
    for i in range(10):
        await asyncio.sleep(1)
        filled = sum(1 for t in tickers.values() if t.close > 0 or t.last > 0 or t.bid > 0)
        print(f"Data Fill: {filled}/{len(tickers)}...")
        if filled == len(tickers): break

    print(f"\n{Fore.GREEN}=== ANALYSIS RESULTS ===")

    while True:
        opportunities = []
        
        for combo in spread_combos:
            short_c = combo['short']
            long_c = combo['long']
            width = combo['width']
            
            t_short = tickers.get(short_c.conId)
            t_long = tickers.get(long_c.conId)
            
            if not t_short or not t_long: continue
            
            price_short = get_robust_price(t_short, 'sell')
            price_long = get_robust_price(t_long, 'buy')
            
            if price_short == 0 or price_long == 0: continue

            net_credit = price_short - price_long
            if net_credit < 0.01: continue 
            
            # --- FHS METRICS ---
            short_payoff = np.maximum(short_c.strike - sim_prices, 0)
            long_payoff = np.maximum(long_c.strike - sim_prices, 0)
            pnl_vector = net_credit - (short_payoff - long_payoff)
            
            tail_cutoff = int(len(pnl_vector) * 0.05)
            sorted_pnl = np.sort(pnl_vector)
            worst_5 = sorted_pnl[:tail_cutoff]
            losses = worst_5[worst_5 < 0]
            cvar = abs(np.mean(losses)) if len(losses) > 0 else 0
            
            ratio = net_credit / cvar if cvar > 0.01 else 20.0
            pop = np.sum(pnl_vector > 0) / len(pnl_vector)
            
            opportunities.append({
                'Short': short_c.strike,
                'Width': width,
                'Credit': net_credit,
                'CVaR': cvar,
                'Ratio': ratio,
                'PoP': pop * 100
            })
            
        df = pd.DataFrame(opportunities)
        if not df.empty:
            df = df.sort_values(by='Ratio', ascending=False).head(10)
            print(f"\n{datetime.now().strftime('%H:%M:%S')} | TOP 10 OPPORTUNITIES")
            print("-" * 65)
            print(f"{'Short':<8} {'Width':<6} {'Credit':<8} {'CVaR':<8} {'Ratio':<8} {'PoP':<6}")
            print("-" * 65)
            for _, row in df.iterrows():
                c = Fore.GREEN if row['Ratio'] > 0.4 else Fore.WHITE
                print(f"{c}{row['Short']:<8.1f} {row['Width']:<6} {row['Credit']:<8.2f} {row['CVaR']:<8.2f} {row['Ratio']:<8.3f} {row['PoP']:<6.1f}")
        else:
            print(f"{Fore.YELLOW}No valid spreads found.")

        break # Exit after one pass for snapshot mode

    ib.disconnect()

if __name__ == '__main__':
    asyncio.run(main())