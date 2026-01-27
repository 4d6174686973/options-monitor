import asyncio
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from ib_insync import *
from datetime import datetime, timedelta
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

# ==========================================
# 1. SILENCE THE LOGGING SPAM
# ==========================================
# This is the nuclear option to stop Error 10091 from flooding the console
logging.getLogger('ib_insync').setLevel(logging.CRITICAL)

def onError(reqId, errorCode, errorString, contract):
    # We still keep this to catch critical connection issues
    if errorCode in [10091, 2104, 2106, 2158]: return
    print(f"{Fore.RED}IBKR Error {errorCode}: {errorString}")

# ==========================================
# 2. MATH ENGINE (FHS Simulation)
# ==========================================
def get_fhs_distribution(ticker, days_to_expiry=30, n_paths=5000):
    print(f"{Fore.CYAN}--- FHS: Fetching history for {ticker} ---")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
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
    current_vol = df['GK_Vol'].iloc[-1]
    if isinstance(current_vol, pd.Series): current_vol = current_vol.item()
    current_price = df['Close'].iloc[-1]
    if isinstance(current_price, pd.Series): current_price = current_price.item()

    random_shocks = np.random.choice(shock_pool, size=(days_to_expiry, n_paths))
    sim_returns = random_shocks * current_vol
    cum_returns = np.sum(sim_returns, axis=0)
    sim_prices = current_price * np.exp(cum_returns)
    
    return sim_prices, current_price

# ==========================================
# 3. ROBUST PRICING LOGIC
# ==========================================
def get_robust_price(ticker, side):
    """
    Prioritizes CLOSE price over LAST price to avoid stale data issues.
    """
    # 1. If Market is Open, use Bid/Ask
    if side == 'sell' and ticker.bid > 0: return ticker.bid
    if side == 'buy' and ticker.ask > 0: return ticker.ask
    
    # 2. If Market Closed/Frozen, use CLOSE (Settlement)
    # This is safer than 'Last' which can be asynchronous/stale
    if ticker.close > 0: return ticker.close
    
    # 3. Fallback to Last, then midpoint
    if ticker.last > 0: return ticker.last
    
    return 0.0

# ==========================================
# 4. MAIN LOOP
# ==========================================
async def main():
    ib = IB()
    ib.errorEvent += onError
    
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=97)
    except Exception as e:
        print(f"{Fore.RED}Connection Failed: {e}")
        return

    # Use 'Frozen' data (Type 4) so we get Closing Prices
    ib.reqMarketDataType(4)
    print(f"{Fore.YELLOW}Connected. Using Delayed Frozen Data (Type 4).")

    SYMBOL = 'SPY'

    today = datetime.now().date()
    days_until_friday = (4 - today.weekday()) % 7
    this_friday = today + timedelta(days=days_until_friday)

    # Calculate DTE dynamically
    dte = (this_friday - today).days
    if dte == 0: dte = 1 # Handle Friday 0DTE as 1-day vol

    # Pass the actual DTE into the simulation
    sim_prices, spot_ref = get_fhs_distribution(SYMBOL, days_to_expiry=dte)

    # --- CHAIN SELECTION ---
    contract = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(contract)
    chains = await ib.reqSecDefOptParamsAsync(contract.symbol, '', contract.secType, contract.conId)
    
    # Find Weekly Expiry
    expiry_str = this_friday.strftime('%Y%m%d')

    target_chain = None
    for c in chains:
        if expiry_str in c.expirations:
            target_chain = c
            break
            
    if not target_chain:
        print(f"Expiry {expiry_str} not found.")
        return

    print(f"Targeting Expiry: {expiry_str} (Class: {target_chain.tradingClass})")

    # --- STRIKE FILTERING (Weekly Aware) ---
    valid_strikes = sorted(list(target_chain.strikes))
    
    # Look for strikes 0.5% to 5.0% OTM
    upper_bound = spot_ref * 0.995
    lower_bound = spot_ref * 0.950
    
    potential_shorts = [k for k in valid_strikes if lower_bound < k < upper_bound]
    print(f"Scanning {len(potential_shorts)} strikes between {lower_bound:.1f} and {upper_bound:.1f}")

    # --- BUILD SPREADS ---
    SPREAD_WIDTHS = [1, 2, 3, 5, 10, 15, 20]
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
    print(f"Qualifying {len(contracts_to_req)} contracts...")
    contracts_to_req = await ib.qualifyContractsAsync(*contracts_to_req)
    
    # --- REQUEST DATA (SNAPSHOT MODE) ---
    # Snapshot=True is often cleaner for frozen/post-market data
    print("Requesting Market Data Snapshots...")
    tickers = {}
    for c in contracts_to_req:
        tickers[c.conId] = ib.reqMktData(c, '', True, False) # Snapshot=True
        
    # Wait for Data
    print("Waiting for data fill...")
    for i in range(10):
        await asyncio.sleep(1)
        filled = sum(1 for t in tickers.values() if t.close > 0 or t.last > 0 or t.bid > 0)
        print(f"Data Fill: {filled}/{len(tickers)}...")
        if filled == len(tickers): break

    print(f"\n{Fore.GREEN}=== DATA DEBUG (First 3 Contracts) ===")
    debug_count = 0
    for t in tickers.values():
        if debug_count >= 3: break
        print(f"Strike: {t.contract.strike} | Bid: {t.bid} | Ask: {t.ask} | Last: {t.last} | Close: {t.close}")
        debug_count += 1

    print(f"\n{Fore.GREEN}=== RUNNING ANALYSIS ===")

    while True:
        opportunities = []
        
        for combo in spread_combos:
            short_c = combo['short']
            long_c = combo['long']
            width = combo['width']
            
            t_short = tickers.get(short_c.conId)
            t_long = tickers.get(long_c.conId)
            
            if not t_short or not t_long: continue
            
            # GET PRICES (Prioritizing Close)
            price_short = get_robust_price(t_short, 'sell')
            price_long = get_robust_price(t_long, 'buy')
            
            if price_short == 0 or price_long == 0: continue

            net_credit = price_short - price_long
            
            # Filter low value trades
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
            
            pop = np.sum(pnl_vector > 0) / len(pnl_vector)
            ratio = net_credit / cvar if cvar > 0.01 else 10.0
            
            opportunities.append({
                'Short': short_c.strike,
                'Width': width,
                'Credit': net_credit,
                'CVaR': cvar,
                'Ratio': ratio,
                'PoP': pop * 100
            })
            
        # Display
        df = pd.DataFrame(opportunities)
        if not df.empty:
            df = df.sort_values(by='Ratio', ascending=False).head(5)
            print(f"\n{datetime.now().strftime('%H:%M:%S')} | TOP OPPORTUNITIES")
            print("-" * 65)
            print(f"{'Short':<8} {'Width':<6} {'Credit':<8} {'CVaR':<8} {'Ratio':<8} {'PoP':<6}")
            print("-" * 65)
            for _, row in df.iterrows():
                c = Fore.GREEN if row['Ratio'] > 0.4 else Fore.WHITE
                print(f"{c}{row['Short']:<8.1f} {row['Width']:<6} {row['Credit']:<8.2f} {row['CVaR']:<8.2f} {row['Ratio']:<8.3f} {row['PoP']:<6.1f}")
        else:
            print(f"{Fore.YELLOW}No valid spreads found. (Credit too low or Prices missing)")

        # In snapshot mode, data doesn't update, so we break after one pass
        # In live mode (snapshot=False), you would sleep and loop.
        break 

    print(f"\n{Fore.CYAN}Analysis Complete. Disconnecting...")
    ib.disconnect()

if __name__ == '__main__':
    asyncio.run(main())