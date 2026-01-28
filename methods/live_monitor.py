import asyncio
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from ib_insync import *
from datetime import datetime, timedelta
import colorama
from colorama import Fore, Style
from concurrent.futures import ThreadPoolExecutor

colorama.init(autoreset=True)

# 1. SILENCE LOGGING
logging.getLogger('ib_insync').setLevel(logging.CRITICAL)

def onError(reqId, errorCode, errorString, contract):
    if errorCode in [10091, 2104, 2106, 2158, 10276]: return
    print(f"{Fore.RED}IBKR Error {errorCode}: {errorString}")

# ==========================================
# 2. ROBUST EVENT CHECKER (Threaded)
# ==========================================
executor = ThreadPoolExecutor(max_workers=1)

def _blocking_fetch_events(ticker, days_ahead):
    """
    Fetches events from Yahoo Finance. Runs in a separate thread.
    """
    risk_msg = []
    has_risk = False
    today = datetime.now().date()
    limit_date = today + timedelta(days=days_ahead)

    try:
        t = yf.Ticker(ticker)
        
        # --- 1. CHECK EARNINGS (Robust Dictionary Handling) ---
        earnings_dates = []
        try:
            cal = t.calendar
            # Fix: Check if 'cal' is a dictionary (new behavior) or DataFrame (old behavior)
            if isinstance(cal, dict):
                # New yfinance returns dict: {'Earnings Date': [datetime.date(2025, 1, 30)]}
                if 'Earnings Date' in cal:
                    dates = cal['Earnings Date']
                    for d in dates:
                        # Sometimes d is a date object, sometimes a datetime
                        if hasattr(d, 'date'): earnings_dates.append(d.date())
                        else: earnings_dates.append(d) # Assume it's already a date
            elif cal is not None and not cal.empty:
                # Old yfinance returns DataFrame
                if 'Earnings Date' in cal:
                    for d in cal['Earnings Date']:
                         if hasattr(d, 'date'): earnings_dates.append(d.date())
        except Exception: 
            # 404 Errors or missing data often happen here for ETFs (SPY has no earnings)
            pass

        # Fallback: check t.earnings_dates if calendar failed
        if not earnings_dates:
            try:
                ed_df = t.earnings_dates
                if ed_df is not None and not ed_df.empty:
                    future_df = ed_df[(ed_df.index > pd.Timestamp(today)) & 
                                      (ed_df.index <= pd.Timestamp(limit_date))]
                    for ts in future_df.index:
                        earnings_dates.append(ts.date())
            except Exception: pass

        for ed in earnings_dates:
            if today <= ed <= limit_date:
                has_risk = True
                risk_msg.append(f"EARNINGS ({ed})")

        # --- 2. CHECK DIVIDENDS (Robust Timezone Handling) ---
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                try: divs.index = divs.index.tz_localize(None)
                except: pass
                
                future_divs = divs[(divs.index >= pd.Timestamp(today)) & 
                                   (divs.index <= pd.Timestamp(limit_date))]
                for ts in future_divs.index:
                    risk_msg.append(f"DIVIDEND ({ts.date()})")
        except Exception: pass

    except Exception as e:
        # Return the specific error message for debugging, but don't crash
        return True, f"Data Fetch Error: {str(e)[:100]}"

    if has_risk:
        return True, " | ".join(set(risk_msg))
    
    return False, "Clear"

async def check_events_async(ticker, days_ahead):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _blocking_fetch_events, ticker, days_ahead)

# ==========================================
# 3. MATH ENGINE (Advanced FHS)
# ==========================================
def run_advanced_fhs(df, days_to_expiry, n_paths=20000, skew_factor=1.5):
    # Stats
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    gk_var = np.maximum(0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2, 0)
    df['GK_Vol'] = np.sqrt(gk_var)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Current Regime
    current_vol = df['GK_Vol'].iloc[-1]
    if isinstance(current_vol, pd.Series): current_vol = current_vol.item()
    current_price = df['Close'].iloc[-1]
    if isinstance(current_price, pd.Series): current_price = current_price.item()
    
    # Normalize
    df['Z'] = df['Log_Ret'] / (df['GK_Vol'] + 1e-9)
    shock_pool = df['Z'].dropna().values
    
    # Simulation
    sim_prices = np.full(n_paths, current_price)
    sim_vols = np.full(n_paths, current_vol)
    
    for day in range(days_to_expiry):
        random_shocks = np.random.choice(shock_pool, size=n_paths)
        step_ret = random_shocks * sim_vols
        sim_prices = sim_prices * np.exp(step_ret)
        
        # Vol Sway Update
        price_ratio = current_price / sim_prices
        vol_scaler = np.power(price_ratio, skew_factor)
        vol_scaler = np.clip(vol_scaler, 0.5, 3.0) 
        sim_vols = current_vol * vol_scaler

    avg_sim_vol = np.mean(sim_vols) * np.sqrt(252)
    return sim_prices, current_price, avg_sim_vol

# ==========================================
# 4. IBKR UTILS
# ==========================================
async def fetch_ib_history(ib, contract):
    bars = await ib.reqHistoricalDataAsync(
        contract, endDateTime='', durationStr='2 Y',
        barSizeSetting='1 day', whatToShow='TRADES', useRTH=True, formatDate=1, keepUpToDate=False
    )
    df = util.df(bars)
    df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df.set_index('Date', inplace=True)
    return df

def get_market_data(ticker):
    bid = ticker.bid if ticker.bid > 0 else (ticker.close if ticker.close > 0 else ticker.last)
    ask = ticker.ask if ticker.ask > 0 else (ticker.close if ticker.close > 0 else ticker.last)
    iv = ticker.modelGreeks.impliedVol if (ticker.modelGreeks and ticker.modelGreeks.impliedVol) else 0.0
    return bid, ask, iv

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
async def main():
    ib = IB()
    ib.errorEvent += onError
    
    try:
        await ib.connectAsync('127.0.0.1', 7497, clientId=93)
    except Exception as e:
        print(f"{Fore.RED}Connection Failed: {e}"); return

    # Use Frozen (Type 4) for post-market testing compatibility
    ib.reqMarketDataType(4)
    print(f"{Fore.YELLOW}Connected. Data Type: Frozen/Delayed.")

    SYMBOL = 'VST'
    contract = Stock(SYMBOL, 'SMART', 'USD')
    await ib.qualifyContractsAsync(contract)
    
    # --- 1. INITIALIZATION & DATA FETCH ---
    try:
        hist_df = await fetch_ib_history(ib, contract)
    except Exception as e:
        print(f"Data Error: {e}"); return

    # Calculate Dates once for initial check
    today = datetime.now().date()
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0: days_until_friday = 0 
    sim_days = max(1, days_until_friday)

    # --- 2. ROBUST EVENT CHECK (ONCE) ---
    print(f"{Fore.CYAN}--- Checking Upcoming Events for {SYMBOL} ---")
    risky_event, event_msg = await check_events_async(SYMBOL, sim_days)
    
    event_status_str = ""
    if risky_event:
        event_status_str = f"{Fore.RED}[WARNING: {event_msg}]"
        print(f"{Fore.RED}!!! EVENT RISK DETECTED: {event_msg} !!!")
        print(f"{Fore.YELLOW}Proceeding with monitor for informational purposes only.")
        await asyncio.sleep(2) # Pause so user sees it
    else:
        event_status_str = f"{Fore.GREEN}[Events: Clear]"
        print(f"{Fore.GREEN}Events Check: Clear. Safe to proceed.")

    # --- 3. MONITORING LOOP ---
    while True: 
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"\n{Fore.CYAN}--- SCAN: {SYMBOL} ({timestamp}) {event_status_str} {Fore.CYAN}---")
        
        # Run FHS
        sim_prices, spot_ref, fhs_iv = run_advanced_fhs(hist_df, sim_days, skew_factor=1.2)
        print(f"Spot: {spot_ref:.2f} | Forecast Risk (IV): {fhs_iv*100:.1f}%")

# 1. Define the "Wildcard" Contract
        # We ask for: NVDA, Puts, Smart, Specific Expiry.
        # We leave 'strike' BLANK (0.0). This tells IBKR: "Give me ALL strikes."
        this_friday = today + timedelta(days=days_until_friday)
        expiry_str = this_friday.strftime('%Y%m%d')
        
        wildcard_contract = Option(SYMBOL, expiry_str, 0, 'P', 'SMART')
        
        print(f"Fetching valid contracts for expiry {expiry_str}...")
        try:
            # This returns a list of ContractDetails objects
            contract_details_list = await ib.reqContractDetailsAsync(wildcard_contract)
        except Exception as e:
            print(f"Failed to fetch contracts: {e}"); await asyncio.sleep(60); continue

        if not contract_details_list:
            print(f"No contracts found for {expiry_str}. Market holiday?"); await asyncio.sleep(60); continue

        # 2. Extract Valid Contracts & Strikes
        # We create a dictionary {strike: contract_object} for instant lookup
        valid_contracts = {}
        for cd in contract_details_list:
            c = cd.contract
            valid_contracts[c.strike] = c
            
        print(f"Found {len(valid_contracts)} valid Puts.")

        # 3. Filter Strikes (Using the GUARANTEED valid list)
        valid_strikes = sorted(list(valid_contracts.keys()))
        upper = spot_ref * 0.995
        lower = spot_ref * 0.95
        shorts = [k for k in valid_strikes if lower < k < upper]
        
        spreads = []
        req_list = []
        seen = set()
        
        for k_short in shorts:
            for w in [1, 5, 10]:
                k_long = k_short - w
                
                # Check if the long leg exists in our valid list
                if k_long in valid_contracts:
                    # USE THE EXACT OBJECT IBKR GAVE US
                    # No more guessing classes or symbols.
                    s_c = valid_contracts[k_short]
                    l_c = valid_contracts[k_long]
                    
                    spreads.append({'s': s_c, 'l': l_c, 'w': w})
                    
                    for c in [s_c, l_c]:
                        if c.conId not in seen:
                            req_list.append(c)
                            seen.add(c.conId)

        print(f"Constructed {len(spreads)} spreads. Requesting data...")
        
        # [Proceed to Request Data...]
        # Note: These contracts are already fully qualified (they have conIds),
        # so you don't technically need qualifyContractsAsync, but it hurts nothing.
        if req_list:
            req_list = await ib.qualifyContractsAsync(*req_list)
        # Request Streaming Data (snapshot=False for IV)
        req_list = await ib.qualifyContractsAsync(*req_list)
        tickers = {c.conId: ib.reqMktData(c, '100,101,104,106', False, False) for c in req_list}
        
        print(f"Waiting for Greeks ({len(req_list)} contracts)...")
        # Smart Wait: Wait until 80% of tickers have IV data
        for _ in range(10):
            await asyncio.sleep(0.5)
            filled = sum(1 for t in tickers.values() if t.modelGreeks and t.modelGreeks.impliedVol)
            if filled > len(tickers) * 0.8: break
            
        # Analysis
        opportunities = []
        for s in spreads:
            t_s = tickers.get(s['s'].conId)
            t_l = tickers.get(s['l'].conId)
            
            if not t_s or not t_l: continue
            
            bid_s, _, iv_s = get_market_data(t_s)
            _, ask_l, _ = get_market_data(t_l)
            
            if iv_s == 0: continue 
            
            # VRP Check
            vrp_ratio = iv_s / fhs_iv
            if vrp_ratio < 1.0: continue

            credit = bid_s - ask_l
            if credit < 0.01: continue
            
            s_pay = np.maximum(s['s'].strike - sim_prices, 0)
            l_pay = np.maximum(s['l'].strike - sim_prices, 0)
            pnl = credit - (s_pay - l_pay)
            
            worst_5 = np.sort(pnl)[:int(len(pnl)*0.05)]
            losses = worst_5[worst_5 < 0]
            cvar = abs(np.mean(losses)) if len(losses) > 0 else 0
            
            pop = np.sum(pnl > 0) / len(pnl)
            ratio = credit / cvar if cvar > 0.01 else 20.0
            
            opportunities.append({
                'Strike': s['s'].strike,
                'Width': s['w'],
                'Credit': credit,
                'IV': iv_s * 100,
                'VRP': vrp_ratio,
                'CVaR': cvar,
                'Ratio': ratio,
                'PoP': pop * 100
            })

        # Cleanup
        for c in req_list: ib.cancelMktData(c)
            
        # Display
        df = pd.DataFrame(opportunities)
        if not df.empty:
            df = df.sort_values(by='Ratio', ascending=False).head(10)
            print(f"\n{Fore.GREEN}TOP VRP OPPORTUNITIES")
            print("-" * 85)
            print(f"{'Strike':<8} {'Width':<6} {'Credit':<8} {'IV%':<6} {'VRP':<6} {'CVaR':<8} {'Sortino':<8} {'PoP%':<6}")
            print("-" * 85)
            for _, r in df.iterrows():
                print(f"{r['Strike']:<8.1f} {r['Width']:<6} {r['Credit']:<8.2f} {r['IV']:<6.1f} {r['VRP']:<6.2f} {r['CVaR']:<8.2f} {r['Ratio']:<8.3f} {r['PoP']:<6.1f}")
        else:
            print(f"{Fore.YELLOW}No trades found (VRP < 1.0 or Credit < 0.01).")

        print(f"\nScanning again in 30 seconds...")
        await asyncio.sleep(30)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")