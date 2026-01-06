import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Competition Alpha Dashboard", layout="wide")

# --- 1. GLOBAL MARKET DATA (SIDEBAR) ---
st.sidebar.header("🌍 Market Data Feed")
st.sidebar.caption("Update current underlying prices here to drive all calculations.")

if 'market_data' not in st.session_state:
    st.session_state.market_data = {"NVDA": 115.00, "TSLA": 240.00, "SPY": 580.00}

# Dynamic Input for Market Prices
tickers_to_edit = list(st.session_state.market_data.keys())
new_ticker = st.sidebar.text_input("Add Ticker (e.g., AMD)", "").upper()
if st.sidebar.button("Add"):
    if new_ticker and new_ticker not in st.session_state.market_data:
        st.session_state.market_data[new_ticker] = 100.00

# Display Editable Price Inputs
for t in tickers_to_edit:
    val = st.sidebar.number_input(f"{t} Price", value=float(st.session_state.market_data[t]), step=0.50, key=f"price_{t}")
    st.session_state.market_data[t] = val

st.sidebar.markdown("---")
portfolio_value = st.sidebar.number_input("💰 Total Portfolio Value ($)", value=10000.0)

# --- TABS FOR WORKFLOW ---
tab1, tab2, tab3 = st.tabs(["🏗️ Trade Builder", "📏 Kelly Sizer", "📊 Active Monitor"])

# ==============================================================================
# TAB 1: OPTIONS TRADE BUILDER (Straddle/Spread Calculator)
# ==============================================================================
with tab1:
    st.header("Strategy Constructor")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_ticker = st.selectbox("Select Underlying", list(st.session_state.market_data.keys()))
        current_ul_price = st.session_state.market_data[selected_ticker]
        st.info(f"Current {selected_ticker} Price: ${current_ul_price:.2f}")
        
        strat_type = st.selectbox("Strategy Type", ["Long Straddle", "Credit Spread (Bull Put)", "Credit Spread (Bear Call)"])
        
        legs = []
        if strat_type == "Long Straddle":
            strike = st.number_input("ATM Strike", value=current_ul_price)
            call_price = st.number_input("Call Option Price ($)", value=2.50)
            put_price = st.number_input("Put Option Price ($)", value=2.50)
            total_cost = (call_price + put_price) * 100
            max_risk = total_cost
            max_return = "Unlimited"
            breakeven_upper = strike + (total_cost/100)
            breakeven_lower = strike - (total_cost/100)
            
        elif "Credit Spread" in strat_type:
            is_bull = "Bull" in strat_type
            # Bull Put: Sell Low Strike, Buy Lower Strike
            # Bear Call: Sell High Strike, Buy Higher Strike
            
            st.subheader("Short Leg (Sell)")
            short_strike = st.number_input("Short Strike", value=current_ul_price)
            short_price = st.number_input("Short Price ($)", value=1.50)
            
            st.subheader("Long Leg (Buy Protection)")
            long_strike = st.number_input("Long Strike", value=current_ul_price - 5 if is_bull else current_ul_price + 5)
            long_price = st.number_input("Long Price ($)", value=0.50)
            
            net_credit = (short_price - long_price) * 100
            width = abs(short_strike - long_strike) * 100
            max_risk = width - net_credit
            max_return = net_credit
            
            breakeven = (short_strike - (net_credit/100)) if is_bull else (short_strike + (net_credit/100))

    with col2:
        st.subheader("Risk/Reward Profile")
        
        # Metrics Display
        m1, m2, m3 = st.columns(3)
        m1.metric("Max Risk (Margin)", f"${max_risk:.2f}")
        
        # Handle unlimited returns in metric
        if max_return == "Unlimited":
            m2.metric("Max Return", "∞")
            ror = "N/A"
        else:
            m2.metric("Max Return", f"${max_return:.2f}")
            ror = f"{(max_return / max_risk) * 100:.1f}%"
        
        m3.metric("Return on Risk", ror)
        
        # Visualizer (Payoff Diagram)
        x_range = np.linspace(current_ul_price * 0.8, current_ul_price * 1.2, 100)
        y_pnl = []
        
        for x in x_range:
            if strat_type == "Long Straddle":
                val = (max(0, x - strike) + max(0, strike - x)) * 100 - total_cost
            elif "Credit Spread" in strat_type:
                # Basic Spread Logic
                if is_bull: # Bull Put
                    val_short = -max(0, short_strike - x) + short_price
                    val_long = max(0, long_strike - x) - long_price
                else: # Bear Call
                    val_short = -max(0, x - short_strike) + short_price
                    val_long = max(0, x - long_strike) - long_price
                val = (val_short + val_long) * 100
            y_pnl.append(val)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=y_pnl, mode='lines', name='PnL at Expiration', fill='tozeroy'))
        fig.add_vline(x=current_ul_price, line_dash="dash", annotation_text="Current Price")
        fig.update_layout(title="PnL Diagram", xaxis_title="Stock Price", yaxis_title="Profit/Loss ($)")
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 2: KELLY BET SIZING (Probabilistic Version)
# ==============================================================================
with tab2:
    st.header("🎲 Kelly Sizer (Statistical Boundaries)")
    
    k_type = st.radio("Bet Type", ["Fixed Payout (Credit Spread)", "Probabilistic (Straddle/Lotto)"])
    
    col_k1, col_k2 = st.columns(2)
    
    with col_k1:
        p = st.slider("Win Probability (%)", 5, 95, 30 if k_type == "Probabilistic" else 70) / 100
        cost_of_entry = st.number_input("Cost of Trade (Premium Paid/Margin)", value=500.0)
        
        if k_type == "Probabilistic":
            st.subheader("Statistical Upside")
            avg_move_pct = st.slider("Typical 'Big' Move (%)", 1.0, 30.0, 10.0)
            # Estimate: A 10% move on a 5% cost straddle is roughly a 100% gain
            est_payout_pct = st.number_input("Estimated Payout on Win (%)", value=150.0)
            potential_gain = cost_of_entry * (est_payout_pct / 100)
        else:
            potential_gain = st.number_input("Fixed Net Profit ($)", value=200.0)
            
        potential_loss = cost_of_entry

    with col_k2:
        if potential_loss > 0:
            b = potential_gain / potential_loss
            f_star = p - ((1 - p) / b)
            
            st.metric("Profit Multiplier (b)", f"{b:.2f}x")
            
            if f_star <= 0:
                st.error("⚠️ Mathematical Edge is Negative. Re-evaluate your entry or probability.")
            else:
                st.success(f"Optimal Allocation: {f_star*100:.1f}%")
                # Show Half-Kelly for the competition (more robust)
                half_k_amt = (f_star * 0.5) * portfolio_value
                st.metric("Half-Kelly Stake ($)", f"${half_k_amt:,.2f}")
                
                st.info(f"""
                **Strategy Logic:**
                To justify a {p*100:.0f}% win rate, you expect the winner to pay 
                out ${potential_gain:,.2f} for every ${potential_loss:,.2f} risked.
                """)

                
# ==============================================================================
# TAB 3: CURRENT TRADES MONITOR (Updated)
# ==============================================================================
with tab3:
    st.header("📡 Live Trade Monitor")
    
    # Session State for Trades
    if 'active_trades' not in st.session_state:
        st.session_state.active_trades = []

    # --- INPUT FORM ---
    with st.expander("➕ Add New Position from Builder", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        t_ticker = c1.selectbox("Ticker", list(st.session_state.market_data.keys()), key="monitor_ticker")
        t_strat = c2.selectbox("Strategy", ["Credit Spread", "Straddle", "Long Call/Put"], key="monitor_strat")
        t_entry = c3.number_input("Net Entry Price", value=0.0)
        t_size = c4.number_input("Position Size ($ Risk)", value=0.0)
        
        if c5.button("Log Trade"):
            st.session_state.active_trades.append({
                "Ticker": t_ticker,
                "Strategy": t_strat,
                "Entry": t_entry,
                "Current Value": t_entry, # Initialize Current Value same as Entry
                "Size": t_size,
            })
            st.rerun() # Force rerun to show new trade immediately

    # --- MONITOR TABLE ---
    if st.session_state.active_trades:
        # Convert list to DataFrame
        df_trades = pd.DataFrame(st.session_state.active_trades)
        
        # Ensure "Current Value" exists (for legacy records)
        if "Current Value" not in df_trades.columns:
            df_trades["Current Value"] = df_trades["Entry"]

        # --- CALCULATIONS (Run BEFORE displaying) ---
        # We calculate PnL and Alerts here so we can show them in the same table
        pnl_values = []
        alerts = []
        
        for index, row in df_trades.iterrows():
            entry = row['Entry']
            curr = row['Current Value']
            strat = row['Strategy']
            
            # PnL Logic
            if "Credit" in strat:
                # Credit Spread: Win if Price Goes Down (Current < Entry)
                # PnL = (Entry - Current)
                raw_pnl = (entry - curr) * 100 # Approx $ per contract
                pnl_pct = ((entry - curr) / entry) * 100 if entry != 0 else 0
                
                # Alerts
                if curr >= (entry * 2):
                    alerts.append("🛑 STOP (2x Loss)")
                elif pnl_pct >= 50:
                    alerts.append("💰 TAKE PROFIT")
                else:
                    alerts.append("Hold")
                    
            else:
                # Debit Trade (Straddle/Long): Win if Price Goes Up (Current > Entry)
                raw_pnl = (curr - entry) * 100
                pnl_pct = ((curr - entry) / entry) * 100 if entry != 0 else 0
                
                # Alerts
                if curr <= (entry * 0.5):
                    alerts.append("🛑 STOP (-50%)")
                elif pnl_pct >= 100:
                    alerts.append("🚀 MOONBAG")
                else:
                    alerts.append("Hold")
            
            pnl_values.append(f"{pnl_pct:.1f}%")

        # Add calculated columns to DF for display
        df_trades["PnL %"] = pnl_values
        df_trades["Action"] = alerts

        st.markdown("### Active Positions")
        st.caption("📝 Edit 'Current Value' to update PnL. Select rows and press 'Delete' key to remove.")
        
        # Display the Editor
        # column_config allows us to make PnL and Action "Read Only" while Entry/Current are editable
        edited_df = st.data_editor(
            df_trades,
            num_rows="dynamic", # THIS ALLOWS DELETION
            key="trade_editor",
            use_container_width=True,
            column_config={
                "Ticker": st.column_config.TextColumn(disabled=True),
                "Strategy": st.column_config.TextColumn(disabled=True),
                "Entry": st.column_config.NumberColumn(disabled=True),
                "Size": st.column_config.NumberColumn(disabled=True),
                "PnL %": st.column_config.TextColumn(disabled=True),
                "Action": st.column_config.TextColumn(disabled=True),
                "Current Value": st.column_config.NumberColumn(
                    "Current Value",
                    help="Update this manually based on broker price",
                    min_value=0.0,
                    step=0.05
                )
            }
        )

        # --- SAVE CHANGES (PERSISTENCE) ---
        # Crucial Step: We sync the edited dataframe back to session state.
        # We only save the "Core" columns, dropping the calculated PnL/Action columns
        # so they don't get stored as static text.
        cols_to_save = ["Ticker", "Strategy", "Entry", "Current Value", "Size"]
        
        # Check if changes happened to avoid unnecessary reruns
        new_state = edited_df[cols_to_save].to_dict('records')
        
        if new_state != st.session_state.active_trades:
            st.session_state.active_trades = new_state
            st.rerun() # Rerun immediately to update the PnL calculations based on new inputs

    else:
        st.info("No active trades. Use the form above to add one.")