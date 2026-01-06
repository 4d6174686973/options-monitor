import streamlit as st
import pandas as pd

st.set_page_config(page_title="Competition Monitor", layout="wide")

# --- CUSTOM CSS FOR STOP-LOSS FLAGS ---
st.markdown("""
    <style>
    .stop-hit { background-color: #ff4b4b; color: white; border-radius: 5px; padding: 2px 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🦅 Competition Alpha Monitor")

# --- SIDEBAR: POSITION SIZING ---
with st.sidebar:
    st.header("📏 Kelly Size Calculator")
    port_value = st.number_input("Portfolio Value ($)", value=10000)
    prob_win = st.slider("Estimated Win Prob (%)", 10, 95, 70) / 100
    
    col_a, col_b = st.columns(2)
    win_amt = col_a.number_input("If Win ($)", value=300)
    loss_amt = col_b.number_input("If Loss ($)", value=700)
    
    if loss_amt > 0:
        b = win_amt / loss_amt
        q = 1 - prob_win
        kelly_f = prob_win - (q / b)
        st.success(f"Full Kelly: {max(0, kelly_f*100):.1f}%")
        st.info(f"Half-Kelly (Safe): {max(0, kelly_f*50):.1f}%")

# --- DATA STORAGE ---
if 'trades' not in st.session_state:
    st.session_state.trades = []

# --- MANUAL ENTRY FORM ---
with st.expander("➕ Log New Position", expanded=True):
    with st.form("trade_form"):
        c1, c2, c3, c4 = st.columns(4)
        ticker = c1.text_input("Ticker")
        strategy = c2.selectbox("Type", ["Credit Spread", "Straddle", "OTM Lotto"])
        entry_price = c3.number_input("Entry (Net)", format="%.2f")
        current_price = c4.number_input("Current (Net)", format="%.2f")
        
        if st.form_submit_button("Add Trade"):
            st.session_state.trades.append({
                "Ticker": ticker.upper(),
                "Strategy": strategy,
                "Entry": entry_price,
                "Current": current_price,
            })

# --- MONITORING TABLE ---
if st.session_state.trades:
    st.header("📊 Active Inventory")
    
    data = []
    for t in st.session_state.trades:
        # PnL Logic: Credit Spreads profit when Current < Entry
        if t['Strategy'] == "Credit Spread":
            pnl = (t['Entry'] - t['Current']) * 100
            # Stop loss flag: if current price > 2x entry price
            status = "🚨 STOP HIT" if t['Current'] >= (t['Entry'] * 2) else "OK"
        else:
            pnl = (t['Current'] - t['Entry']) * 100
            # Straddle stop loss: if current price < 0.5x entry price
            status = "🚨 STOP HIT" if t['Current'] <= (t['Entry'] * 0.5) else "OK"
            
        data.append({**t, "PnL": pnl, "Status": status})

    df = pd.DataFrame(data)

    # Styling the status column
    def color_status(val):
        color = 'red' if 'STOP' in val else 'green'
        return f'color: {color}; font-weight: bold'

    st.dataframe(df.style.applymap(color_status, subset=['Status']), use_container_width=True)
    
    # Competition Total
    total_pnl = df['PnL'].sum()
    st.metric("Total Competition Gain", f"${total_pnl:,.2f}", 
              delta=f"{(total_pnl/port_value)*100:.2f}%")