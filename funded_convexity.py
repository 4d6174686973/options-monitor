import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Funded Convexity Simulator", layout="wide")

st.title("🛡️ Funded Convexity: 'House Money' Strategy")
st.markdown("""
This model simulates a strategy where **Hunter bets (Convexity)** are funded **exclusively by Farmer profits**.
* **Principal Protection:** We try not to touch the initial capital for speculation.
* **The Flywheel:** Farmer profits $\rightarrow$ Buy Moonshots $\rightarrow$ If hit, Principal grows $\rightarrow$ Farmer bets get bigger.
""")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("1. Setup")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000)
weeks = st.sidebar.slider("Duration (Weeks)", 4, 24, 12)
n_simulations = st.sidebar.slider("Monte Carlo Runs", 100, 2000, 500)

st.sidebar.markdown("---")
st.sidebar.header("2. The Farmer (The Engine)")
# Farmer now uses a percentage of TOTAL equity, because it's the core strategy
farmer_utilization = st.sidebar.slider("Capital Utilization for Farmer (%)", 10, 100, 80) / 100.0
farmer_win_rate = st.sidebar.slider("Farmer Win Rate (%)", 50, 95, 80) / 100.0
farmer_roi = st.sidebar.slider("Farmer Yield per Week (%)", 0.5, 5.0, 2.0) / 100.0
farmer_loss_pct = st.sidebar.slider("Max Loss on Farmer Risk (%)", 10, 100, 100) / 100.0

st.sidebar.markdown("---")
st.sidebar.header("3. The Hunter (The Sweep)")
st.sidebar.info("Hunter bets are only placed if there is 'House Money' available.")
# Strategy: How much of the profits do we burn?
profit_reinvest_rate = st.sidebar.slider("% of Profits Swept to Hunter", 0, 100, 100) / 100.0
hunter_win_rate = st.sidebar.slider("Hunter Win Rate (%)", 1, 30, 10) / 100.0
hunter_payout_mult = st.sidebar.slider("Hunter Payout Multiple (x)", 2.0, 50.0, 15.0)

def run_funded_simulation():
    # Store equity paths
    equity_paths = np.zeros((n_simulations, weeks + 1))
    equity_paths[:, 0] = initial_capital
    
    # Store how much we bet on hunters (to visualize aggressive capacity)
    hunter_wagers = np.zeros((n_simulations, weeks))
    
    # Pre-generate random variables
    farmer_rands = np.random.rand(n_simulations, weeks)
    hunter_rands = np.random.rand(n_simulations, weeks)
    
    for t in range(1, weeks + 1):
        prev_equity = equity_paths[:, t-1]
        
        # 1. FARMER STEP
        # Calculate Farmer Risk Capital
        farmer_bet_size = prev_equity * farmer_utilization
        
        # Determine Farmer Outcome
        # Win: +ROI | Loss: -Loss_Pct * (Risk/Reward Ratio adjustment)
        # We assume 1:1 risk/reward for simplicity unless tailored, but usually credit spreads risk $1 to make $0.20
        # Let's model standard spread risk: Risk 2.0 to make 1.0 is common, so loss multiplier = 2.0/1.0 * ROI? 
        # Let's keep it simple: Loss is a % of the collateral engaged.
        farmer_pnl = np.where(
            farmer_rands[:, t-1] < farmer_win_rate,
            farmer_bet_size * farmer_roi,
            -farmer_bet_size * (farmer_roi * 2.5) # Penalty: losing a spread hurts more than winning helps
        )
        
        # 2. HUNTER STEP (The Sweep)
        # Logic: We check if we have a "Surplus" from this week (or generally).
        # Strategy: If Farmer Won -> Take (Profit * Reinvest%) and bet it.
        # If Farmer Lost -> Hunter Bet = 0 (Defense mode).
        
        available_for_hunter = np.maximum(0, farmer_pnl) * profit_reinvest_rate
        hunter_wagers[:, t-1] = available_for_hunter
        
        # Hunter Outcome
        # If Win: Payoff - Cost. If Loss: -Cost.
        hunter_pnl = np.where(
            hunter_rands[:, t-1] < hunter_win_rate,
            (available_for_hunter * hunter_payout_mult) - available_for_hunter,
            -available_for_hunter
        )
        
        # 3. UPDATE EQUITY
        # New Equity = Old + Farmer PnL (net of sweep) + Hunter PnL
        # Note: If we sweep into hunter, that money is technically "spent" until the hunter resolves.
        # But here they resolve same week.
        
        # If we swept the profit, it is removed from Farmer PnL and moved to Hunter PnL logic
        # Net Farmer PnL after sweep = Farmer_PnL - Available_For_Hunter
        # Total PnL = (Farmer_PnL - Hunter_Bet) + (Hunter_Payout)
        
        net_change = (farmer_pnl - available_for_hunter) + (available_for_hunter + hunter_pnl)
        
        equity_paths[:, t] = prev_equity + net_change
        
        # Floor at 0
        equity_paths[:, t] = np.maximum(equity_paths[:, t], 0)
        
    return equity_paths, hunter_wagers

paths, wagers = run_funded_simulation()

# --- METRICS ---
final_equity = paths[:, -1]
avg_hunter_bet = np.mean(wagers)
prob_survival = np.mean(final_equity > initial_capital * 0.9) * 100 # Survival > 90% capital

col1, col2, col3 = st.columns(3)
col1.metric("Median Final Equity", f"${np.median(final_equity):,.0f}")
col2.metric("Avg Weekly Hunter Bet", f"${avg_hunter_bet:,.0f}")
col3.metric("Survival Rate (>90% Cap)", f"{prob_survival:.1f}%")

# --- VISUALIZATION ---
st.subheader("Scenario Analysis")
tab1, tab2 = st.tabs(["Equity Curves", "The 'Free' Upside"])

with tab1:
    # Plot first 100 paths
    df_paths = pd.DataFrame(paths[:100].T)
    fig = px.line(df_paths, title="Equity Curves (Sample of 100)", labels={"index": "Week", "value": "Equity"})
    fig.update_layout(showlegend=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Histogram
    fig_hist = px.histogram(final_equity, title="Final Wealth Distribution", nbins=50, template="plotly_dark")
    fig_hist.add_vline(x=initial_capital, line_dash="dash", line_color="red", annotation_text="Principal")
    st.plotly_chart(fig_hist, use_container_width=True)

st.info("""
**Interpretation:** Notice the 'Wall' at the left of the distribution near your Initial Capital. 
Because we only bet profits, it is much harder to lose principal compared to the previous model. 
However, the right tail (the big wins) builds up slower because the Hunter bets start small and only grow as you win.
""")