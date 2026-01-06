import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Barbell Strategy Simulator", layout="wide")

# --- HEADER ---
st.title("🏹 Hunter & Farmer: Barbell Strategy Monte Carlo")
st.markdown("""
This dashboard simulates a **Barbell Strategy** over a fixed timeframe. 
It helps verify if the **Convexity** of the Hunter bets can overcome the steady risk of the Farmer bets.
""")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("1. Portfolio Settings")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000, step=10000)
weeks = st.sidebar.slider("Duration (Weeks)", min_value=4, max_value=52, value=12)
n_simulations = st.sidebar.slider("Monte Carlo Runs", min_value=100, max_value=2000, value=500)

st.sidebar.markdown("---")
st.sidebar.header("2. The Farmer (Income)")
st.sidebar.info("High probability, limited payout (e.g., Credit Spreads).")
farmer_alloc = 0.60
farmer_win_rate = st.sidebar.slider("Farmer Win Rate (%)", 60, 95, 80) / 100.0
farmer_roi = st.sidebar.slider("Farmer ROI per Week (%)", 1.0, 10.0, 2.0) / 100.0
farmer_loss_pct = st.sidebar.slider("Max Loss on Farmer Risk (%)", 50, 100, 100) / 100.0
# Note: In credit spreads, if you lose, you often lose 1x-3x the premium collected.
# Here we model it as losing a percentage of the allocated capital.

st.sidebar.markdown("---")
st.sidebar.header("3. The Hunter (Convexity)")
st.sidebar.info("Low probability, massive upside (e.g., Long Volatility).")
hunter_alloc = 0.20
hunter_win_rate = st.sidebar.slider("Hunter Win Rate (%)", 1, 30, 10) / 100.0
hunter_payout_mult = st.sidebar.slider("Hunter Payout Multiple (x)", 2.0, 50.0, 10.0)
# A payout multiple of 10.0 means if you bet $1k, you get $10k back.

# Cash is the remainder
cash_alloc = 1.0 - farmer_alloc - hunter_alloc

# --- SIMULATION ENGINE ---
def run_simulation():
    # Arrays to store results
    all_paths = np.zeros((n_simulations, weeks + 1))
    all_paths[:, 0] = initial_capital
    
    # Random generation (Vectorized for speed)
    # Generate random floats [0,1] for all weeks and sims
    farmer_rands = np.random.rand(n_simulations, weeks)
    hunter_rands = np.random.rand(n_simulations, weeks)
    
    for t in range(1, weeks + 1):
        # Current Equity from previous step
        current_equity = all_paths[:, t-1]
        
        # Allocations
        farmer_bet = current_equity * farmer_alloc
        hunter_bet = current_equity * hunter_alloc
        cash = current_equity * cash_alloc
        
        # Farmer Outcomes
        # If rand < win_rate: Gain ROI * Bet. Else: Lose Loss_Pct * Bet
        farmer_pnl = np.where(
            farmer_rands[:, t-1] < farmer_win_rate,
            farmer_bet * farmer_roi,
            -farmer_bet * farmer_loss_pct * (farmer_roi * 2) # Heuristic: Losses on spreads are usually > gains
        )
        
        # Hunter Outcomes
        # If rand < win_rate: Gain (Multiple * Bet) - Bet. Else: Lose Bet (Premium paid)
        hunter_pnl = np.where(
            hunter_rands[:, t-1] < hunter_win_rate,
            (hunter_bet * hunter_payout_mult) - hunter_bet, 
            -hunter_bet
        )
        
        # Update Equity
        all_paths[:, t] = current_equity + farmer_pnl + hunter_pnl
        
        # Bankruptcy Check (Floor at 0)
        all_paths[:, t] = np.maximum(all_paths[:, t], 0)
        
    return all_paths

results = run_simulation()

# --- STATISTICS ---
final_values = results[:, -1]
median_result = np.median(final_values)
mean_result = np.mean(final_values)
max_result = np.max(final_values)
min_result = np.min(final_values)
prob_profit = np.mean(final_values > initial_capital) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Median Equity", f"${median_result:,.0f}")
col2.metric("Mean Equity (Convexity)", f"${mean_result:,.0f}", delta_color="normal")
col3.metric("Max Upside (Outlier)", f"${max_result:,.0f}")
col4.metric("Prob. of Profit", f"{prob_profit:.1f}%")

# --- VISUALIZATION 1: EQUITY CURVES ---
st.subheader("Simulated Equity Curves")
# Downsample for charting if too many simulations
display_indices = np.random.choice(n_simulations, size=min(100, n_simulations), replace=False)
df_paths = pd.DataFrame(results[display_indices, :].T)
df_paths.columns = [f"Sim {i}" for i in display_indices]
df_paths['Week'] = range(weeks + 1)

fig_lines = px.line(df_paths, x='Week', y=df_paths.columns[:-1], 
                    title="Monte Carlo Paths (Sample of 100)", labels={'value': 'Equity', 'variable': 'Path'})
fig_lines.update_layout(showlegend=False, template="plotly_dark")
st.plotly_chart(fig_lines, use_container_width=True)

# --- VISUALIZATION 2: DISTRIBUTION ---
st.subheader("Terminal Wealth Distribution (The Shape of Convexity)")
fig_hist = px.histogram(final_values, nbins=50, title="Distribution of Final Equity", 
                        labels={'value': 'Final Equity'}, template="plotly_dark")
fig_hist.add_vline(x=initial_capital, line_dash="dash", line_color="red", annotation_text="Break Even")
fig_hist.add_vline(x=mean_result, line_dash="dash", line_color="green", annotation_text="Mean")
st.plotly_chart(fig_hist, use_container_width=True)

# --- INSIGHTS ---
st.markdown("""
### 🧠 Manager's Analysis
1.  **Look at the Mean vs. Median:** If the **Mean** is significantly higher than the **Median**, your strategy has **Positive Skew**. This is the definition of convexity. You rely on those few "Hunter" wins to drag the average up, even if most simulations feel "grindy."
2.  **The "Ruin" Tail:** Look at the left side of the histogram. If too many paths hit 0, your sizing (60/20) is too aggressive for the volatility of the Hunter bets.
3.  **Competition Mode:** To win a competition, you generally need to be in the top 5% of the distribution (the far right tail). Adjust the **Hunter Payout Multiple** and **Allocation** to see what it takes to get that tail to extend.
""")