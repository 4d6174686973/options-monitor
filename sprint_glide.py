import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sprint & Glide Strategy", layout="wide")

st.title("🚀 Sprint & Glide: Regime Switching Strategy")
st.markdown("""
**The Competition Winner's Logic:**
1.  **Phase 1 (Sprint):** Take aggressive risks (dip into principal) to catch an early lead.
2.  **Phase 2 (Glide):** Once **Target Equity** is reached, switch to defensive mode (protect the lead).
""")

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.header("1. Competition Setup")
initial_capital = st.sidebar.number_input("Starting Capital", value=100000, step=10000)
weeks = st.sidebar.slider("Total Duration (Weeks)", 4, 24, 12)
n_sims = st.sidebar.slider("Simulations", 100, 3000, 1000)

# --- SIDEBAR: THE SWITCH ---
st.sidebar.markdown("---")
st.sidebar.header("2. The Trigger")
target_multiplier = st.sidebar.slider("Target Return to Switch (%)", 10, 200, 50) / 100.0
target_equity = initial_capital * (1 + target_multiplier)
st.sidebar.metric("Target Equity", f"${target_equity:,.0f}")

# --- SIDEBAR: REGIME A (SPRINT) ---
st.sidebar.markdown("---")
st.sidebar.header("3. Phase 1: The Sprint (Aggressive)")
sprint_farmer_alloc = st.sidebar.slider("[Sprint] Farmer Alloc (%)", 0, 100, 40) / 100.0
sprint_hunter_alloc = st.sidebar.slider("[Sprint] Hunter Alloc (%)", 0, 100, 30) / 100.0
# Remainder is cash

# --- SIDEBAR: REGIME B (GLIDE) ---
st.sidebar.markdown("---")
st.sidebar.header("4. Phase 2: The Glide (Defensive)")
glide_farmer_alloc = st.sidebar.slider("[Glide] Farmer Alloc (%)", 0, 100, 80) / 100.0
glide_hunter_alloc = st.sidebar.slider("[Glide] Hunter Alloc (%)", 0, 50, 5) / 100.0
# Usually Glide has very low hunter alloc, or funded only.

# --- SIDEBAR: MARKET ASSUMPTIONS ---
st.sidebar.markdown("---")
st.sidebar.header("Market Assumptions")
farmer_roi = st.sidebar.number_input("Farmer Weekly ROI (%)", value=2.0) / 100.0
farmer_win = st.sidebar.slider("Farmer Win Rate (%)", 50, 99, 85) / 100.0
hunter_mult = st.sidebar.number_input("Hunter Payout (x)", value=12.0)
hunter_win = st.sidebar.slider("Hunter Win Rate (%)", 1, 40, 12) / 100.0


def run_regime_sim():
    # Paths: [Simulations, Weeks]
    equity = np.zeros((n_sims, weeks + 1))
    equity[:, 0] = initial_capital
    
    # Track which regime we are in (0 = Sprint, 1 = Glide)
    regime_tracker = np.zeros((n_sims, weeks + 1))
    
    # Pre-calc randoms
    rand_farm = np.random.rand(n_sims, weeks)
    rand_hunt = np.random.rand(n_sims, weeks)
    
    for t in range(1, weeks + 1):
        prev = equity[:, t-1]
        
        # Determine Regime for EACH simulation individually
        # If prev equity >= target, use Glide params. Else Sprint.
        is_glide = prev >= target_equity
        regime_tracker[:, t] = is_glide * 1.0
        
        # Set allocations based on regime vectors
        # Using np.where to vectorize the choice
        curr_farm_alloc = np.where(is_glide, glide_farmer_alloc, sprint_farmer_alloc)
        curr_hunt_alloc = np.where(is_glide, glide_hunter_alloc, sprint_hunter_alloc)
        
        # Bet Sizes
        bet_farm = prev * curr_farm_alloc
        bet_hunt = prev * curr_hunt_alloc
        
        # Calculate PnL
        # Farmer
        pnl_farm = np.where(rand_farm[:, t-1] < farmer_win,
                            bet_farm * farmer_roi,
                            -bet_farm * (farmer_roi * 2.0)) # Loss penalty
        
        # Hunter
        pnl_hunt = np.where(rand_hunt[:, t-1] < hunter_win,
                            (bet_hunt * hunter_mult) - bet_hunt,
                            -bet_hunt)
        
        # Update
        equity[:, t] = prev + pnl_farm + pnl_hunt
        equity[:, t] = np.maximum(equity[:, t], 0) # Ruin floor
        
    return equity, regime_tracker

paths, regimes = run_regime_sim()
final_vals = paths[:, -1]

# --- METRICS & ANALYSIS ---
# "Winners" are those who hit target at least once (or end above it)
# Let's define Success as Ending above Target
success_rate = np.mean(final_vals >= target_equity) * 100
ruin_rate = np.mean(final_vals < initial_capital * 0.2) * 100 # < 20% remaining
median_end = np.median(final_vals)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏆 Success Rate (Hit Target)", f"{success_rate:.1f}%")
col2.metric("💀 Ruin Rate (<20% Cap)", f"{ruin_rate:.1f}%")
col3.metric("Median Ending Equity", f"${median_end:,.0f}")
col4.metric("Avg Weeks in 'Sprint'", f"{weeks - np.mean(np.sum(regimes, axis=1)):.1f}")

# --- VISUALIZATIONS ---
tab1, tab2 = st.tabs(["Paths & Regimes", "Success vs Ruin"])

with tab1:
    st.subheader("Simulated Paths (Color-coded by Final Outcome)")
    # Sample 100 lines
    indices = np.random.choice(n_sims, min(100, n_sims), replace=False)
    sample_paths = paths[indices]
    
    # Create DF for Plotly
    df_plot = pd.DataFrame(sample_paths.T)
    
    fig = go.Figure()
    for i in range(df_plot.shape[1]):
        # Color logic: Green if ends above target, Red if ruin, Grey otherwise
        end_val = sample_paths[i, -1]
        color = 'gray'
        opacity = 0.3
        if end_val >= target_equity:
            color = '#00CC96' # Green
            opacity = 0.8
        elif end_val < initial_capital * 0.5:
            color = '#EF553B' # Red
            opacity = 0.5
            
        fig.add_trace(go.Scatter(y=sample_paths[i], mode='lines', 
                                 line=dict(color=color, width=1), opacity=opacity,
                                 name=f"Sim {i}", showlegend=False))
        
    fig.add_hline(y=target_equity, line_dash="dash", line_color="yellow", annotation_text="Target (Switch Trigger)")
    fig.add_hline(y=initial_capital, line_dash="dot", line_color="white", annotation_text="Start")
    fig.update_layout(title="Monte Carlo Paths (Green = Won, Red = Lost)", template="plotly_dark",
                      xaxis_title="Week", yaxis_title="Equity")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Risk/Reward Trade-off")
    fig_hist = px.histogram(final_vals, nbins=60, title="Final Distribution", template="plotly_dark")
    fig_hist.add_vline(x=target_equity, line_color="yellow", annotation_text="Goal")
    fig_hist.add_vline(x=initial_capital, line_color="white", annotation_text="Start")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("""
    **How to read this:**
    * **The Bi-Modal Distribution:** You will likely see two "humps". 
        * **Left Hump:** The simulations that failed the sprint and bled out (The price of admission).
        * **Right Hump:** The simulations that hit the target, switched to "Glide", and locked in the win.
    * **The Sweet Spot:** You want to maximize the size of the Right Hump while keeping the Left Hump from hitting absolute zero (ruin).
    """)