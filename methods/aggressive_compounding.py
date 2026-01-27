import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup Tournament Parameters
initial_capital = 10000
weeks = 6
win_rate = 0.70 # Strong bull market assumption
payout = 0.30   # 30% Return on Risk (Credit received / Max Risk)

# Strategy A: Conservative (Fixed Risk)
# Risking $500 per trade (5% of starting balance)
fixed_risk_amt = 500 

# Strategy B: Tournament (Compounding Risk)
# Risking 25% of CURRENT account balance every week
compounding_risk_pct = 0.25

# 2. Simulate Trajectory (A "Lucky Streak" scenario)
# Assuming we hit a streak of wins which is the goal of tournament play
outcomes = ['Win'] * weeks 

equity_conservative = [initial_capital]
equity_tournament = [initial_capital]

for i in range(weeks):
    # Conservative Update
    profit_cons = fixed_risk_amt * payout
    equity_conservative.append(equity_conservative[-1] + profit_cons)
    
    # Tournament Update
    risk_amt = equity_tournament[-1] * compounding_risk_pct
    profit_tourn = risk_amt * payout
    equity_tournament.append(equity_tournament[-1] + profit_tourn)

# 3. Visualization
plt.figure(figsize=(10, 6))
plt.plot(equity_conservative, marker='o', label='Conservative (Fixed Risk $500)')
plt.plot(equity_tournament, marker='s', label='Tournament (Risk 25% of Equity)', color='orange')

plt.title('The "Catch-Up" Effect: Compounding vs Fixed Risk')
plt.xlabel('Week')
plt.ylabel('Account Balance ($)')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

total_return_cons = (equity_conservative[-1] - initial_capital) / initial_capital
total_return_tourn = (equity_tournament[-1] - initial_capital) / initial_capital

print(f"Conservative Total Return: {total_return_cons:.1%}")
print(f"Tournament Total Return:   {total_return_tourn:.1%}")