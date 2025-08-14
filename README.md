# Poker Bankroll Simulator

An interactive tool built with Streamlit to simulate poker bankroll progression based on user-defined performance data and bankroll management strategies.

This application allows users to:
- Input their win rates, standard deviations, and other stats for various poker stakes.
- Define and compare multiple complex bankroll management strategies, including standard threshold-based rules and "sticky" hysteresis rules.
- Run thousands of Monte Carlo simulations to analyze long-term outcomes.
- View detailed reports and visualizations, including risk of ruin, bankroll growth, and final stake distributions.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Streamlit app:**
   ```bash
   streamlit run simulator_app.py
   ```