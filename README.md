# Poker Bankroll Simulator

An interactive tool built with Streamlit to simulate poker bankroll progression based on user-defined performance data and bankroll management strategies.

This application allows users to:
- Model win rates with a sophisticated Bayesian-inspired approach that accounts for sample size and long-term luck.
- Input detailed performance stats (EV Win Rate, Std. Dev, etc.) for various poker stakes.
- Define and compare multiple complex bankroll management strategies, including standard threshold-based rules and "sticky" hysteresis rules.
- Configure advanced gameplay options like session stop-losses and periodic bankroll withdrawals.
- Run thousands of high-speed Monte Carlo simulations to analyze long-term outcomes.
- Validate the engine's accuracy with a built-in "Sanity Check" mode.
- Analyze detailed downswing probabilities, showing the likelihood of experiencing downswings of a certain depth (in big blinds) or duration (in hands).
- Generate and download comprehensive, multi-page PDF reports with detailed visualizations and analysis.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Streamlit app:**
   ```bash
   streamlit run simulator_app.py
   ```