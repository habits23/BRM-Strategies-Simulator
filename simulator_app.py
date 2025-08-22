"""
This script creates a comprehensive Streamlit web application for simulating poker bankroll progression.

It allows users to:
- Define their own win rates, standard deviations, and other statistics for various poker stakes.
- Create and compare multiple bankroll management (BRM) strategies, including standard threshold-based rules
  and a "sticky" hysteresis model.
- Configure global simulation parameters like starting bankroll, simulation length, and rakeback.
- Run thousands of simulations to generate robust statistical data on the performance of each strategy.
- View detailed results through interactive tables, plots, and an automated qualitative analysis.
- Save and load their entire configuration to a JSON file for later use.

The application is structured to separate the UI (this file) from the core simulation logic (simulation_engine.py).
It heavily utilizes Streamlit's session state to provide a seamless, interactive user experience.
"""
import streamlit as st
# --- Core Libraries ---
import pandas as pd
import datetime
import traceback
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
# --- Custom Simulation Engine ---
# This imports the file containing all the core Monte Carlo simulation logic,
# statistical analysis, and plotting functions.
import simulation_engine as engine

# --- Downswing Analysis Configuration ---
# These constants define the thresholds for analyzing downswing severity and duration.
# They are passed to the simulation engine to generate specific probability tables.
# These are the thresholds used to generate the "Downswing Extents" and "Downswing Stretches" analysis,
# inspired by Prime Dope's variance calculator.
DOWNSWING_DEPTH_THRESHOLDS_BB = [300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500]
DOWNSWING_DURATION_THRESHOLDS_HANDS = [5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000, 150000, 200000, 300000]


# --- Default Data for First Run ---
# This section defines the default data that populates the application when a user visits for the first time.
# It provides a sensible starting point to demonstrate the app's functionality.
DEFAULT_STAKES_DATA = pd.DataFrame([
    # Using more realistic win rates for today's online games, based on user feedback.
    {"name": "NL20", "bb_size": 0.20, "bb_per_100": 2.0, "ev_bb_per_100": 2.0, "std_dev_per_100": 100.0, "sample_hands": 50000, "win_rate_drop": 0.0, "rake_bb_per_100": 10.0},
    {"name": "NL50", "bb_size": 0.50, "bb_per_100": 2.0, "ev_bb_per_100": 2.0, "std_dev_per_100": 100.0, "sample_hands": 40000, "win_rate_drop": 0.0, "rake_bb_per_100": 8.0},
    {"name": "NL100", "bb_size": 1.00, "bb_per_100": 1.0, "ev_bb_per_100": 1.0, "std_dev_per_100": 100.0, "sample_hands": 20000, "win_rate_drop": 1.0, "rake_bb_per_100": 7.0},
    {"name": "NL200", "bb_size": 2.00, "bb_per_100": 0.5, "ev_bb_per_100": 0.5, "std_dev_per_100": 100.0, "sample_hands": 0, "win_rate_drop": 0.5, "rake_bb_per_100": 5.0},
])

DEFAULT_STRATEGIES = {
    "Balanced (50 BI)": { # 50 BI for 100% mix, ~37.5 BI for shot-taking
        "type": "standard",
        "rules": [
            {"threshold": 10000, "tables": {"NL200": "100%"}},
            {"threshold": 7500, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 5000, "tables": {"NL100": "100%"}},
            {"threshold": 3750, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 2500, "tables": {"NL50": "100%"}},
            {"threshold": 1875, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 1000, "tables": {"NL20": "100%"}},
        ]
    },
    "Aggressive (30 BI)": { # 30 BI for 100% mix, ~22.5 BI for shot-taking
        "type": "standard",
        "rules": [
            {"threshold": 6000, "tables": {"NL200": "100%"}},
            {"threshold": 4500, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 3000, "tables": {"NL100": "100%"}},
            {"threshold": 2250, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 1500, "tables": {"NL50": "100%"}},
            {"threshold": 1125, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 600, "tables": {"NL20": "100%"}},
        ]
    },
    "Conservative (100 BI)": { # 100 BI for 100% mix, ~75 BI for shot-taking
        "type": "standard",
        "rules": [
            {"threshold": 20000, "tables": {"NL200": "100%"}},
            {"threshold": 15000, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 10000, "tables": {"NL100": "100%"}},
            {"threshold": 7500, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 5000, "tables": {"NL50": "100%"}},
            {"threshold": 3750, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 2000, "tables": {"NL20": "100%"}},
        ]
    },
    "Hysteresis (Sticky)": {
        "type": "hysteresis",
        "num_buy_ins": 40
    }
}

# --- Page and UI Configuration ---
st.set_page_config(layout="wide", page_title="Poker Bankroll Simulator")

st.title("Poker Bankroll Simulator")
st.write("An interactive tool to simulate poker bankroll progression based on your data and strategies. Based on the logic from `Final BR Simulator v1_5.py`.")

# The user guide is placed in an expander to avoid cluttering the main interface.
with st.expander("Need Help? Click here for the User Guide"):
    # The guide is written in Markdown for easy formatting. It explains every feature of the app
    # to help users understand the inputs and interpret the results correctly.
    st.markdown("""
    ## A User's Guide to the Poker Bankroll Simulator

    Welcome! This tool is designed to help you test different bankroll management (BRM) strategies to see how they might perform over hundreds of thousands of hands. It's not a crystal ball, but by running thousands of simulations, it can give you powerful insights into the risks and rewards of your chosen approach.

    This guide will walk you through every setting and show you how to make sense of the results.

    ### Section 1: The Control Panel (Sidebar)

    This is where you set the core parameters for the simulation.

    #### General Settings
    *   **Starting Bankroll**: How much money you're starting with.
    *   **Target Bankroll**: Your goal. The simulation will tell you the probability of hitting this target.
    *   **Ruin Threshold**: Your "game over" point. If a simulated version of you drops to this amount, that run stops.
    *   **Number of Simulations**: How many "alternate realities" to run. More is better (2,000+ is recommended for accurate results), but it takes longer. Think of it as running the same experiment thousands of times to get a reliable average.
    *   **Total Hands to Simulate**: The time horizon for each simulation. How many hands will each "alternate you" play? 50,000 is a good starting point.

    #### Gameplay & Rakeback Settings
    *   **Hands per Bankroll Check**: How often the simulation checks your bankroll to decide if you should move up or down in stakes. 1,000 hands is a common choice.
    *   **Rakeback (%)**: The percentage of rake you get back. This is a critical, variance-free boost to your win rate. Setting it to 0% can make aggressive strategies unprofitable and significantly increase your Risk of Ruin.
    *   **Enable Stop-Loss**: If enabled, simulations will 'sit out' for the next hand block (defined by 'Hands per Bankroll Check') after losing more than the specified amount in a single block. This simulates taking a break after a big losing session.

    #### Withdrawal & Deposit Settings
    *   **Enable Monthly Withdrawals**: Turns on the withdrawal feature.
    *   **Monthly Volume (Hands)**: Defines how many hands make up a "month". The simulation will attempt a withdrawal after this many hands are played.
    *   **Withdrawal Strategy**: Choose how to withdraw money:
        *   **Fixed Amount**: Withdraw a set amount each month.
        *   **Percentage of Profits**: Withdraw a percentage of the profits made in that month.
        *   **Withdraw Down to Threshold**: Withdraw any amount above a specific bankroll level.
    *   **Minimum Bankroll for Withdrawal**: A safety net. No withdrawals will be made if the bankroll is below this amount, regardless of the chosen strategy.

    #### Advanced Statistical Settings
    This is the "secret sauce" of the simulator that makes it more realistic than a simple variance calculator.

    *   **Prior Sample Size**: This setting controls the model's **uncertainty** about your true skill level. It's a subtle but powerful concept.
        *   **Analogy**: Think of your `Sample Hands` as a single poll you conducted. The `Prior Sample Size` is like a large collection of historical data from a professional polling firm. The simulation combines both to make a final prediction.
        *   **Low Prior (e.g., 5,000)**: The total "pool of evidence" (your hands + the prior) is small. This makes the model **uncertain**, so it creates a *wide* range of long-term luck outcomes (`Assigned WR`s) to reflect that uncertainty.
        *   **High Prior (e.g., 50,000)**: The total "pool of evidence" is large. This makes the model very **certain**, so it creates a *narrow* range of long-term luck outcomes. The `Assigned WR`s will be very consistent across simulations.
        *   **In short**: To get a wider, more "lucky" distribution of outcomes, use a lower prior. To get a more consistent, less "lucky" distribution, use a higher prior.
    *   **Weight for 0-Hand Stake Estimates**: For stakes you've never played, this slider balances your own guess (1.0) vs. the model's guess based on the stakes below it (0.0). A value of 0.5 is a good middle ground.

    #### Plotting & Display Settings
    *   **Distribution Plot Percentile Range**: Controls the 'zoom' on the Final Bankroll Distribution comparison plot. A value of 99 shows the 1st to 99th percentile of outcomes. A value of 95 shows the 5th to 95th, zooming in more on the central results.

    #### Model Validation
    *   **Load Sanity Check Config**: This is an advanced feature for validating the simulation engine. It loads a simple, single-stake scenario that can be compared against standard variance calculator results to confirm the core math is sound. This will overwrite your current settings.

    #### Other Settings
    *   **Random Seed**: This is like the specific shuffle of a deck of cards. Using the same seed will always give you the exact same results. Change the seed (or click the 🎲 button) to get a different "shuffle" and a new set of random outcomes.
    *   **Save & Load Configuration**: Use these buttons to save all your settings to a file and load them back later. Very useful!

    ---

    ### Section 2: Your Data & Strategies (Main Tabs)

    #### Tab 1: Stakes Data
    This is where you tell the simulator about your performance.

    *   **`EV Win Rate (bb/100)`**: **This is the most important input for the simulation!** Use your "All-in Adj bb/100" from your poker tracker. This is the best measure of your true skill and is used for all core calculations.
    *   **`Std Dev (bb/100)`**: Your standard deviation. This measures how "swingy" your results are. You can find this in your tracker. 80-120 is typical for No-Limit Hold'em.
    *   **`Sample Hands`**: How many hands of data you have for this stake. This is crucial! A large sample tells the model to be confident in your EV Win Rate. A small sample tells the model that your true win rate is uncertain, so it will simulate a wider range of possibilities (the "luck" factor).
    *   **`Win Rate Drop`**: The estimated change in your win rate (in bb/100) when moving up from the *previous* stake. Use a positive value for a drop (e.g., 1.5) and a negative value for an expected increase (e.g., -1.0). The lowest stake should have a value of 0.
    *   **`Rake (bb/100)`**: How much rake you pay at this stake. Used to calculate your rakeback profit.
    *   **Save and Sort Stakes Button**: **Always click this after making changes!** This is required for the Hysteresis (Sticky) strategy to function correctly.

    #### Tab 2: Bankroll Management Strategies
    Here you define the rules for moving up and down stakes.

    *   **Standard Strategy**:
        *   You set bankroll `thresholds`. When your bankroll is above a threshold, the corresponding rule applies.
        *   The `table mix` tells the simulator what games to play. You can use:
            *   **Fixed Ratios**: `NL20: 1`, `NL50: 3` means you play 25% of your tables at NL20 and 75% at NL50.
            *   **Percentages**: `NL100: "100%"` means you play only NL100.
            *   **Percentage Ranges**: `NL200: "20-40%"` models uncertainty in your game selection.
    *   **Hysteresis (Sticky) Strategy**:
        *   This is a "move up fast, move down slow" strategy designed to prevent you from dropping stakes during small downswings. You can set a single buy-in buffer for all stakes or check the **"Use per-stake buy-in buffers"** box to define a unique buffer for each stake individually.
        *   **Moving Up**: You move up to a new stake (e.g., NL50) only when you have enough buy-ins for it (e.g., 40 BIs for NL50).
        *   **Moving Down (The "Sticky" Part)**: Once you're playing NL50, you *only* move back down to NL20 if your bankroll drops below the requirement for NL20 (e.g., 40 BIs for NL20). This creates a "buffer zone" where you stick to the higher stake.

    ---

    ### Section 3: Interpreting the Results

    This is where you see the outcome of thousands of possible poker careers.

    #### Automated Strategy Analysis
    *   This is the first section you'll see in the results. It's an AI-generated summary that gives you plain-English insights into your strategies. It will tell you which strategy was the safest, which had the most upside, and why one might have performed better than another. This analysis is also included in the PDF report.

    #### Strategy Comparison
    *   **Summary Table**: A quick overview of the most important metrics.
        *   **Median Final BR**: The 50th percentile outcome. Half the time you do better, half the time you do worse. It's a good measure of the "typical" result.
        *   **Mode Final BR**: The most frequently occurring final bankroll outcome.
        *   **Median Growth**: The median percentage growth from the starting bankroll.
        *   **Median Hands Played**: The median number of hands played per simulation. This can be lower than the total if a stop-loss is frequently triggered.
        *   **Median Profit (Play)**: The median profit from gameplay only, excluding rakeback.
        *   **Median Rakeback**: The median rakeback earned. Compare this to "Median Profit (Play)" to see how much your strategy relies on rakeback.
        *   **Risk of Ruin (RoR)**: Your chance of going broke according to your ruin threshold. A critical risk metric.
        *   **Target Prob**: Your chance of hitting your goal.
        *   **5th %ile BR**: The 5th percentile final bankroll. 95% of simulations ended with a bankroll higher than this value.
        *   **P95 Max Downswing**: A measure of a "worst-case" downswing. 5% of the time, you will have a downswing even bigger than this number.
    *   **Comparison Plots**:
        *   **Median Bankroll Progression**: Shows the typical journey of your bankroll over time for each strategy.
        *   **Final Bankroll Distribution**: A key plot. A tall, narrow peak means a strategy is very consistent. A short, wide curve means the strategy has a wider range of outcomes (higher risk/reward).
        *   **Psychological Cost: Time Spent Below Bankroll Peak**: A bar chart showing the median percentage of hands a strategy spends 'underwater'. A lower percentage indicates a smoother, less stressful journey.
        *   **Risk vs. Reward Analysis**: A scatter plot showing the trade-off between risk (X-axis) and reward (Y-axis). The ideal strategy is in the top-left corner (low risk, high reward).

    #### Detailed Analysis (Per Strategy)
    This section gives you a deep dive into each strategy.

    *   **Visuals**:
        *   **Bankroll Progression**: Shows the median path (blue line), the 25th-75th percentile range (shaded area), and 50 random individual simulations (grey lines) to give you a feel for the variance.
        *   **Distribution of Assigned Luck (WR)**: This chart visualizes the "luck" factor. It shows the range of "true skill + long-term luck" the simulation assigns to different runs. The width of this curve is determined by your `Sample Hands` input—less data means more uncertainty and a wider curve.
        *   **Maximum Downswing Distribution**: Shows the full range of potential downswings. Helps you mentally prepare for the worst!
    *   **Key Insights**:
        *   **Hands Distribution**: Shows where you'll spend most of your time. For each stake, it provides:
            *   **Percentage**: The share of total hands played at this stake across all simulations.
            *   **Avg. WR**: The average win rate the simulation used for this stake. This includes the "luck" factor, so it will differ slightly from your input. It represents the average "true skill" assigned to players at this stake.
            *   **Trust**: A percentage showing how much the model "trusts" your input EV Win Rate. It's calculated based on your `Sample Hands` versus the `Prior Sample Size`. A high trust percentage means the model is confident in your data and will apply less long-term luck (variance) to your win rate. A low trust percentage means the model is uncertain and will simulate a wider range of good and bad luck.
        *   **Median Hands Played**: The median number of hands played per simulation. This can be lower than the total if a stop-loss is frequently triggered.
        *   **Median Stop-Losses**: If enabled, this metric shows the typical number of times a stop-loss was triggered during a simulation run. It's a good indicator of session volatility.
        *   **Risk of Demotion**: The chance you'll have to move down after successfully moving up to a stake.
    *   **Percentile Win Rate Analysis**: This section is crucial for understanding *why* some runs did well and others did poorly. It shows a five-number summary (5th, 25th, Median, 75th, 95th) of outcomes, giving you a detailed look at the full spectrum of possibilities.
        *   **5th Percentile**: A typical "bad run" or unlucky outcome.
        *   **25th Percentile**: A "mildly bad" run. Not a disaster, but a common losing scenario.
        *   **Median (50th)**: The outcome for the run that finished with the median final bankroll.
        *   **75th Percentile**: A "mildly good" run. A common winning scenario.
        *   **95th Percentile**: A typical "good run" or heater.
    *   For each percentile, you'll see these metrics:
        *   **Assigned WR**: The "true" win rate (skill + long-term luck) the simulator assigned to the entire run. A high number here means this simulated "you" was on a long-term heater.
            It is influenced by:
            *   **`EV Win Rate (bb/100)`**: Your stated skill level, which is the starting point.
            *   **`Sample Hands`**: The more hands you have, the more the model trusts your EV Win Rate, and the smaller the "luck" adjustment will be.
            *   **`Std Dev (bb/100)`**: Higher volatility means more uncertainty, which allows for a wider range of possible long-term luck adjustments.
            *   **`Prior Sample Size`**: The model's "skepticism." A high value here makes the model more skeptical of small samples, leading to a wider luck distribution.
        *   **Play WR**: This is the actual, realized win rate from playing the hands. It's the final result after all the session-to-session variance is accounted for. It's calculated by taking the `Assigned WR` and adding random variance for every block of hands played.
            It is influenced by:
            *   **`Assigned WR`**: The baseline skill + long-term luck for the run.
            *   **`Std Dev (bb/100)`**: Directly determines the magnitude of the upswings and downswings in each session.
            *   **`Hands per Bankroll Check`**: The length of the "session" being simulated.
        *   **Rakeback WR**: The extra win rate you got from rakeback.

    Finally, you can download a **Full PDF Report** with all of this information, including the Automated Strategy Analysis, for offline viewing and sharing. Happy simulating!
    """)

# --- Session State Initialization ---
# This is a crucial block for any complex Streamlit app. It ensures that all necessary keys
# are present in `st.session_state` when the app first loads.
# By checking for each key individually, we can add new features and settings over time
# without breaking the app for users who might have an older version of the session state in their browser cache.
# Initialize each key separately to ensure new features work for users with old session states.
if 'start_br' not in st.session_state: st.session_state.start_br = 2500
if 'target_br' not in st.session_state: st.session_state.target_br = 3000
if 'ruin_thresh' not in st.session_state: st.session_state.ruin_thresh = 750
if 'num_sims' not in st.session_state: st.session_state.num_sims = 2000
if 'total_hands' not in st.session_state: st.session_state.total_hands = 50000
if 'hands_per_check' not in st.session_state: st.session_state.hands_per_check = 1000
if 'rb_percent' not in st.session_state: st.session_state.rb_percent = 20
if 'prior_sample' not in st.session_state: st.session_state.prior_sample = 50000
if 'zero_hands_weight' not in st.session_state: st.session_state.zero_hands_weight = 0.5
if 'enable_stop_loss' not in st.session_state: st.session_state.enable_stop_loss = False
if 'stop_loss_bb' not in st.session_state: st.session_state.stop_loss_bb = 300
if 'seed' not in st.session_state: st.session_state.seed = 98765
if 'enable_withdrawals' not in st.session_state: st.session_state.enable_withdrawals = False
if 'monthly_volume_hands' not in st.session_state: st.session_state.monthly_volume_hands = 15000
if 'withdrawal_strategy' not in st.session_state: st.session_state.withdrawal_strategy = "Fixed Amount (€)"
if 'withdrawal_value' not in st.session_state: st.session_state.withdrawal_value = 500
if 'min_br_for_withdrawal' not in st.session_state: st.session_state.min_br_for_withdrawal = 2000
if 'plot_percentile_limit' not in st.session_state: st.session_state.plot_percentile_limit = 99

# Initialize flags and data containers.
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
    st.session_state.simulation_output = None

# Load default data into session state if it doesn't already exist.
if 'stakes_data' not in st.session_state:
    st.session_state.stakes_data = DEFAULT_STAKES_DATA

if 'strategy_configs' not in st.session_state:
    st.session_state.strategy_configs = DEFAULT_STRATEGIES.copy()

# --- Callback Functions ---
# Callbacks are functions that are executed in response to a user interaction, like clicking a button
# or editing a widget. They are essential for creating a dynamic and responsive UI.

def click_run_button():
    """Callback function to set the simulation flag when the button is clicked."""
    st.session_state.run_simulation = True
    st.session_state.simulation_output = None # Clear old results when a new run is requested

def setup_sanity_check():
    """
    Callback to load a simple, predictable configuration for validating the simulation engine.
    This setup uses a single stake with a very large sample size to minimize the Bayesian "luck" model,
    allowing the simulation results to be compared directly against standard variance calculator formulas.
    """
    st.session_state.stakes_data = pd.DataFrame([
        {"name": "NL20", "bb_size": 0.20, "bb_per_100": 5.0, "ev_bb_per_100": 5.0, "std_dev_per_100": 100.0, "sample_hands": 10_000_000, "win_rate_drop": 0.0, "rake_bb_per_100": 0.0},
    ])
    st.session_state.strategy_configs = {
        "Sanity Check (NL20 Only)": {
            "type": "standard",
            "rules": [{"threshold": 0, "tables": {"NL20": "100%"}}]
        }
    }
    st.session_state.target_br = 11000 # Set to the expected final BR (10k start + 1k profit)
    st.session_state.start_br = 10000 # Use a large BR to make ruin effectively impossible
    st.session_state.total_hands = 100000
    st.session_state.rb_percent = 0 # Turn off rakeback for simplicity
    st.session_state.num_sims = 10000 # Use a high number for an accurate check
    st.session_state.prior_sample = 10_000_000 # Effectively disable Bayesian model
    st.session_state.ruin_thresh = 0 # Set ruin to 0 to avoid truncating results
    st.session_state.enable_stop_loss = False # Ensure stop-loss is off for validation
    st.session_state.enable_withdrawals = False # Ensure withdrawals are off for validation
    st.session_state.hands_per_check = 1000 # Reset to a standard value for consistency
    st.session_state.zero_hands_weight = 0.5 # Reset to default, though not used in this mode
    st.session_state.plot_percentile_limit = 99 # Reset plot zoom to default
    st.toast("Sanity Check configuration loaded. You can now click 'Run Simulation'.", icon="🔬")

def add_strategy():
    """
    Callback to add a new, blank strategy to the `strategy_configs` dictionary.
    It finds a unique name (e.g., "New Strategy 1", "New Strategy 2") to avoid conflicts
    and pre-populates the strategy with a sensible default rule.
    """
    i = st.session_state.get('strategy_counter', 0) + 1
    while True:
        new_name = f"New Strategy {i}"
        if new_name not in st.session_state.strategy_configs:
            st.session_state.strategy_counter = i
            break
        i += 1

    # Use the first available stake from the user's data as a sensible default for the new rule.
    available_stakes = [
        name for name in st.session_state.stakes_data['name']
        if pd.notna(name) and str(name).strip()
    ]
    if available_stakes:
        default_tables = {available_stakes[0]: "100%"}
    else:
        # If no stakes are defined, create an empty rule
        default_tables = {}

    st.session_state.strategy_configs[new_name] = {
        "type": "standard",
        "rules": [{"threshold": 1000, "tables": default_tables}]
    }

def remove_strategy(name_to_remove):
    """Callback to remove a strategy from the `strategy_configs` dictionary."""
    if name_to_remove in st.session_state.strategy_configs:
        del st.session_state.strategy_configs[name_to_remove]

def clone_strategy(name_to_clone):
    """
    Callback to clone an existing strategy.
    It finds a unique name for the clone and performs a `copy.deepcopy()` to ensure that
    the new strategy is a completely independent object, preventing unintended side effects.
    """
    if name_to_clone not in st.session_state.strategy_configs:
        return  # Should not happen

    original_config = st.session_state.strategy_configs[name_to_clone]

    # Find a unique name for the clone
    i = 1
    while True:
        new_name = f"{name_to_clone} (Copy {i})"
        if new_name not in st.session_state.strategy_configs:
            break
        i += 1

    # Create a deep copy to avoid shared references, especially for the 'rules' list
    cloned_config = copy.deepcopy(original_config)
    st.session_state.strategy_configs[new_name] = cloned_config

def sync_stakes_data():
    """
    Callback to apply edits from the data_editor to the main stakes_data DataFrame.
    This function is triggered by the "Save and Sort Stakes" button and handles the dictionary
    of changes (`edited_rows`, `added_rows`, `deleted_rows`) provided by `st.data_editor`.
    """
    editor_state = st.session_state.stakes_data_editor

    # Make a copy of the original DataFrame to apply changes to
    # This is a good practice to avoid modifying the state directly until all validation passes.
    df = st.session_state.stakes_data.copy()

    # Apply deletions first, in reverse order to not mess up indices
    if editor_state["deleted_rows"]:
        df = df.drop(index=editor_state["deleted_rows"]).reset_index(drop=True)

    # Apply edits
    if editor_state["edited_rows"]:
        for index, changes in editor_state["edited_rows"].items():
            for col_name, new_value in changes.items():
                df.loc[index, col_name] = new_value

    # Apply additions
    if editor_state["added_rows"]:
        added_df = pd.DataFrame(editor_state["added_rows"])
        df = pd.concat([df, added_df], ignore_index=True)

    # --- Sorting Logic: Ensure stakes are always ordered by bb_size ---
    df['bb_size'] = pd.to_numeric(df['bb_size'], errors='coerce')
    df = df.sort_values(by='bb_size', ascending=True, na_position='last').reset_index(drop=True)

    # --- Validation Logic: Check for missing required values ---
    invalid_rows = df[df['name'].isnull() | (df['name'].astype(str).str.strip() == '') | df['bb_size'].isnull()]
    if not invalid_rows.empty:
        st.error("One or more rows have a missing 'name' or an invalid 'bb_size'. Please fill in all required fields. Changes have not been saved.")
        return # Abort the sync

    # --- Validation Logic: Check for duplicate stake names ---
    # We only consider non-empty, non-null names for duplication checks.
    stake_names = df['name'].dropna().astype(str).str.strip()
    # Filter out empty strings after stripping whitespace
    stake_names = stake_names[stake_names != '']
    duplicates = stake_names[stake_names.duplicated()].unique()

    if len(duplicates) > 0:
        st.error(f"Duplicate stake names found: {', '.join(duplicates)}. Please ensure all stake names are unique. Changes have not been saved.")
        return # IMPORTANT: Abort the save if duplicates are found

    # Update the main session state with the modified DataFrame
    st.session_state.stakes_data = df

def sync_strategy_rules(strategy_name):
    """
    Callback to sync a "Standard" strategy's data editor state back to the main `strategy_configs` dictionary.
    This is more complex than `sync_stakes_data` because it needs to convert the edited DataFrame
    back into the specific list-of-dictionaries format that the simulation engine expects.
    """
    edits = st.session_state[f"rules_{strategy_name}"]
    original_rules = st.session_state.strategy_configs[strategy_name].get("rules", [])

    available_stakes = [
        name for name in st.session_state.stakes_data['name']
        if pd.notna(name) and str(name).strip()
    ]
    expected_columns = ['threshold'] + available_stakes

    # Convert the original rules from list-of-dicts to a DataFrame to make editing easier.
    df = pd.DataFrame(
        [{'threshold': r['threshold'], **r.get('tables', {})} for r in original_rules],
        columns=expected_columns
    )

    # Apply changes from the editor
    if edits["deleted_rows"]:
        df = df.drop(index=edits["deleted_rows"])
    if edits["edited_rows"]:
        for index, changes in edits["edited_rows"].items():
            for col_name, new_value in changes.items():
                df.loc[index, col_name] = new_value
    if edits["added_rows"]:
        added_df = pd.DataFrame(edits["added_rows"], columns=expected_columns)
        df = pd.concat([df, added_df], ignore_index=True)

    # --- Sorting Logic: Rules must be sorted by threshold, descending ---
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')
    df = df.sort_values(by='threshold', ascending=False, na_position='last').reset_index(drop=True)

    # Convert the modified DataFrame back into the required list-of-dicts format for the engine.
    new_rules = []
    for _, row in df.iterrows():
        threshold_val = row.get('threshold')
        if pd.isna(threshold_val) or threshold_val <= 0:
            continue
        tables = {}
        for stake in available_stakes:
            # Check if the stake column exists and has a non-empty value.
            # This handles cases where columns might be missing or have NaN values.
            if stake in row and pd.notna(row[stake]) and row[stake] != "":
                try:
                    tables[stake] = int(row[stake])
                except (ValueError, TypeError):
                    tables[stake] = str(row[stake])
        # A rule is considered valid for the UI as long as it has a threshold.
        # An empty table mix is acceptable during editing and will be treated as "No Play"
        # by the simulation engine. Removing the `if tables:` check fixes the sorting bug.
        new_rules.append({"threshold": int(threshold_val), "tables": tables})
    st.session_state.strategy_configs[strategy_name]['rules'] = new_rules

# --- Sidebar for User Inputs ---
# The sidebar contains all the global parameters for the simulation.
st.sidebar.header("Simulation Parameters")

st.sidebar.button(
    "**Run Simulation**",
    on_click=click_run_button,
    use_container_width=True,
    type="primary",
    help="Click to run the simulation with the current settings."
)

# Using expanders helps organize the large number of settings into logical groups.
with st.sidebar.expander("General Settings", expanded=True):
    st.number_input("Starting Bankroll (€)", min_value=0, step=100, help="The amount of money you are starting with for the simulation.", key="start_br")
    st.number_input("Target Bankroll (€)", min_value=0, step=100, help="The bankroll amount you are aiming to reach. This is used to calculate 'Target Probability'.", key="target_br")
    st.number_input("Ruin Threshold (€)", min_value=0, step=50, help="If a simulation's bankroll drops to or below this value, it is considered 'ruined' and stops.", key="ruin_thresh")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Number of Simulations", min_value=10, max_value=50000, step=100, help="How many times to run the entire simulation from start to finish. Higher numbers give more accurate results but take longer. (e.g., 2000-10000)", key="num_sims")
    with col2:
        st.number_input("Total Hands to Simulate", min_value=1000, step=1000, help="The total number of hands to play in a single simulation run. This determines the time horizon.", key="total_hands")

with st.sidebar.expander("Gameplay & Rakeback Settings", expanded=True):
    st.number_input("Hands per Bankroll Check", min_value=100, step=100, help="How often (in hands) to check your bankroll and apply your BRM rules. A common value is 1000.", key="hands_per_check")
    st.slider("Rakeback (%)", 0, 100, help="The percentage of rake you get back from the poker site. This is added to your profit at the end of each 'hand block' (the interval defined by 'Hands per Bankroll Check').", key="rb_percent")
    if st.session_state.rb_percent == 0:
        st.info("Note: Setting rakeback to 0% can significantly reduce profitability and increase Risk of Ruin, especially for aggressive strategies.")
    st.checkbox("Enable Stop-Loss", key="enable_stop_loss", help="If enabled, simulations will 'sit out' for the next hand block after losing more than the specified amount in a single block. This simulates taking a break after a big losing session. The loss is calculated based on gameplay profit (before rakeback).")
    if st.session_state.enable_stop_loss:
        st.number_input(
            "Stop-Loss (in big blinds)",
            value=st.session_state.stop_loss_bb,
            min_value=1,
            step=10,
            key="stop_loss_bb",
            help="The number of big blinds lost in a single block that will trigger the stop-loss. A common value is 300-500bb (3-5 buy-ins)."
        )

with st.sidebar.expander("Withdrawal & Deposit Settings", expanded=False):
    st.checkbox("Enable Monthly Withdrawals", key="enable_withdrawals", help="If enabled, the simulation will attempt to withdraw funds periodically based on the rules below.")
    if st.session_state.enable_withdrawals:
        # We handle validation manually after the input to avoid unexpected widget behavior
        # when the min_value property changes dynamically.
        hands_per_check_val = st.session_state.hands_per_check    

        st.number_input(
            "Monthly Volume (Hands)",
            min_value=1, # Use a static minimum to avoid widget conflicts
            step=1000,
            key="monthly_volume_hands",
            help="The number of hands that constitutes a 'month'. A withdrawal will be attempted after this many hands are played."
        )
        
        # Post-render validation: If the current monthly volume is invalid, correct it and notify the user.
        if st.session_state.monthly_volume_hands < hands_per_check_val:
            st.warning(f"Monthly Volume must be at least {hands_per_check_val:,} (the 'Hands per Bankroll Check' value). Adjusting...")
            st.session_state.monthly_volume_hands = hands_per_check_val
            st.rerun()
            
        st.selectbox(
            "Withdrawal Strategy",
            options=["Fixed Amount (€)", "Percentage of Profits (%)", "Withdraw Down to Threshold (€)"],
            key="withdrawal_strategy",
            help=(
                "- **Fixed Amount:** Withdraw a set amount each month.\n"
                "- **Percentage of Profits:** Withdraw a percentage of profits made during that month.\n"
                "- **Withdraw Down to Threshold:** Withdraw any amount above a specified bankroll level."
            )
        )

        # Determine the label for the next input based on the strategy
        if "Fixed Amount" in st.session_state.withdrawal_strategy:
            label, min_val, step = "Withdrawal Amount (€)", 1, 50
        elif "Percentage" in st.session_state.withdrawal_strategy:
            label, min_val, step = "Withdrawal Percentage (%)", 1, 5
        else: # Threshold
            label, min_val, step = "Bankroll Threshold to Withdraw Down To (€)", 0, 100

        st.number_input(label, min_value=min_val, step=step, key="withdrawal_value")

        st.number_input(
            "Minimum Bankroll for Withdrawal (€)",
            min_value=st.session_state.ruin_thresh,
            step=100,
            key="min_br_for_withdrawal",
            help="A safety net. No withdrawals will be made if the bankroll is below this amount, regardless of the strategy."
        )


with st.sidebar.expander("Advanced Statistical Settings", expanded=False):
    st.number_input(
        "Prior Sample Size (for Bayesian model)",
        min_value=1000, step=1000,
        help=(
            "Controls the model's confidence in your win rate data. Think of it as a 'benchmark' sample size against which your own sample is compared.\n\n"
            "**Intuition:** If this is 50,000 and your stake has a 50,000 hand sample, the model is confident. If your sample is only 5,000 hands, the model is uncertain and will simulate a wider range of long-term luck (both good and bad heaters) for your 'true' win rate."
        ),
        key="prior_sample"
    )
    st.slider("Weight for 0-Hand Stake Estimates", 0.0, 1.0, step=0.05, help="For stakes where you have no hands played, this slider balances between your manual win rate estimate (1.0) and the model's extrapolation from other stakes (0.0).", key="zero_hands_weight")

with st.sidebar.expander("Plotting & Display Settings", expanded=False):
    st.slider(
        "Distribution Plot Percentile Range",
        min_value=90, max_value=99, value=99,
        help="Controls the 'zoom' on the Final Bankroll Distribution comparison plot. A value of 99 shows the 1st to 99th percentile of outcomes. A value of 95 shows the 5th to 95th, zooming in more on the central results but hiding more of the tails.",
        key="plot_percentile_limit"
    )

with st.sidebar.expander("Model Validation", expanded=False):
    st.button("Load Sanity Check Config", on_click=setup_sanity_check, help="Loads a simple configuration to validate the simulation engine against a standard variance calculator. This will overwrite your current settings.", use_container_width=True)


def randomize_seed():
    """Generates a new random seed."""
    # This is a simple helper to provide a new random number for the seed.
    import random
    st.session_state.seed = random.randint(1, 1_000_000)

st.sidebar.caption("Random Seed")
# Using columns to place the "randomize" button next to the input field.
seed_col1, seed_col2 = st.sidebar.columns([3, 1])
with seed_col1:
    st.number_input("Random Seed", label_visibility="collapsed", step=1, help="A fixed number that ensures the exact same random results every time. Change it to get a different set of random outcomes.", key="seed")
with seed_col2:
    st.button("🎲", on_click=randomize_seed, help="Generate a new random seed.", use_container_width=True)

st.sidebar.header("Save & Load Configuration")

def get_full_config_as_json():
    """
    Gathers all relevant session state variables into a single dictionary.
    This dictionary is then serialized to a JSON string, which can be downloaded by the user.
    This function is called by the `st.download_button`.
    """
    config = {
        "parameters": {
            "start_br": st.session_state.start_br, "target_br": st.session_state.target_br,
            "ruin_thresh": st.session_state.ruin_thresh, "num_sims": st.session_state.num_sims, "total_hands": st.session_state.total_hands,
            "hands_per_check": st.session_state.hands_per_check,
            "rb_percent": st.session_state.rb_percent,
            "enable_stop_loss": st.session_state.enable_stop_loss, "stop_loss_bb": st.session_state.stop_loss_bb,
            "enable_withdrawals": st.session_state.enable_withdrawals,
            "monthly_volume_hands": st.session_state.monthly_volume_hands,
            "withdrawal_strategy": st.session_state.withdrawal_strategy, "withdrawal_value": st.session_state.withdrawal_value,
            "min_br_for_withdrawal": st.session_state.min_br_for_withdrawal,
            "prior_sample": st.session_state.prior_sample, "zero_hands_weight": st.session_state.zero_hands_weight, "plot_percentile_limit": st.session_state.plot_percentile_limit,
            "seed": st.session_state.seed,
        },
        "stakes_data": st.session_state.stakes_data.to_dict('records'),
        "strategy_configs": st.session_state.strategy_configs,
    }
    return json.dumps(config, indent=4)

st.sidebar.download_button(
    label="Save Configuration to File",
    data=get_full_config_as_json(),
    file_name=f"poker_sim_config_{datetime.datetime.now().strftime('%Y%m%d')}.json",
    mime="application/json",
    use_container_width=True
)

def process_uploaded_config():
    """
    Callback to load a configuration from an uploaded JSON file.
    This function is triggered when a user uploads a file to the `st.file_uploader`.
    It parses the JSON, validates its structure, and updates the session state with the loaded values.
    """
    uploaded_file = st.session_state.get("config_uploader")
    if uploaded_file is None:
        return

    try:
        config_str = uploaded_file.getvalue().decode("utf-8")
        loaded_data = json.loads(config_str)

        # --- Validate and load parameters ---
        if "parameters" in loaded_data and isinstance(loaded_data["parameters"], dict):
            for key, value in loaded_data["parameters"].items():
                if key in st.session_state:  # Only load keys that exist in session state
                    st.session_state[key] = value
        else:
            st.error("Invalid config file: 'parameters' section is missing or malformed.")
            return #

        # --- Validate and load stakes data ---
        if "stakes_data" in loaded_data and isinstance(loaded_data["stakes_data"], list):
            df = pd.DataFrame(loaded_data["stakes_data"])
            # Also sort the stakes data, just like the 'Save and Sort' button does, to ensure consistency.
            if 'bb_size' in df.columns:
                df['bb_size'] = pd.to_numeric(df['bb_size'], errors='coerce')
                df = df.sort_values(by='bb_size', ascending=True, na_position='last').reset_index(drop=True)
            st.session_state.stakes_data = df
        else:
            st.error("Invalid config file: 'stakes_data' section is missing or malformed.")
            return

        # --- Validate and load strategies ---
        if "strategy_configs" in loaded_data and isinstance(loaded_data["strategy_configs"], dict):
            st.session_state.strategy_configs = loaded_data["strategy_configs"]
        else:
            st.error("Invalid config file: 'strategy_configs' section is missing or malformed.")
            return

        st.success("Configuration loaded successfully! The app has been updated with the new settings.")
        st.session_state.simulation_output = None  # Clear results from any previous run

    except json.JSONDecodeError:
        st.error("Error: The uploaded file is not a valid JSON file.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {e}")

st.sidebar.file_uploader(
    "Load Configuration from File",
    type="json",
    on_change=process_uploaded_config,
    key="config_uploader",
    help="Upload a previously saved JSON configuration file to restore all settings."
)

# --- Main Area for Data and Strategy Inputs ---
# The main area of the app is organized into tabs for a clean user experience.
st.header("Player & Strategy Configuration")

tab1, tab2 = st.tabs(["Stakes Data", "Bankroll Management Strategies"])

with tab1:
    st.subheader("Stakes Data")
    st.write("Enter your performance statistics for each stake you play. You can add or remove rows.")
    # The `st.data_editor` widget provides a powerful, Excel-like interface for editing DataFrames.
    # The data_editor now uses a key and an on_change callback to prevent losing
    # uncommitted data during a rerun. This provides a much smoother editing experience.
    st.data_editor(
        st.session_state.stakes_data,
        key="stakes_data_editor",
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn(
                "Name",
                help="The name of the stake (e.g., 'NL20', 'NL50'). This must be unique."
            ),
            "bb_size": st.column_config.NumberColumn(
                "bb Size (€)",
                help="The size of the big blind in your currency (e.g., 0.20 for NL20).",
                format="€ %.2f"
            ),
            "bb_per_100": st.column_config.NumberColumn(
                "Win Rate (bb/100)",
                help="Your actual, observed win rate at this stake. This is for your reference only and is not used in the simulation calculations.",
                format="%.2f"
            ),
            "ev_bb_per_100": st.column_config.NumberColumn(
                "EV Win Rate (bb/100)",
                help="Your EV-adjusted win rate (All-in Adj bb/100). This is the most important input and is used for the simulation's core calculations.",
                format="%.2f"
            ),
            "std_dev_per_100": st.column_config.NumberColumn(
                "Std Dev (bb/100)",
                help="Your standard deviation in bb/100 hands. A typical value for No-Limit Hold'em is between 80 and 120.",
                format="%.2f"
            ),
            "sample_hands": st.column_config.NumberColumn(
                "Sample Hands",
                help="The number of hands you have played at this stake. A larger sample size tells the model to be more confident in your win rate estimate, resulting in less long-term variance (luck) between simulation runs. With a very large sample, the results will converge with standard variance calculators.",
                format="%d"
            ),
            "win_rate_drop": st.column_config.NumberColumn(
                "Win Rate Drop",
                help="The estimated change in your win rate (in bb/100) when moving up from the *previous* stake. Use a positive value for a drop (e.g., 1.5) and a negative value for an expected increase (e.g., -1.0). The lowest stake should have a value of 0.",
                format="%.2f"
            ),
            "rake_bb_per_100": st.column_config.NumberColumn(
                "Rake (bb/100)",
                help="The average amount of rake you pay in big blinds per 100 hands at this stake. This is used to calculate rakeback.",
                format="%.2f"
            ),
        }
    )
    st.button(
        "Save and Sort Stakes",
        # This button triggers the `sync_stakes_data` callback to process and save the user's changes.
        on_click=sync_stakes_data,
        help="Click to save any changes and sort the table by 'bb_size'. This is required for the Hysteresis strategy."
    )

with tab2:
    st.subheader("Bankroll Management Strategies")
    st.write(
        "Define your strategies below. For 'Standard' strategies, use the data editor to set bankroll thresholds and table mixes. "
        "You can use fixed ratios (e.g., `1`), percentages (e.g., `'80%'`), or percentage ranges (e.g., `'20-40%'`). "
        "Hover over the 'Mix' column headers for more details."
    )
    st.button("Add New Strategy", on_click=add_strategy, use_container_width=True)
    st.write("---")

    # Get the list of available stake names from the stakes data tab.
    # This is crucial for dynamically creating the columns in the strategy rules editor.
    # We filter out any empty/NaN names that can occur when a user adds a new row but hasn't filled it in yet.
    available_stakes = [
        name for name in st.session_state.stakes_data['name']
        if pd.notna(name) and str(name).strip()
    ]
    strategy_names = list(st.session_state.strategy_configs.keys())

    for name in strategy_names:
        if name not in st.session_state.strategy_configs:
            continue # Skip if it was just removed

        # Each strategy gets its own expander to keep the UI clean.
        with st.expander(f"Edit Strategy: {name}", expanded=False):
            current_config = st.session_state.strategy_configs[name]

            # --- Row 1: Name, Type, Clone, Remove ---
            col1, col2, col3, col4 = st.columns([4, 3, 1, 1])
            with col1:
                new_name = st.text_input("Strategy Name", value=name, key=f"name_{name}")
            with col2:
                strategy_type = st.selectbox(
                    "Strategy Type",
                    options=["standard", "hysteresis"],
                    index=0 if current_config.get("type", "standard") == "standard" else 1,
                    key=f"type_{name}"
                )
            with col3:
                st.write("")  # Spacer
                st.write("")  # Spacer
                st.button("Clone", key=f"clone_{name}", on_click=clone_strategy, args=(name,), use_container_width=True)
            with col4:
                st.write("")  # Spacer
                st.write("")  # Spacer
                st.button("Remove", key=f"remove_{name}", on_click=remove_strategy, args=(name,), use_container_width=True)

            # Handle strategy renaming.
            if new_name != name:
                if not new_name.strip():
                    st.warning("Strategy name cannot be empty.")
                elif new_name in st.session_state.strategy_configs:
                    st.warning(f"A strategy named '{new_name}' already exists. Please choose a unique name.")
                else:
                    st.session_state.strategy_configs[new_name] = st.session_state.strategy_configs.pop(name)
                    # A rerun is necessary to rebuild the UI with the new strategy name and keys.
                    st.rerun()

            st.session_state.strategy_configs[name]['type'] = strategy_type

            # --- Row 2: Display type-specific inputs ---
            if strategy_type == 'hysteresis':
                # The Hysteresis strategy has a simpler UI with just buy-in buffer inputs.
                st.info(
                    "**How Hysteresis (Sticky) Strategy Works:**\n\n"
                    "**Important:** This strategy requires the stakes in the 'Stakes Data' tab to be sorted by `bb_size`. Use the 'Sort Stakes' button on that tab.\n\n"
                    "This strategy prevents rapid switching between stakes. You define a buy-in (BI) buffer for each stake.\n\n"
                    "- **Moving Up:** To move up to a higher stake, your bankroll must meet the BI requirement for that new stake.\n"
                    "  - *Example:* To play NL50 with a 40 BI buffer, you need `40 * 100 * €0.50 = €2000`.\n"
                    "- **Moving Down (The \"Sticky\" Part):** Once you are playing a higher stake, you will *only* move down if your bankroll drops below the BI requirement of the *lower* stake.\n"
                    "  - *Example:* If you're playing NL50, you will only drop to NL20 if your bankroll falls below NL20's requirement (e.g., `40 * 100 * €0.20 = €800`).\n\n"
                    "This creates a buffer zone (e.g., between €800 and €2000) where you 'stick' to the higher stake, avoiding moving down on small downswings."
                )

                # Get the current config for buy-ins, which can be an int or a dict
                num_buy_ins_config = current_config.get("num_buy_ins", 40)

                # Determine if we are in per-stake mode based on the data type of the `num_buy_ins` value.
                is_per_stake_mode = isinstance(num_buy_ins_config, dict)

                use_per_stake_checkbox = st.checkbox(
                    "Use per-stake buy-in buffers",
                    value=is_per_stake_mode,
                    key=f"per_stake_cb_{name}"
                )

                if use_per_stake_checkbox:
                    # --- Per-Stake Buy-in Buffer Mode ---
                    if not is_per_stake_mode:
                        # If transitioning from single to per-stake mode, initialize the dictionary with the old value.
                        old_value = num_buy_ins_config if isinstance(num_buy_ins_config, int) else 40
                        new_dict = {stake: old_value for stake in available_stakes}
                        st.session_state.strategy_configs[name]['num_buy_ins'] = new_dict
                        num_buy_ins_config = new_dict # update local var

                    st.write("Define the buy-in buffer required to play at each stake:")

                    cols = st.columns(len(available_stakes) if available_stakes else 1)
                    new_buy_ins_dict = {}
                    for i, stake_name in enumerate(available_stakes):
                        with cols[i]:
                            stake_value = num_buy_ins_config.get(stake_name, 40)
                            new_buy_ins_dict[stake_name] = st.number_input(
                                f"{stake_name} BIs", value=stake_value, min_value=1, key=f"bi_{name}_{stake_name}"
                            )
                    st.session_state.strategy_configs[name]['num_buy_ins'] = new_buy_ins_dict

                else:
                    # --- Single Buy-in Buffer Mode ---
                    if is_per_stake_mode:
                        # If transitioning from per-stake to single mode, take the first value from the dict or a default.
                        first_stake_value = next(iter(num_buy_ins_config.values()), 40)
                        st.session_state.strategy_configs[name]['num_buy_ins'] = first_stake_value
                        num_buy_ins_config = first_stake_value # update local var

                    st.session_state.strategy_configs[name]['num_buy_ins'] = st.number_input(
                        "Buy-in Buffer (BIs)", value=num_buy_ins_config, min_value=1, key=f"bi_{name}",
                        help="The number of buy-ins (100 bbs) used to calculate the bankroll thresholds for moving between stakes. See the info box above for a detailed explanation of the 'sticky' logic."
                    )

                # Clean up keys from the other strategy type to avoid conflicts.
                if 'rules' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['rules']
            elif strategy_type == 'standard':
                if 'num_buy_ins' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['num_buy_ins']

                # --- Real-time Validation for Standard Strategy ---
                rules = current_config.get("rules", [])

                # --- Check for orphaned stake names in rules ---
                orphaned_stakes = set() # Stakes that exist in rules but not in the main stakes data.
                for rule in rules:
                    for stake_name in rule.get("tables", {}).keys():
                        if stake_name not in available_stakes:
                            orphaned_stakes.add(stake_name)
                if orphaned_stakes:
                    st.warning(f"**Warning:** The following stakes are referenced in this strategy's rules but no longer exist in the 'Stakes Data' tab: **{', '.join(sorted(list(orphaned_stakes)))}**. These rule parts will be ignored and removed upon saving.")

                if not rules:
                    st.warning("This strategy has no rules. Please add at least one rule below.")
                else:
                    # Check if any rule applies to the starting bankroll
                    start_br = st.session_state.start_br
                    if not any(start_br >= rule['threshold'] for rule in rules):
                        st.warning(f"No rule applies to the starting bankroll of €{start_br}. The simulation will not run for this strategy.")

                # --- Convert rules to a DataFrame for the editor ---
                # We must ensure all table mix values are strings for the data_editor, as it's configured with
                # TextColumn. Pandas might otherwise infer float types for columns with only numbers,
                # which would cause a type mismatch error in Streamlit.
                rules_list = current_config.get("rules", [])
                df_data = []
                for r in rules_list:
                    tables_str = {k: str(v) for k, v in r.get('tables', {}).items()}
                    df_data.append({'threshold': r['threshold'], **tables_str})

                rules_df = pd.DataFrame(df_data, columns=['threshold'] + available_stakes)

                # CRITICAL FIX: Convert any NaN (float) values in stake columns to empty strings.
                # This prevents a type mismatch error in st.data_editor where the data type is float
                # (due to NaN) but the column is configured as Text.
                stake_cols_in_df = [col for col in rules_df.columns if col in available_stakes]
                if stake_cols_in_df:
                    rules_df[stake_cols_in_df] = rules_df[stake_cols_in_df].astype(str).replace('nan', '')

                # --- Display the data editor for the standard strategy rules ---
                st.data_editor(
                    rules_df,
                    key=f"rules_{name}",
                    num_rows="dynamic",
                    column_config={
                        "threshold": st.column_config.NumberColumn(
                            "Bankroll Threshold (€)",
                            help="The bankroll amount at which this rule applies.",
                            min_value=1,
                            format="€ %d",
                        ),
                        **{
                            stake: st.column_config.TextColumn(
                                f"{stake} Mix",
                                help=f"Table mix for {stake}. Use a fixed ratio (e.g., 1), a percentage (e.g., '80%'), or a range (e.g., '20-40%'). Ratios are proportional (e.g., NL20: 1, NL50: 3 is a 25%/75% split).",
                            )
                            for stake in available_stakes
                        },
                    },
                )
                st.button(
                    f"Save and Sort '{name}' Rules",
                    on_click=sync_strategy_rules, args=(name,),
                    help="Click to save any changes and sort the rules by 'Bankroll Threshold'."
                )

# --- Main Logic to Run Simulation and Display Results ---

# This block runs ONLY when the "Run Simulation" button is clicked, which sets the `run_simulation` flag to True.
if st.session_state.run_simulation:
    # --- 1. Assemble the config dictionary from session_state ---
    config = {
        "STARTING_BANKROLL_EUR": st.session_state.start_br,
        "TARGET_BANKROLL": st.session_state.target_br,
        "RUIN_THRESHOLD": st.session_state.ruin_thresh,
        "NUMBER_OF_SIMULATIONS": st.session_state.num_sims,
        "TOTAL_HANDS_TO_SIMULATE": st.session_state.total_hands,
        "HANDS_PER_CHECK": st.session_state.hands_per_check,
        "RAKEBACK_PERCENTAGE": st.session_state.rb_percent / 100.0,
        "STOP_LOSS_BB": st.session_state.stop_loss_bb if st.session_state.enable_stop_loss else 0,
        "PRIOR_SAMPLE_SIZE": st.session_state.prior_sample,
        "ZERO_HANDS_INPUT_WEIGHT": st.session_state.zero_hands_weight,
        "SEED": st.session_state.seed,
        "WITHDRAWAL_SETTINGS": {
            "enabled": st.session_state.enable_withdrawals,
            "monthly_volume": st.session_state.monthly_volume_hands,
            "strategy": st.session_state.withdrawal_strategy,
            "value": st.session_state.withdrawal_value,
            "min_bankroll": st.session_state.min_br_for_withdrawal
        } if st.session_state.enable_withdrawals else {"enabled": False},
        "PLOT_PERCENTILE_LIMIT": st.session_state.plot_percentile_limit,
        # Pass Downswing Analysis constants to the engine.
        # These are defined at the top of the file.
        "DOWNSWING_DEPTH_THRESHOLDS_BB": DOWNSWING_DEPTH_THRESHOLDS_BB,
        "DOWNSWING_DURATION_THRESHOLDS_HANDS": DOWNSWING_DURATION_THRESHOLDS_HANDS,
    }

    # --- 2. Parse and validate the inputs for stakes and strategies ---
    try:
        # The data_editor state is a DataFrame, convert it to the list of dicts the engine expects.
        config["STAKES_DATA"] = st.session_state.stakes_data.to_dict('records')
        # The strategies are already in the correct dictionary format in session state.
        config["STRATEGIES_TO_RUN"] = st.session_state.strategy_configs

        # Store the final config used for this run, so we can access it for display later.
        st.session_state.config_for_display = config
        inputs_are_valid = True

    except Exception as e: # Catch any other potential errors during config assembly
        st.error(f"Error preparing simulation configuration. Details: {e}")
        inputs_are_valid = False
        st.session_state.simulation_output = None

    # --- 3. Run the simulation if inputs are valid ---
    if inputs_are_valid:
        st.header("Simulation Results")
        try:
            # --- Progress Bar Setup ---
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Starting simulation...")
            
            # Define a callback function that the engine can call to update the UI.
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)

            # THIS IS THE MAIN CALL TO THE SIMULATION ENGINE
            # Pass the callback function to the engine
            st.session_state.simulation_output = engine.run_full_analysis(config, progress_callback=update_progress)

            # Clean up the progress bar and status text after completion
            status_text.text("Simulation complete! Displaying results...")
            progress_bar.empty()
            status_text.empty()
        except ValueError as e:
            st.error(f"A configuration error prevented the simulation from running: {e}")
            st.info("Please check your strategy rules and stake definitions for issues.")
            st.session_state.simulation_output = None # Clear results on error
        except Exception as e:
            st.error("An error occurred during the simulation.")
            st.exception(e)
            st.session_state.simulation_output = None # Clear results on error

    # --- 4. Reset the run flag so the simulation doesn't run again on the next user interaction ---
    st.session_state.run_simulation = False

# This block displays the results if they exist in the session state.
# It runs on every script rerun as long as `simulation_output` is present.
if st.session_state.get("simulation_output"):
    all_results = st.session_state.simulation_output['results']
    analysis_report = st.session_state.simulation_output['analysis_report']
    diagnostic_log = st.session_state.simulation_output.get('diagnostic_log', []) # Safely get the log
    config = st.session_state.get('config_for_display', {}) # Get the config used for the run

    # Calculate a representative input win rate for the "Assigned WR Distribution" plot's label.
    weighted_input_wr = 1.5 # Default fallback
    # This logic handles cases where sample hands might be 0 to avoid division by zero.
    if config: # Ensure config exists before trying to access it
        stakes_data = config.get('STAKES_DATA', [])
        total_sample_hands = sum(s.get('sample_hands', 0) for s in stakes_data)
        if total_sample_hands > 0:
            weighted_input_wr = sum(s.get('ev_bb_per_100', 0) * s.get('sample_hands', 0) for s in stakes_data) / total_sample_hands
        elif stakes_data:
            # Fallback if no sample hands are provided
            weighted_input_wr = stakes_data[0].get('ev_bb_per_100', 1.5)


    st.header("Simulation Results")

    # --- Display the new Diagnostic Log ---
    # This is the first thing shown in the results, making it easy to spot issues.
    if diagnostic_log:
        with st.expander("🔬 Diagnostic Log", expanded=True):
            st.code("\n".join(diagnostic_log))

    # Display the AI-generated qualitative analysis if it exists.
    if analysis_report:
        with st.expander("Automated Strategy Analysis", expanded=False):
            st.markdown(analysis_report)

    st.subheader("Strategy Comparison")

    # --- Assemble and Display the Main Summary Table ---
    summary_data = []
    for name, res in all_results.items():
        summary_data.append({
            "Strategy": name,
            "Median Final BR": res['median_final_bankroll'],
            "Mode Final BR": res['final_bankroll_mode'],
            "Median Growth": res['growth_rate'],
            "Median Hands Played": res.get('median_hands_played', 0),
            "Median Profit (Play)": res.get('median_profit_from_play_eur', 0.0),
            "Median Total Withdrawn": res.get('median_total_withdrawn', 0.0),
            "Median Total Return": res.get('median_total_return', 0.0),
            "Median Rakeback": res.get('median_rakeback_eur', 0.0),
            "Risk of Ruin (%)": res['risk_of_ruin'],
            "Target Prob (%)": res['target_prob'],
            "5th %ile BR": res['p5'],
            "P95 Max Downswing": res['p95_max_downswing']
        })
    summary_df = pd.DataFrame(summary_data)

    # Use st.dataframe with style formatting for a clean, professional look.
    st.dataframe(
        summary_df.style.format({
            "Median Final BR": "€{:,.2f}", "Mode Final BR": "€{:,.2f}",
            "Median Growth": "{:.2%}", "Median Hands Played": "{:,.0f}",
            "Median Profit (Play)": "€{:,.2f}", "Median Total Withdrawn": "€{:,.2f}",
            "Median Total Return": "€{:,.2f}", "Median Rakeback": "€{:,.2f}", "Risk of Ruin (%)": "{:.2f}%",
            "Target Prob (%)": "{:.2f}%", "5th %ile BR": "€{:,.2f}",
            "P95 Max Downswing": "€{:,.2f}"
        }).hide(axis="index"),
        column_config={
            "Strategy": st.column_config.TextColumn(
                "Strategy",
                help="The name of the bankroll management strategy."
            ),
            "Median Final BR": st.column_config.TextColumn(
                "Median Final BR",
                help="The median (50th percentile) final bankroll across all simulations. This value includes both profit from play and rakeback."
            ),
            "Mode Final BR": st.column_config.TextColumn(
                "Mode Final BR",
                help="The most frequently occurring final bankroll outcome, calculated using Kernel Density Estimation."
            ),
            "Median Growth": st.column_config.TextColumn(
                "Median Growth",
                help="The median percentage growth from the starting bankroll."
            ),
            "Median Hands Played": st.column_config.TextColumn(
                "Median Hands Played",
                help="The median number of hands played. This can be lower than the 'Total Hands to Simulate' if a stop-loss is frequently triggered."
            ),
            "Median Profit (Play)": st.column_config.TextColumn(
                "Median Profit (Play)",
                help="The median profit from gameplay only, excluding rakeback. This shows how much was won or lost at the tables."
            ),
            "Median Total Withdrawn": st.column_config.TextColumn(
                "Median Total Withdrawn",
                help="The median amount of money withdrawn over the entire simulation. This represents the typical income generated by the strategy."
            ),
            "Median Total Return": st.column_config.TextColumn(
                "Median Total Return",
                help="The median total value generated. Calculated as: (Final Bankroll - Starting Bankroll) + Total Withdrawn."
            ),
            "Median Rakeback": st.column_config.TextColumn(
                "Median Rakeback",
                help="The median amount of rakeback earned in Euros. Compare this to 'Median Profit (Play)' to see how much the strategy relies on rakeback."
            ),
            "Risk of Ruin (%)": st.column_config.TextColumn(
                "Risk of Ruin (%)",
                help="The percentage of simulations where the bankroll dropped to or below the 'Ruin Threshold'."
            ),
            "Target Prob (%)": st.column_config.TextColumn(
                "Target Prob (%)",
                help="The percentage of simulations where the bankroll reached or exceeded the 'Target Bankroll' at any point."
            ),
            "5th %ile BR": st.column_config.TextColumn(
                "5th %ile BR",
                help="The 5th percentile final bankroll. 95% of simulations ended with a bankroll higher than this value."
            ),
            "P95 Max Downswing": st.column_config.TextColumn(
                "P95 Max Downswing",
                help="The 95th percentile of the maximum downswing. 5% of simulations experienced a worse downswing (peak-to-trough loss) than this value."
            ),
        }
    )
    # --- Display Comparison Plots in a 2-column layout ---
    st.subheader("Strategy Comparison Visuals")

    # Create a consistent, colorblind-friendly color map to use for all comparison plots.
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("###### Median Bankroll Progression")
        fig = engine.plot_median_progression_comparison(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)
    with comp_col2:
        st.markdown("###### Final Bankroll Distribution", help="This chart shows the full range of outcomes for each strategy. A taller, narrower peak indicates more consistent results. A wider, flatter curve with a long tail to the right indicates higher risk but also higher reward potential.")
        fig = engine.plot_final_bankroll_comparison(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)

    # --- Display new comparison plots in a second 2-column layout ---
    comp_col3, comp_col4 = st.columns(2)
    with comp_col3:
        st.markdown("###### Psychological Cost: Time Spent Below Bankroll Peak", help="This chart shows the median percentage of hands a strategy spends 'underwater' (with a bankroll below a previous all-time high). A lower percentage indicates a smoother, less stressful journey.")
        fig = engine.plot_time_underwater_comparison(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)
    with comp_col4:
        st.markdown("###### Risk vs. Reward Analysis", help="This scatter plot shows the trade-off between risk (X-axis) and reward (Y-axis). The ideal strategy is in the top-left corner (low risk, high reward). Strategies in the bottom-right are clearly inferior.")
        fig = engine.plot_risk_reward_scatter(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)
    
    # --- Display new withdrawal plot if applicable, with robust error handling ---
    try:
        fig_withdrawn = engine.plot_total_withdrawn_comparison(all_results, config, color_map=color_map)
        if fig_withdrawn:
            comp_col5, _ = st.columns(2)
            with comp_col5:
                st.markdown("###### Income Generation: Median Total Withdrawn", help="This chart shows the median total amount of money withdrawn over the course of the simulation for each strategy. It's a direct measure of the income-generating potential of a strategy.")
                st.pyplot(fig_withdrawn)
                plt.close(fig_withdrawn)
    except Exception as e:
        st.error("A critical error occurred while generating the 'Total Withdrawn' comparison plot. The application would have hung here.")
        st.exception(e)

def display_detailed_strategy_results(strategy_name, result, config, color_map, weighted_input_wr):
    """
    Helper function to display the full detailed analysis for a single strategy.
    This function is called in a loop to render the results for each strategy,
    which keeps the main part of the script clean and avoids code duplication.
    """
    # --- DEBUGGING: Scorched Earth Approach ---
    # This function is temporarily simplified to its bare minimum to isolate the hang.
    # We will only display a success message inside an expander. If this renders,
    # it proves the problem is with one of the components that was previously inside this function.
    with st.expander(f"Detailed Analysis for: {strategy_name}", expanded=True):
        st.success(f"Successfully started rendering the detailed report for **{strategy_name}**.")
        st.write("If you can see this, the hang is caused by one of the `st.metric`, `st.dataframe`, or `st.pyplot` calls that were previously here.")

    # --- DEBUGGING: The for loop below is temporarily disabled to isolate the hang. ---
    # If the PDF Download button appears after this change, we know the problem is
    # within the for loop or the display_detailed_strategy_results function.
    st.warning("DEBUG: The detailed analysis section is temporarily disabled to isolate a UI hang.")

    # --- PDF Download Button ---
    st.subheader("Download Full Report")
    # The PDF generation can be slow, so it's wrapped in a spinner to give the user feedback.
    with st.spinner("Generating PDF report..."):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_buffer = engine.generate_pdf_report(all_results, analysis_report, config, timestamp)
        st.download_button(
            label="Download Full PDF Report",
            data=pdf_buffer,
            file_name=f"simulation_report_{timestamp}.pdf",
            mime="application/pdf"
        )