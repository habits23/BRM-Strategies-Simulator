import streamlit as st
import pandas as pd
import datetime
import traceback
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

# Import the actual simulation engine we just built
import simulation_engine as engine

# --- Default Data for First Run ---
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

st.set_page_config(layout="wide", page_title="Poker Bankroll Simulator")

st.title("Poker Bankroll Simulator")
st.write("An interactive tool to simulate poker bankroll progression based on your data and strategies. Based on the logic from `Final BR Simulator v1_5.py`.")

with st.expander("Need Help? Click here for the User Guide"):
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
if 'plot_percentile_limit' not in st.session_state: st.session_state.plot_percentile_limit = 99

if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
    st.session_state.simulation_output = None

if 'stakes_data' not in st.session_state:
    st.session_state.stakes_data = DEFAULT_STAKES_DATA

if 'strategy_configs' not in st.session_state:
    st.session_state.strategy_configs = DEFAULT_STRATEGIES.copy()

def click_run_button():
    """Callback function to set the simulation flag when the button is clicked."""
    st.session_state.run_simulation = True
    st.session_state.simulation_output = None # Clear old results when a new run is requested

def setup_sanity_check():
    """Callback to load a simple config for validating the simulation engine."""
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
    st.toast("Sanity Check configuration loaded. You can now click 'Run Simulation'.", icon="🔬")

def add_strategy():
    """Callback to add a new, blank strategy with a unique name."""
    i = st.session_state.get('strategy_counter', 0) + 1
    while True:
        new_name = f"New Strategy {i}"
        if new_name not in st.session_state.strategy_configs:
            st.session_state.strategy_counter = i
            break
        i += 1

    # Use the first available stake as a sensible default
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
    """Callback to remove a strategy."""
    if name_to_remove in st.session_state.strategy_configs:
        del st.session_state.strategy_configs[name_to_remove]

def clone_strategy(name_to_clone):
    """Callback to clone a strategy with a unique name."""
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
    This function correctly handles the dictionary of changes provided by Streamlit's
    on_change callback for st.data_editor.
    """
    editor_state = st.session_state.stakes_data_editor

    # Make a copy of the original DataFrame to apply changes to
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

    # --- Sorting Logic ---
    df['bb_size'] = pd.to_numeric(df['bb_size'], errors='coerce')
    df = df.sort_values(by='bb_size', ascending=True, na_position='last').reset_index(drop=True)

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
    """Callback to sync a strategy's data editor state back to the strategy config."""
    edits = st.session_state[f"rules_{strategy_name}"]
    original_rules = st.session_state.strategy_configs[strategy_name].get("rules", [])

    available_stakes = [
        name for name in st.session_state.stakes_data['name']
        if pd.notna(name) and str(name).strip()
    ]
    expected_columns = ['threshold'] + available_stakes

    # Convert the original rules to a DataFrame to work with
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

    # --- Sorting Logic ---
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')
    df = df.sort_values(by='threshold', ascending=False, na_position='last').reset_index(drop=True)

    # Convert the modified DataFrame back into the list-of-dicts format
    new_rules = []
    for _, row in df.iterrows():
        threshold_val = row.get('threshold')
        if pd.isna(threshold_val) or threshold_val <= 0:
            continue
        tables = {}
        for stake in available_stakes:
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
st.sidebar.header("Simulation Parameters")

st.sidebar.button(
    "**Run Simulation**",
    on_click=click_run_button,
    use_container_width=True,
    type="primary",
    help="Click to run the simulation with the current settings."
)

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
    import random
    st.session_state.seed = random.randint(1, 1_000_000)

st.sidebar.caption("Random Seed")
seed_col1, seed_col2 = st.sidebar.columns([3, 1])
with seed_col1:
    st.number_input("Random Seed", label_visibility="collapsed", step=1, help="A fixed number that ensures the exact same random results every time. Change it to get a different set of random outcomes.", key="seed")
with seed_col2:
    st.button("🎲", on_click=randomize_seed, help="Generate a new random seed.", use_container_width=True)

st.sidebar.header("Save & Load Configuration")

def get_full_config_as_json():
    """Gathers all relevant session state into a dictionary and returns it as a JSON string."""
    config = {
        "parameters": {
            "start_br": st.session_state.start_br, "target_br": st.session_state.target_br,
            "ruin_thresh": st.session_state.ruin_thresh, "num_sims": st.session_state.num_sims, "total_hands": st.session_state.total_hands,
            "hands_per_check": st.session_state.hands_per_check,
            "rb_percent": st.session_state.rb_percent,
            "enable_stop_loss": st.session_state.enable_stop_loss, "stop_loss_bb": st.session_state.stop_loss_bb,
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
    """Callback to load a configuration from an uploaded JSON file stored in session state."""
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
            st.session_state.stakes_data = pd.DataFrame(loaded_data["stakes_data"])
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
st.header("Player & Strategy Configuration")

tab1, tab2 = st.tabs(["Stakes Data", "Bankroll Management Strategies"])

with tab1:
    st.subheader("Stakes Data")
    st.write("Enter your performance statistics for each stake you play. You can add or remove rows.")
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

    # Get the list of available stake names from the stakes data tab
    # This is crucial for creating the columns in the data editor
    # We filter out any empty/NaN names that can occur when a user adds a new row.
    available_stakes = [
        name for name in st.session_state.stakes_data['name']
        if pd.notna(name) and str(name).strip()
    ]
    strategy_names = list(st.session_state.strategy_configs.keys())

    for name in strategy_names:
        if name not in st.session_state.strategy_configs:
            continue # Skip if it was just removed

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

            # Update state if name changed
            if new_name != name:
                if not new_name.strip():
                    st.warning("Strategy name cannot be empty.")
                elif new_name in st.session_state.strategy_configs:
                    st.warning(f"A strategy named '{new_name}' already exists. Please choose a unique name.")
                else:
                    st.session_state.strategy_configs[new_name] = st.session_state.strategy_configs.pop(name)
                    st.rerun()

            st.session_state.strategy_configs[name]['type'] = strategy_type

            # --- Row 2: Type-specific inputs ---
            if strategy_type == 'hysteresis':
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

                # Determine if we are in per-stake mode based on the data type
                is_per_stake_mode = isinstance(num_buy_ins_config, dict)

                use_per_stake_checkbox = st.checkbox(
                    "Use per-stake buy-in buffers",
                    value=is_per_stake_mode,
                    key=f"per_stake_cb_{name}"
                )

                if use_per_stake_checkbox:
                    # --- Per-Stake Buy-in Buffer Mode ---
                    if not is_per_stake_mode:
                        # Transitioning from single to per-stake: initialize dict
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
                        # Transitioning from per-stake to single: take first value or default
                        first_stake_value = next(iter(num_buy_ins_config.values()), 40)
                        st.session_state.strategy_configs[name]['num_buy_ins'] = first_stake_value
                        num_buy_ins_config = first_stake_value # update local var

                    st.session_state.strategy_configs[name]['num_buy_ins'] = st.number_input(
                        "Buy-in Buffer (BIs)", value=num_buy_ins_config, min_value=1, key=f"bi_{name}",
                        help="The number of buy-ins (100 bbs) used to calculate the bankroll thresholds for moving between stakes. See the info box above for a detailed explanation of the 'sticky' logic."
                    )

                if 'rules' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['rules']
            elif strategy_type == 'standard':
                # Clean up keys from the other strategy type
                if 'num_buy_ins' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['num_buy_ins']

                # --- Real-time Validation for Standard Strategy ---
                rules = current_config.get("rules", [])

                # --- Check for orphaned stake names in rules ---
                orphaned_stakes = set()
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
                # We must ensure all table mix values are strings for the data_editor,
                # as it's configured with TextColumn. Pandas might otherwise infer
                # float types for columns with only numbers, causing a type mismatch.
                rules_list = current_config.get("rules", [])
                df_data = []
                for r in rules_list:
                    tables_str = {k: str(v) for k, v in r.get('tables', {}).items()}
                    df_data.append({'threshold': r['threshold'], **tables_str})

                rules_df = pd.DataFrame(df_data, columns=['threshold'] + available_stakes)

                # CRITICAL FIX: Convert any NaN (float) values in stake columns to empty
                # strings. This prevents a type mismatch error in st.data_editor where
                # the data type is float (due to NaN) but the column is configured as Text.
                stake_cols_in_df = [col for col in rules_df.columns if col in available_stakes]
                if stake_cols_in_df:
                    rules_df[stake_cols_in_df] = rules_df[stake_cols_in_df].astype(str).replace('nan', '')

                # --- Display the data editor ---
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

# This block runs ONLY when the "Run Simulation" button is clicked
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
        "PLOT_PERCENTILE_LIMIT": st.session_state.plot_percentile_limit,
    }

    # --- 2. Parse and validate the inputs for stakes and strategies ---
    try:
        # The data_editor state is a DataFrame, convert it to the list of dicts the engine expects.
        config["STAKES_DATA"] = st.session_state.stakes_data.to_dict('records')

        # The strategies are now directly in the correct dictionary format. No more parsing needed!
        config["STRATEGIES_TO_RUN"] = st.session_state.strategy_configs

        # Also store the config used for this run, so we can access it for display later
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
            with st.spinner("Running thousands of simulations... this may take a moment."):
                # THIS IS WHERE WE CALL OUR REFACTORED ENGINE
                # Store results in session state so they persist across reruns
                st.session_state.simulation_output = engine.run_full_analysis(config)
        except ValueError as e:
            st.error(f"A configuration error prevented the simulation from running: {e}")
            st.info("Please check your strategy rules and stake definitions for issues.")
            st.session_state.simulation_output = None # Clear results on error
        except Exception as e:
            st.error("An error occurred during the simulation.")
            st.exception(e)
            st.session_state.simulation_output = None # Clear results on error

    # --- 4. Reset the run flag so it doesn't run again on the next interaction ---
    st.session_state.run_simulation = False

# This block displays the results if they exist in the session state
if st.session_state.get("simulation_output"):
    all_results = st.session_state.simulation_output['results']
    analysis_report = st.session_state.simulation_output['analysis_report']
    config = st.session_state.get('config_for_display', {}) # Get the config used for the run

    # Calculate a representative input win rate for the plot's label
    weighted_input_wr = 1.5 # Default fallback
    if config: # Ensure config exists before trying to access it
        stakes_data = config.get('STAKES_DATA', [])
        total_sample_hands = sum(s.get('sample_hands', 0) for s in stakes_data)
        if total_sample_hands > 0:
            weighted_input_wr = sum(s.get('ev_bb_per_100', 0) * s.get('sample_hands', 0) for s in stakes_data) / total_sample_hands
        elif stakes_data:
            # Fallback if no sample hands are provided
            weighted_input_wr = stakes_data[0].get('ev_bb_per_100', 1.5)


    st.header("Simulation Results")

    if analysis_report:
        with st.expander("Automated Strategy Analysis", expanded=False):
            st.markdown(analysis_report)

    st.subheader("Strategy Comparison")

    # --- Display Summary Table ---
    summary_data = []
    for name, res in all_results.items():
        summary_data.append({
            "Strategy": name,
            "Median Final BR": res['median_final_bankroll'],
            "Mode Final BR": res['final_bankroll_mode'],
            "Median Growth": res['growth_rate'],
            "Median Hands Played": res.get('median_hands_played', 0),
            "Median Profit (Play)": res.get('median_profit_from_play_eur', 0.0),
            "Median Rakeback": res.get('median_rakeback_eur', 0.0),
            "Risk of Ruin (%)": res['risk_of_ruin'],
            "Target Prob (%)": res['target_prob'],
            "5th %ile BR": res['p5'],
            "P95 Max Downswing": res['p95_max_downswing']
        })
    summary_df = pd.DataFrame(summary_data)

    st.dataframe(
        summary_df.style.format({
            "Median Final BR": "€{:,.2f}", "Mode Final BR": "€{:,.2f}",
            "Median Growth": "{:.2%}", "Median Hands Played": "{:,.0f}", "Median Profit (Play)": "€{:,.2f}", "Median Rakeback": "€{:,.2f}", "Risk of Ruin (%)": "{:.2f}%",
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

    # Create a consistent, colorblind-friendly color map for all comparison plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("###### Median Bankroll Progression")
        fig = engine.reporting.plot_median_progression_comparison(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)
    with comp_col2:
        st.markdown("###### Final Bankroll Distribution", help="This chart shows the full range of outcomes for each strategy. A taller, narrower peak indicates more consistent results. A wider, flatter curve with a long tail to the right indicates higher risk but also higher reward potential.")
        fig = engine.reporting.plot_final_bankroll_comparison(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)

    # --- Display new comparison plots in a second 2-column layout ---
    comp_col3, comp_col4 = st.columns(2)
    with comp_col3:
        st.markdown("###### Psychological Cost: Time Spent Below Bankroll Peak", help="This chart shows the median percentage of hands a strategy spends 'underwater' (with a bankroll below a previous all-time high). A lower percentage indicates a smoother, less stressful journey.")
        fig = engine.reporting.plot_time_underwater_comparison(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)
    with comp_col4:
        st.markdown("###### Risk vs. Reward Analysis", help="This scatter plot shows the trade-off between risk (X-axis) and reward (Y-axis). The ideal strategy is in the top-left corner (low risk, high reward). Strategies in the bottom-right are clearly inferior.")
        fig = engine.reporting.plot_risk_reward_scatter(all_results, config, color_map=color_map)
        st.pyplot(fig)
        plt.close(fig)


    # --- Display Detailed Results for Each Strategy ---
    for strategy_name, result in all_results.items():
        with st.expander(f"Detailed Analysis for: {strategy_name}", expanded=False):
            # --- Special Sanity Check Analysis Box ---
            if strategy_name == "Sanity Check (NL20 Only)":
                with st.container(border=True):
                    st.subheader("🔬 Sanity Check Analysis")
                    start_br = config['STARTING_BANKROLL_EUR']
                    total_hands = config['TOTAL_HANDS_TO_SIMULATE']
                    stake_data = config['STAKES_DATA'][0]
                    ev_wr = stake_data['ev_bb_per_100']
                    std_dev = stake_data['std_dev_per_100']
                    bb_size = stake_data['bb_size']

                    # Analytical calculations
                    expected_profit = (total_hands / 100) * ev_wr * bb_size
                    expected_final_br = start_br + expected_profit
                    expected_std_dev_eur = (std_dev / np.sqrt(100)) * np.sqrt(total_hands) * bb_size

                    # Actual results from simulation
                    actual_median_br = result['median_final_bankroll']
                    actual_std_dev_br = np.std(result['final_bankrolls'])

                    # Calculate percentage differences
                    median_diff_pct = ((actual_median_br - expected_final_br) / expected_final_br) * 100 if expected_final_br != 0 else 0
                    std_dev_diff_pct = ((actual_std_dev_br - expected_std_dev_eur) / expected_std_dev_eur) * 100 if expected_std_dev_eur != 0 else 0

                    st.markdown("This mode compares the simulation's output against known mathematical formulas for a simple, single-stake scenario. The 'Actual' values from the simulation should be very close to the 'Expected' values calculated analytically.")
                    col1, col2 = st.columns(2)
                    col1.metric("Expected Median Final Bankroll", f"€{expected_final_br:,.2f}", help="Calculated as: Start BR + (EV Win Rate * Total Hands * bb Size)")
                    col1.metric("Actual Median Final Bankroll", f"€{actual_median_br:,.2f}", delta=f"€{actual_median_br - expected_final_br:,.2f} ({median_diff_pct:+.2f}%)")
                    col2.metric("Expected Std. Dev. of Final BR", f"€{expected_std_dev_eur:,.2f}", help="Calculated as: (Std Dev / 10) * sqrt(Total Hands) * bb Size")
                    col2.metric("Actual Std. Dev. of Final BR", f"€{actual_std_dev_br:,.2f}", delta=f"€{actual_std_dev_br - expected_std_dev_eur:,.2f} ({std_dev_diff_pct:+.2f}%)")

            st.subheader(f"Key Metrics for '{strategy_name}'")
            num_metrics = 9 if config.get("STOP_LOSS_BB", 0) > 0 else 8
            metric_cols = st.columns(num_metrics)
            col1, col2, col3, col4, col5, col6, col7, col8 = metric_cols[:8]
            col1.metric("Median Final Bankroll", f"€{result['median_final_bankroll']:,.2f}", help="The median (50th percentile) final bankroll, including both profit from play and rakeback.")
            col2.metric("Median Hands Played", f"{result.get('median_hands_played', 0):,.0f}", help="The median number of hands played. This can be lower than the total if a stop-loss is frequently triggered.")
            col3.metric("Median Profit (Play)", f"€{result.get('median_profit_from_play_eur', 0.0):,.2f}", help="The median profit from gameplay only, excluding rakeback. Compare this to Median Rakeback to see the impact of rakeback.")
            col4.metric("Median Rakeback", f"€{result.get('median_rakeback_eur', 0.0):,.2f}", help="The median amount of total rakeback earned over the course of a simulation. This is extra profit on top of what you win at the tables.")
            col5.metric("Risk of Ruin", f"{result['risk_of_ruin']:.2f}%", help="The percentage of simulations where the bankroll dropped to or below the 'Ruin Threshold'.")
            col6.metric("Target Probability", f"{result['target_prob']:.2f}%", help="The percentage of simulations where the bankroll reached or exceeded the 'Target Bankroll' at any point.")
            col7.metric("Median Downswing", f"€{result['median_max_downswing']:,.2f}", help="The median of the maximum peak-to-trough loss experienced in each simulation. Represents a typical worst-case downswing.")
            col8.metric("95th Pct. Downswing", f"€{result['p95_max_downswing']:,.2f}", help="The 95th percentile of the maximum downswing. 5% of simulations experienced a worse downswing (peak-to-trough loss) than this value.")
            if num_metrics == 9:
                metric_cols[8].metric("Median Stop-Losses", f"{result.get('median_stop_losses', 0):.1f}", help="The median number of times the stop-loss was triggered per simulation run.")

            st.subheader("Visual Analysis")
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                st.markdown("###### Bankroll Progression")
                fig = engine.reporting.plot_strategy_progression(result['bankroll_histories'], result['hands_histories'], strategy_name, config)
                st.pyplot(fig)
                plt.close(fig)
            with row1_col2:
                st.markdown("###### Final Bankroll Distribution")
                fig = engine.reporting.plot_final_bankroll_distribution(result['final_bankrolls'], result, strategy_name, config, color_map=color_map)
                st.pyplot(fig)
                plt.close(fig)

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                st.markdown("###### Distribution of Assigned Luck (WR)", help=(
                    "This chart shows the distribution of 'luck' (the pre-assigned win rate) across all simulations.\n\n"
                    "**Why is the distribution so wide?**\n\n"
                    "The width of this 'luck' distribution is determined by your **'Std Dev (bb/100)'** and **'Sample Hands'** inputs. A high standard deviation and/or a low sample size creates more uncertainty about your true win rate. The simulation reflects this by generating a wider range of possible outcomes (both lucky and unlucky).\n\n"
                    "**How to read this chart:**\n- **Blue Line:** Your average win rate, based on your inputs.\n- **Red Line:** The 'luck' of the specific simulation run that resulted in the median final bankroll."
                ))
                if 'avg_assigned_wr_per_sim' in result:
                    fig = engine.reporting.plot_assigned_wr_distribution(
                        result['avg_assigned_wr_per_sim'],
                        result['median_run_assigned_wr'],
                        weighted_input_wr,
                        strategy_name
                    )
                    st.pyplot(fig)
                    plt.close(fig)
            with row2_col2:
                st.markdown("###### Maximum Downswing Distribution", help="This chart shows the distribution of the largest single peak-to-trough loss (a downswing) experienced in each simulation. It gives a clear picture of the potential 'pain' or volatility of a strategy.")
                if 'max_downswings' in result:
                    fig = engine.reporting.plot_max_downswing_distribution(result['max_downswings'], result, strategy_name, color_map=color_map)
                    st.pyplot(fig)
                    plt.close(fig)

            st.subheader("Key Strategy Insights")
            st.markdown("_For a full breakdown, please download the PDF report._")

            # Check for a dominant stake to provide a key insight
            if result.get('hands_distribution_pct'):
                hands_dist = result['hands_distribution_pct']
                if hands_dist: # Ensure it's not empty
                    dominant_stake = max(hands_dist, key=hands_dist.get)
                    dominant_pct = hands_dist[dominant_stake]
                    if dominant_pct > 75:  # Threshold for being "dominant"
                        st.info(f"**Key Insight:** This strategy spent the vast majority of its time at **{dominant_stake}** ({dominant_pct:.1f}% of all hands played). The results are therefore heavily influenced by the performance at this single stake.")


            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    "**Hands Distribution**",
                    help=(
                        "This section shows three key metrics for each stake:\n\n"
                        "1.  **Percentage:** The share of total hands played at this stake.\n\n"
                        "2.  **Avg. WR:** The average win rate the simulation used for this stake across all runs. This includes the 'luck' factor, so it will differ from your input. It represents the average 'true skill' assigned to players at this stake.\n\n"
                        "3.  **Trust:** A percentage showing how much the model 'trusts' your input `EV Win Rate`. It's calculated from your `Sample Hands` vs. the `Prior Sample Size`. A higher trust percentage means the model is more confident in your data and will apply less long-term luck (variance) to your win rate."
                    ),
                )
                if result.get('hands_distribution_pct'):
                    # Create a map for easy lookup of sample hands
                    stakes_data_list = config.get('STAKES_DATA', [])
                    sample_hands_map = {s['name']: s.get('sample_hands', 0) for s in stakes_data_list}
                    prior_sample_size = config.get('PRIOR_SAMPLE_SIZE', 50000)

                    stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
                    sorted_stakes = sorted(result['hands_distribution_pct'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')))
                    avg_win_rates = result.get('average_assigned_win_rates', {})

                    for stake, pct in sorted_stakes:
                        if pct > 0.01:
                            # Calculate data weight (trust factor)
                            sample_hands = sample_hands_map.get(stake, 0)
                            trust_factor = 0.0
                            if sample_hands > 0 and (sample_hands + prior_sample_size) > 0:
                                trust_factor = sample_hands / (sample_hands + prior_sample_size)

                            details_parts = []
                            if stake in avg_win_rates:
                                details_parts.append(f"Avg. WR: {avg_win_rates[stake]:.2f}")
                            details_parts.append(f"Trust: {trust_factor:.0%}")

                            details_str = f" ({', '.join(details_parts)})"
                            st.write(f"- {stake}: {pct:.2f}%{details_str}")
                else:
                    st.write("No hands played.")

            with col2:
                st.markdown("**Final Stake**", help="The percentage of simulations that finished with this as their highest active stake.")
                if result.get('final_highest_stake_distribution'):
                    stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
                    sorted_dist = sorted(result['final_highest_stake_distribution'].items(), key=lambda item: stake_order_map.get(item[0], -1), reverse=True)
                    for stake, pct in sorted_dist:
                        if pct > 0.01:
                            display_stake = "Below Min. Threshold / Ruined" if stake == "No Play" else stake
                            st.write(f"- {display_stake}: {pct:.2f}%")
                else:
                    st.write("N/A")

            # --- Risk of Demotion Section ---
            if result.get('risk_of_demotion'):
                st.markdown("---")
                st.markdown("**Risk of Demotion**", help="The probability of being demoted from a stake after you've reached it. Calculated as: (Simulations demoted from Stake X) / (Simulations that ever reached Stake X). A high percentage indicates an unstable strategy.")
                stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
                sorted_demotions = sorted(result['risk_of_demotion'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')), reverse=True)

                demotion_markdown = ""
                for stake, data in sorted_demotions:
                    if data['reached_count'] > 0: # Only show relevant stakes
                        demotion_markdown += f"- From **{stake}**: **{data['prob']:.2f}%** _(of {int(data['reached_count']):,} sims)_\n"
                st.markdown(demotion_markdown)

            # --- Percentile Win Rate Analysis Section ---
            if result.get('percentile_win_rates'):
                st.markdown("---")
                st.markdown(
                    "**Percentile Win Rate Analysis (bb/100)**",
                    help=(
                        "This section shows the win rates for simulations that ended near key percentiles. This helps explain *why* the final bankrolls landed where they did.\n\n"
                        "- **Assigned WR:** The 'true' win rate (Skill + Long-Term Luck) assigned to this simulation run. It's influenced by your EV Win Rate, Sample Hands, and Std Dev. It models if a player is on a career-long heater or cooler.\n\n"
                        "- **Play WR:** The actual win rate realized from gameplay after adding short-term (session) variance. It's influenced by the Assigned WR (the baseline), Std Dev (magnitude of swings), and Hands per Bankroll Check (session length).\n\n"
                        "- **Rakeback WR:** The effective win rate gained from rakeback.\n\n"
                        "- **Variance Impact:** The difference between Play WR and Assigned WR, showing the net effect of short-term variance over the entire simulation."
                    )
                )
                st.caption(
                    "**Important:** The 'Median' column shows stats for the single simulation that had the median *final bankroll*, not the median of all stats. "
                    "The `Realized WR (Play)` includes short-term variance, so it will naturally differ from the `Assigned WR` even for this median-outcome run. "
                    "This is expected statistical noise."
                )

                percentile_wrs = result.get('percentile_win_rates', {})
                # Show a 7-number summary to match the data calculated in the analysis module
                percentiles_to_show = {
                    "2.5th": "2.5th Percentile",
                    "5th": "5th Percentile",
                    "25th": "25th Percentile",
                    "Median": "Median Percentile",
                    "75th": "75th Percentile",
                    "95th": "95th Percentile",
                    "97.5th": "97.5th Percentile"
                }

                # Filter to only show percentiles that were actually calculated
                # This also preserves the correct order from the dictionary definition
                available_percentiles = {
                    short: long for short, long in percentiles_to_show.items() if long in percentile_wrs
                }

                if not available_percentiles:
                    st.write("No percentile data available.")
                else:
                    cols = st.columns(len(available_percentiles))
                    for i, (short_name, long_name) in enumerate(available_percentiles.items()):
                        with cols[i]:
                            st.markdown(f"**{short_name} %ile**")
                            data = percentile_wrs.get(long_name, {})
                            st.metric(label="Assigned WR", value=f"{data.get('Assigned WR', 'N/A')}", help="The 'true' win rate (Skill + Long-Term Luck) assigned to this simulation run. It's influenced by your EV Win Rate, Sample Hands, and Std Dev.")
                            st.metric(label="Play WR", value=f"{data.get('Realized WR (Play)', 'N/A')}", help="The actual win rate realized from gameplay after adding short-term (session) variance. It's influenced by: the Assigned WR (the baseline), Std Dev (magnitude of swings), and Hands per Bankroll Check (session length).")
                            st.metric(label="Rakeback WR", value=f"{data.get('Rakeback (bb/100)', 'N/A')}", help="The effective win rate gained from rakeback.")
                            st.metric(label="Variance Impact", value=f"{data.get('Variance Impact', 'N/A')}", help="The difference between Play WR and Assigned WR, showing the net effect of short-term variance.")


    # --- PDF Download Button ---
    st.subheader("Download Full Report")
    with st.spinner("Generating PDF report..."):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_buffer = engine.generate_pdf_report(all_results, analysis_report, config, timestamp)
        st.download_button(
            label="Download Full PDF Report",
            data=pdf_buffer,
            file_name=f"simulation_report_{timestamp}.pdf",
            mime="application/pdf"
        )