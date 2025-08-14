import streamlit as st
import pandas as pd
import datetime
import traceback
import json
import copy

# Import the actual simulation engine we just built
import simulation_engine as engine

# --- Default Data for First Run ---
DEFAULT_STAKES_DATA = pd.DataFrame([
    {
        "name": "NL20", "bb_size": 0.20, "bb_per_100": 8.0, "ev_bb_per_100": 8.0, "std_dev_per_100": 90.0,
        "sample_hands": 40000, "win_rate_drop": 0.0, "rake_bb_per_100": 16.0
    },
    {
        "name": "NL50", "bb_size": 0.50, "bb_per_100": 4.0, "ev_bb_per_100": 4.0, "std_dev_per_100": 100.0,
        "sample_hands": 25000, "win_rate_drop": 2.0, "rake_bb_per_100": 10.0
    },
    {
        "name": "NL100", "bb_size": 1.00, "bb_per_100": 2.5, "ev_bb_per_100": 2.5, "std_dev_per_100": 100.0,
        "sample_hands": 20000, "win_rate_drop": 1.5, "rake_bb_per_100": 7.0
    },
    {
        "name": "NL200", "bb_size": 2.00, "bb_per_100": 2.0, "ev_bb_per_100": 2.0, "std_dev_per_100": 100.0,
        "sample_hands": 0, "win_rate_drop": 1.0, "rake_bb_per_100": 5.0
    },
])

DEFAULT_STRATEGIES = {
    "Balanced (50 BI Progressive)": { # 50 BI for 100% mix, ~35-40 BI for shot-taking
        "type": "standard",
        "rules": [
            {"threshold": 10000, "tables": {"NL200": "100%"}},
            {"threshold": 7500, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 5000, "tables": {"NL100": "100%"}},
            {"threshold": 3750, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 2500, "tables": {"NL50": "100%"}},
            {"threshold": 1750, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 1000, "tables": {"NL20": "100%"}},
        ]
    },
    "Aggressive (30 BI Shot-Take)": { # 30 BI for 100% mix, ~22.5 BI for shot-taking
        "type": "standard",
        "rules": [
            {"threshold": 6000, "tables": {"NL200": "100%"}},
            {"threshold": 4500, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 3000, "tables": {"NL100": "100%"}},
            {"threshold": 2250, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 1500, "tables": {"NL50": "100%"}},
            {"threshold": 900, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 600, "tables": {"NL20": "100%"}},
        ]
    },
    "Conservative (100 BI Rule)": { # 100 BI for 100% mix, ~70-75 BI for shot-taking
        "type": "standard",
        "rules": [
            {"threshold": 20000, "tables": {"NL200": "100%"}},
            {"threshold": 15000, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 10000, "tables": {"NL100": "100%"}},
            {"threshold": 7500, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 5000, "tables": {"NL50": "100%"}},
            {"threshold": 3500, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 2000, "tables": {"NL20": "100%"}},
        ]
    },
    "Hysteresis (Sticky)": {
        "type": "hysteresis",
        "num_buy_ins": 30
    }
}

st.set_page_config(layout="wide", page_title="Poker Bankroll Simulator")

st.title("Poker Bankroll Simulator")
st.write("An interactive tool to simulate poker bankroll progression based on your data and strategies. Based on the logic from `Final BR Simulator v1_5.py`.")

# --- Session State Initialization ---
# This block ensures all necessary keys are in the session state with default values
# on the first run. This prevents Streamlit warnings about setting a widget's value
# both from its `value` parameter and from the session state.
if 'start_br' not in st.session_state:
    st.session_state.start_br = 2500
    st.session_state.target_br = 3000
    st.session_state.ruin_thresh = 750
    st.session_state.num_sims = 1000
    st.session_state.num_sessions = 100
    st.session_state.hands_per_table = 60
    st.session_state.sl_percent = 10
    st.session_state.min_tables = 3
    st.session_state.max_tables = max(5, st.session_state.min_tables)
    st.session_state.target_tables_pct = 4
    st.session_state.rb_percent = 20
    st.session_state.prior_sample = 100000
    st.session_state.zero_hands_weight = 0.5
    st.session_state.min_df = 3
    st.session_state.max_df = 30
    st.session_state.hands_for_max_df = 50000
    st.session_state.seed = 98765

if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
    st.session_state.results = None

if 'stakes_data' not in st.session_state:
    st.session_state.stakes_data = DEFAULT_STAKES_DATA

if 'strategy_configs' not in st.session_state:
    st.session_state.strategy_configs = DEFAULT_STRATEGIES.copy()

def click_run_button():
    """Callback function to set the simulation flag when the button is clicked."""
    st.session_state.run_simulation = True
    st.session_state.results = None # Clear old results when a new run is requested

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

st.sidebar.button("Run Simulation", on_click=click_run_button, use_container_width=True)

with st.sidebar.expander("General Settings", expanded=True):
    st.number_input("Starting Bankroll (€)", min_value=0, step=100, help="The amount of money you are starting with for the simulation.", key="start_br")
    st.number_input("Target Bankroll (€)", min_value=0, step=100, help="The bankroll amount you are aiming to reach. This is used to calculate 'Target Probability'.", key="target_br")
    st.number_input("Ruin Threshold (€)", min_value=0, step=50, help="If a simulation's bankroll drops to or below this value, it is considered 'ruined' and stops.", key="ruin_thresh")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Number of Simulations", min_value=10, max_value=50000, step=100, help="How many times to run the entire simulation from start to finish. Higher numbers give more accurate results but take longer. (e.g., 2000-10000)", key="num_sims")
    with col2:
        st.number_input("Sessions per Simulation", min_value=1, max_value=1000, help="How many playing sessions are in a single simulation run. This determines the time horizon of the simulation.", key="num_sessions")

with st.sidebar.expander("Session & Rakeback Settings", expanded=True):
    st.number_input("Hands per Table per Session", min_value=1, help="The average number of hands you play per table during one session.", key="hands_per_table")
    st.slider("Stop-Loss per Session (%)", 0, 100, help="The session ends early if you lose this percentage of your bankroll at the start of that session. Set to 100% to disable.", key="sl_percent")

    col3, col4 = st.columns(2)
    with col3:
        st.number_input("Min Tables", min_value=1, help="The minimum number of tables you will play in any given session. The app will pick a random number between Min and Max for each session.", key="min_tables")
    with col4:
        st.number_input("Max Tables", min_value=st.session_state.min_tables, help="The maximum number of tables you will play in any given session.", key="max_tables")

    st.number_input("Target Tables (for % display)", min_value=1, help="Used for display purposes in the PDF report to show an example table mix for strategies that use percentages.", key="target_tables_pct")
    st.slider("Rakeback (%)", 0, 100, help="The percentage of rake you get back from the poker site. This is added to your profit at the end of each session.", key="rb_percent")

with st.sidebar.expander("Advanced Statistical Settings", expanded=False):
    st.number_input("Prior Sample Size (for Bayesian model)", min_value=1000, step=1000, help="Represents the strength of the model's prior belief about win rates. A larger value means the model is more confident in its own estimates and less influenced by small sample sizes from your data.", key="prior_sample")
    st.slider("Weight for 0-Hand Stake Estimates", 0.0, 1.0, step=0.05, help="For stakes where you have no hands played, this slider balances between your manual win rate estimate (1.0) and the model's extrapolation from other stakes (0.0).", key="zero_hands_weight")
    st.number_input("Min Degrees of Freedom (t-dist)", min_value=2, help="The starting 'fatness' of the tails for the t-distribution, used for small sample sizes to model higher variance. Must be > 2.", key="min_df")
    st.number_input("Max Degrees of Freedom (t-dist)", min_value=st.session_state.min_df, help="The 'df' for very large sample sizes. As df increases, the t-distribution approaches a normal distribution.", key="max_df")
    st.number_input("Hands to Reach Max DF", min_value=1000, step=1000, help="The number of hands required to transition from Min DF to Max DF. This controls how quickly the model gains confidence in your results.", key="hands_for_max_df")

def randomize_seed():
    """Generates a new random seed."""
    import random
    st.session_state.seed = random.randint(1, 1_000_000)

seed_col1, seed_col2 = st.sidebar.columns([3, 1])
with seed_col1:
    st.number_input("Random Seed", step=1, help="A fixed number that ensures the exact same random results every time. Change it to get a different set of random outcomes.", key="seed")
with seed_col2:
    st.write(" ") # Spacer for alignment
    st.button("🎲 New", on_click=randomize_seed, help="Generate a new random seed.", use_container_width=True)

st.sidebar.header("Save & Load Configuration")

def get_full_config_as_json():
    """Gathers all relevant session state into a dictionary and returns it as a JSON string."""
    config = {
        "parameters": {
            "start_br": st.session_state.start_br, "target_br": st.session_state.target_br,
            "ruin_thresh": st.session_state.ruin_thresh, "num_sims": st.session_state.num_sims,
            "num_sessions": st.session_state.num_sessions, "hands_per_table": st.session_state.hands_per_table,
            "sl_percent": st.session_state.sl_percent, "min_tables": st.session_state.min_tables,
            "max_tables": st.session_state.max_tables, "target_tables_pct": st.session_state.target_tables_pct,
            "rb_percent": st.session_state.rb_percent, "prior_sample": st.session_state.prior_sample,
            "zero_hands_weight": st.session_state.zero_hands_weight, "min_df": st.session_state.min_df,
            "max_df": st.session_state.max_df, "hands_for_max_df": st.session_state.hands_for_max_df,
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
            return

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
        st.session_state.results = None  # Clear results from any previous run

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
                "BB Size (€)",
                help="The size of the big blind in your currency (e.g., 0.20 for NL20).",
                format="€ %.2f"
            ),
            "bb_per_100": st.column_config.NumberColumn(
                "Win Rate (bb/100)",
                help="Your actual, observed win rate at this stake, measured in big blinds per 100 hands.",
                format="%.2f"
            ),
            "ev_bb_per_100": st.column_config.NumberColumn(
                "EV Win Rate (bb/100)",
                help="Your EV-adjusted win rate (All-in Adj BB/100). This is used for the simulation's core calculations.",
                format="%.2f"
            ),
            "std_dev_per_100": st.column_config.NumberColumn(
                "Std Dev (bb/100)",
                help="Your standard deviation in bb/100 hands. A typical value for No-Limit Hold'em is between 80 and 120.",
                format="%.2f"
            ),
            "sample_hands": st.column_config.NumberColumn(
                "Sample Hands",
                help="The number of hands you have played at this stake. This informs the model's confidence in your win rate.",
                format="%d"
            ),
            "win_rate_drop": st.column_config.NumberColumn(
                "Win Rate Drop",
                help="The estimated drop in your win rate (in bb/100) when moving up from the *previous* stake. This helps the model estimate your win rate at stakes with few or no hands. The lowest stake should have a drop of 0.",
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
                        help="The number of buy-ins (100 BBs) required to move up/down stakes."
                    )

                if 'rules' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['rules']
            elif strategy_type == 'standard':
                # Clean up keys from the other strategy type
                if 'num_buy_ins' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['num_buy_ins']

                # --- Real-time Validation for Standard Strategy ---
                rules = current_config.get("rules", [])
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
                            min_value=0,
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
        "TOTAL_SESSIONS_PER_RUN": st.session_state.num_sessions,
        "HANDS_PER_TABLE_PER_SESSION": st.session_state.hands_per_table,
        "STOP_LOSS_PERCENTAGE_PER_SESSION": st.session_state.sl_percent / 100.0,
        "MIN_TABLES_PER_SESSION": st.session_state.min_tables,
        "MAX_TABLES_PER_SESSION": st.session_state.max_tables,
        "TARGET_TOTAL_TABLES_FOR_PERCENTAGES": st.session_state.target_tables_pct,
        "RAKEBACK_PERCENTAGE": st.session_state.rb_percent / 100.0,
        "PRIOR_SAMPLE_SIZE": st.session_state.prior_sample,
        "ZERO_HANDS_INPUT_WEIGHT": st.session_state.zero_hands_weight,
        "MIN_DEGREES_OF_FREEDOM": st.session_state.min_df,
        "MAX_DEGREES_OF_FREEDOM": st.session_state.max_df,
        "HANDS_FOR_MAX_DF": st.session_state.hands_for_max_df,
        "SEED": st.session_state.seed,
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
        st.session_state.results = None

    # --- 3. Run the simulation if inputs are valid ---
    if inputs_are_valid:
        st.header("Simulation Results")
        try:
            with st.spinner("Running thousands of simulations... this may take a moment."):
                # THIS IS WHERE WE CALL OUR REFACTORED ENGINE
                # Store results in session state so they persist across reruns
                st.session_state.results = engine.run_full_analysis(config)
        except ValueError as e:
            st.error(f"A configuration error prevented the simulation from running: {e}")
            st.info("Please check your strategy rules and stake definitions for issues.")
            st.session_state.results = None # Clear results on error
        except Exception as e:
            st.error("An error occurred during the simulation.")
            st.exception(e)
            st.session_state.results = None # Clear results on error

    # --- 4. Reset the run flag so it doesn't run again on the next interaction ---
    st.session_state.run_simulation = False

# This block displays the results if they exist in the session state
if st.session_state.results:
    all_results = st.session_state.results
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
    st.subheader("Strategy Comparison")

    # --- Display Summary Table ---
    summary_data = []
    for name, res in all_results.items():
        summary_data.append({
            "Strategy": name,
            "Median Final BR": res['median_final_bankroll'],
            "Mode Final BR": res['final_bankroll_mode'],
            "Median Growth": res['growth_rate'],
            "Median Rakeback": res.get('median_rakeback_eur', 0.0),
            "Risk of Ruin (%)": res['risk_of_ruin'],
            "Target Prob (%)": res['target_prob'],
            "5th %ile BR": res['p5'],
            "P95 Max Drawdown": res['p95_max_drawdown']
        })
    summary_df = pd.DataFrame(summary_data)

    st.dataframe(
        summary_df.style.format({
            "Median Final BR": "€{:,.2f}", "Mode Final BR": "€{:,.2f}",
            "Median Growth": "{:.2%}", "Median Rakeback": "€{:,.2f}", "Risk of Ruin (%)": "{:.2f}%",
            "Target Prob (%)": "{:.2f}%", "5th %ile BR": "€{:,.2f}",
            "P95 Max Drawdown": "€{:,.2f}"
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
            "Median Rakeback": st.column_config.TextColumn(
                "Median Rakeback",
                help="The median amount of rakeback earned in Euros across all simulations."
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
            "P95 Max Drawdown": st.column_config.TextColumn(
                "P95 Max Drawdown",
                help="The 95th percentile of the maximum drawdown. 5% of simulations experienced a larger peak-to-trough loss than this value."
            ),
        }
    )
    # --- Display Comparison Plot ---
    st.subheader("Median Bankroll Progression Comparison")
    plot_col_left, plot_col_main, plot_col_right = st.columns([1.0, 2.0, 1.0])
    with plot_col_main:
        st.pyplot(engine.plot_median_progression_comparison(all_results, config))

    # --- Display Detailed Results for Each Strategy ---
    for strategy_name, result in all_results.items():
        with st.expander(f"Detailed Analysis for: {strategy_name}", expanded=False):
            st.subheader(f"Key Metrics for '{strategy_name}'")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Median Final Bankroll", f"€{result['median_final_bankroll']:,.2f}", help="The median (50th percentile) final bankroll, including both profit from play and rakeback.")
            col2.metric("Median Rakeback", f"€{result.get('median_rakeback_eur', 0.0):,.2f}", help="The median amount of total rakeback earned over the course of a simulation. This is extra profit on top of what you win at the tables.")
            col3.metric("Risk of Ruin", f"{result['risk_of_ruin']:.2f}%", help="The percentage of simulations where the bankroll dropped to or below the 'Ruin Threshold'.")
            col4.metric("Target Probability", f"{result['target_prob']:.2f}%", help="The percentage of simulations where the bankroll reached or exceeded the 'Target Bankroll' at any point.")
            col5.metric("Median Max Drawdown", f"€{result['median_max_drawdown']:,.2f}", help="The median of the maximum peak-to-trough loss experienced in each simulation. Represents a typical worst-case downswing.")
            col6.metric("95th Pct. Drawdown", f"€{result['p95_max_drawdown']:,.2f}", help="The 95th percentile of the maximum drawdown. 5% of simulations experienced a larger peak-to-trough loss than this value.")

            st.subheader("Charts")
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                fig1 = engine.plot_strategy_progression(result['bankroll_histories'], result['hands_histories'], strategy_name, config)
                st.pyplot(fig1)
            with plot_col2:
                fig2 = engine.plot_final_bankroll_distribution(result['final_bankrolls'], result, strategy_name, config)
                st.pyplot(fig2)
            
            # --- New Assigned WR Distribution Chart ---
            st.markdown("---")
            st.markdown(
                "**Distribution of Assigned Luck (Win Rate)**",
                help=(
                    "This chart shows the distribution of 'luck' (the pre-assigned win rate) across all simulations.\n\n"
                    "**Why is the distribution so wide?**\n\n"
                    "The width of this 'luck' distribution is determined by your **'Std Dev (bb/100)'** and **'Sample Hands'** inputs. A high standard deviation and/or a low sample size creates more uncertainty about your true win rate. The simulation reflects this by generating a wider range of possible outcomes (both lucky and unlucky).\n\n"
                    "A value far from the center represents a simulation run where you experienced a significant, but statistically possible, streak of good or bad luck.\n\n"
                    "**How to read this chart:**\n"
                    "- **Blue Line:** Your average win rate, based on your inputs.\n"
                    "- **Red Line:** The 'luck' of the specific simulation run that resulted in the median final bankroll."
                )
            )
            if 'avg_assigned_wr_per_sim' in result:
                # Use a column layout to constrain the width of the plot, making it consistent
                # with the two charts above it. The second column is intentionally left blank.
                plot_col_luck, _ = st.columns(2)
                with plot_col_luck:
                    fig3 = engine.plot_assigned_wr_distribution(
                        result['avg_assigned_wr_per_sim'],
                        result['median_run_assigned_wr'],
                        weighted_input_wr,
                        strategy_name
                    )
                    st.pyplot(fig3)

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
                        "This section shows two key metrics:\n\n"
                        "1.  **Percentage:** The share of total hands played at this stake.\n"
                        "2.  **Avg. WR (Average Win Rate):** The average win rate the simulation used for this stake across all runs.\n\n"
                        "**Why isn't this just my input win rate?**\n\n"
                        "To be realistic, the simulation models luck and uncertainty. For each of the 1000+ simulation runs, it uses a slightly different win rate based on your sample size and a random factor to simulate running hot or cold. "
                        "The `Avg. WR` is the average of all these slightly different win rates."
                    )
                )
                if result.get('hands_distribution_pct'):
                    stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
                    sorted_stakes = sorted(result['hands_distribution_pct'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')))
                    avg_win_rates = result.get('average_assigned_win_rates', {})
                    for stake, pct in sorted_stakes:
                        if pct > 0.01:
                            wr_str = ""
                            if stake in avg_win_rates:
                                wr = avg_win_rates[stake]
                                wr_str = f" (Avg. WR: {wr:.2f} bb/100)"
                            st.write(f"- {stake}: {pct:.2f}%{wr_str}")
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
                st.markdown("**Risk of Demotion**", help="The percentage of simulations that were forced to move down after reaching a specific stake.")
                stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
                sorted_demotions = sorted(result['risk_of_demotion'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')), reverse=True)
                
                demotion_text = []
                for stake, data in sorted_demotions:
                    if data['reached_count'] > 0: # Only show relevant stakes
                        demotion_text.append(f"From **{stake}**: **{data['prob']:.2f}%** _(of {int(data['reached_count']):,} sims)_")
                st.write(" / ".join(demotion_text))

            # --- Percentile Win Rate Analysis Section ---
            if result.get('percentile_win_rates'):
                st.markdown("---")
                st.markdown(
                    "**Percentile Win Rate Analysis (bb/100)**",
                    help="Shows the win rates for simulations that ended near key percentiles. This helps explain *why* the final bankrolls landed where they did.\n\n- **Assigned WR:** The 'true' win rate the simulation assigned for this entire run (models long-term luck).\n- **Play WR:** The actual, realized win rate from gameplay after session-to-session variance.\n- **Rakeback WR:** The effective win rate gained from rakeback."
                )
                st.caption("Note: The 'Median' column shows the win rate for the simulation that had the median *final bankroll*, not necessarily median *luck*. If most runs are successful, the median run will often be one that was assigned an above-average win rate.")

                percentile_wrs = result.get('percentile_win_rates', {})
                percentiles_to_show = {
                    "5th": "5th Percentile",
                    "Median": "Median Percentile",
                    "95th": "95th Percentile"
                }

                cols = st.columns(len(percentiles_to_show))

                for i, (short_name, long_name) in enumerate(percentiles_to_show.items()):
                    with cols[i]:
                        st.markdown(f"**{short_name} %ile**")
                        if long_name in percentile_wrs:
                            data = percentile_wrs[long_name]
                            st.metric(label="Assigned WR", value=f"{data.get('Assigned WR', 'N/A')}")
                            st.metric(label="Play WR", value=f"{data.get('Realized WR (Play)', 'N/A')}")
                            st.metric(label="Rakeback WR", value=f"{data.get('Rakeback (bb/100)', 'N/A')}")


    # --- PDF Download Button ---
    st.subheader("Download Full Report")
    with st.spinner("Generating PDF report..."):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_buffer = engine.generate_pdf_report(all_results, config, timestamp)
        st.download_button(
            label="Download Full PDF Report",
            data=pdf_buffer,
            file_name=f"simulation_report_{timestamp}.pdf",
            mime="application/pdf"
        )
