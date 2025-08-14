import streamlit as st
import pandas as pd
import datetime
import traceback

# Import the actual simulation engine we just built
import simulation_engine as engine

# --- Default Data for First Run ---
DEFAULT_STAKES_DATA = pd.DataFrame([
    {
        "name": "NL20", "bb_per_100": 8.0, "ev_bb_per_100": 8.3, "std_dev_per_100": 91.4,
        "sample_hands": 77657, "bb_size": 0.20, "win_rate_drop": 0.0, "rake_bb_per_100": 15.8
    },
    {
        "name": "NL50", "bb_per_100": 4.0, "ev_bb_per_100": 4.0, "std_dev_per_100": 100.0,
        "sample_hands": 5681, "bb_size": 0.50, "win_rate_drop": 1.5, "rake_bb_per_100": 10.4
    },
    {
        "name": "NL100", "bb_per_100": 2.5, "ev_bb_per_100": 2.5, "std_dev_per_100": 100.0,
        "sample_hands": 0, "bb_size": 1.00, "win_rate_drop": 1.0, "rake_bb_per_100": 7.0
    },
])

DEFAULT_STRATEGIES = {
    "Standard Progressive": {
        "type": "standard",
        "rules": [
            {"threshold": 5000, "tables": {"NL100": "100%"}},
            {"threshold": 3500, "tables": {"NL50": "100%"}},
            {"threshold": 3000, "tables": {"NL20": "20%", "NL50": "80%"}},
            {"threshold": 2500, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 1200, "tables": {"NL20": "100%"}},
        ]
    },
    "Sticky (40 BIs)": {
        "type": "hysteresis",
        "num_buy_ins": 40
    }
}

st.set_page_config(layout="wide", page_title="Poker Bankroll Simulator")

st.title("Poker Bankroll Simulator")
st.write("An interactive tool to simulate poker bankroll progression based on your data and strategies. Based on the logic from `Final BR Simulator v1_5.py`.")

# --- Session State Initialization ---
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
    st.session_state.results = None

if 'stakes_data' not in st.session_state:
    st.session_state.stakes_data = DEFAULT_STAKES_DATA

if 'strategy_configs' not in st.session_state:
    # We now store the dictionary directly, not a formatted string
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
    ).sort_values(by='threshold', ascending=False).reset_index(drop=True)

    # Apply changes from the editor
    if edits["deleted_rows"]:
        df = df.drop(index=edits["deleted_rows"]).reset_index(drop=True)
    if edits["edited_rows"]:
        for index, changes in edits["edited_rows"].items():
            for col_name, new_value in changes.items():
                df.loc[index, col_name] = new_value
    if edits["added_rows"]:
        added_df = pd.DataFrame(edits["added_rows"], columns=expected_columns)
        df = pd.concat([df, added_df], ignore_index=True)

    # Convert the modified DataFrame back into the list-of-dicts format
    new_rules = []
    for _, row in df.iterrows():
        threshold_val = row.get('threshold')
        if threshold_val is None or pd.isna(threshold_val) or threshold_val <= 0:
            continue
        tables = {}
        for stake in available_stakes:
            if stake in row and pd.notna(row[stake]) and row[stake] != "":
                try:
                    tables[stake] = int(row[stake])
                except (ValueError, TypeError):
                    tables[stake] = str(row[stake])
        if tables:
            new_rules.append({"threshold": int(threshold_val), "tables": tables})
    st.session_state.strategy_configs[strategy_name]['rules'] = sorted(new_rules, key=lambda x: x['threshold'], reverse=True)

# --- Sidebar for User Inputs ---
st.sidebar.header("Simulation Parameters")

st.sidebar.subheader("General Settings")
st.sidebar.number_input("Starting Bankroll (€)", value=2030, min_value=0, step=100, help="The amount of money you are starting with for the simulation.", key="start_br")
st.sidebar.number_input("Target Bankroll (€)", value=3000, min_value=0, step=100, help="The bankroll amount you are aiming to reach. This is used to calculate 'Target Probability'.", key="target_br")
st.sidebar.number_input("Ruin Threshold (€)", value=750, min_value=0, step=50, help="If a simulation's bankroll drops to or below this value, it is considered 'ruined' and stops.", key="ruin_thresh")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.number_input("Number of Simulations", value=2000, min_value=10, max_value=50000, step=100, help="How many times to run the entire simulation from start to finish. Higher numbers give more accurate results but take longer. (e.g., 2000-10000)", key="num_sims")
with col2:
    st.number_input("Sessions per Simulation", value=100, min_value=1, max_value=1000, help="How many playing sessions are in a single simulation run. This determines the time horizon of the simulation.", key="num_sessions")

st.sidebar.subheader("Session Settings")
st.sidebar.number_input("Hands per Table per Session", value=60, min_value=1, help="The average number of hands you play on a single table during one session.", key="hands_per_table")
st.sidebar.slider("Stop-Loss per Session (%)", 0, 100, 10, help="The session ends early if you lose this percentage of your bankroll at the start of that session. Set to 100% to disable.", key="sl_percent")

col3, col4 = st.sidebar.columns(2)
with col3:
    st.number_input("Min Tables", value=3, min_value=1, help="The minimum number of tables you will play in any given session. The app will pick a random number between Min and Max for each session.", key="min_tables")
with col4:
    max_tables_default = max(5, st.session_state.min_tables)
    st.number_input("Max Tables", value=max_tables_default, min_value=st.session_state.min_tables, help="The maximum number of tables you will play in any given session.", key="max_tables")

st.sidebar.number_input("Target Tables (for % display)", value=4, min_value=1, help="Used for display purposes in the PDF report to show an example table mix for strategies that use percentages.", key="target_tables_pct")

st.sidebar.subheader("Rakeback")
st.sidebar.slider("Rakeback (%)", 0, 100, 20, help="The percentage of rake you get back from the poker site. This is added to your profit at the end of each session.", key="rb_percent")

st.sidebar.subheader("Advanced Statistical Settings")
st.sidebar.number_input("Prior Sample Size (for Bayesian model)", value=100000, min_value=1000, step=1000, help="Represents the strength of the model's prior belief about win rates. A larger value means the model is more confident in its own estimates and less influenced by small sample sizes from your data.", key="prior_sample")
st.sidebar.slider("Weight for 0-Hand Stake Estimates", 0.0, 1.0, 0.5, 0.05, help="For stakes where you have no hands played, this slider balances between your manual win rate estimate (1.0) and the model's extrapolation from other stakes (0.0).", key="zero_hands_weight")
st.sidebar.number_input("Min Degrees of Freedom (t-dist)", value=3, min_value=2, help="The starting 'fatness' of the tails for the t-distribution, used for small sample sizes to model higher variance. Must be > 2.", key="min_df")
st.sidebar.number_input("Max Degrees of Freedom (t-dist)", value=30, min_value=st.session_state.min_df, help="The 'df' for very large sample sizes. As df increases, the t-distribution approaches a normal distribution.", key="max_df")
st.sidebar.number_input("Hands to Reach Max DF", value=50000, min_value=1000, step=1000, help="The number of hands required to transition from Min DF to Max DF. This controls how quickly the model gains confidence in your results.", key="hands_for_max_df")
st.sidebar.number_input("Random Seed (for reproducibility)", value=45783, step=1, help="A fixed number that ensures the simulation produces the exact same random results every time. Change it to get a different set of random outcomes.", key="seed")

st.sidebar.button("Run Simulation", on_click=click_run_button, use_container_width=True)


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
        on_change=sync_stakes_data,
        num_rows="dynamic"
    )

with tab2:
    st.subheader("Bankroll Management Strategies")
    st.write("Define your strategies below. Use the data editor for 'Standard' strategies to set thresholds and table mixes. Use percentages (e.g., '80%') or fixed table counts (e.g., 4).")
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

        with st.expander(f"Edit Strategy: {name}", expanded=True):
            current_config = st.session_state.strategy_configs[name]

            # --- Row 1: Name, Type, Remove ---
            col1, col2, col3 = st.columns([2, 1, 1])
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
                st.write("") # Spacer
                st.write("") # Spacer
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
                num_buy_ins_value = current_config.get("num_buy_ins", 40)
                st.session_state.strategy_configs[name]['num_buy_ins'] = st.number_input(
                    "Buy-in Buffer (BIs)",
                    value=num_buy_ins_value, min_value=1, key=f"bi_{name}",
                    help="The number of buy-ins (100 BBs) required to move up/down stakes."
                )
                if 'rules' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['rules']
            
            elif strategy_type == 'standard':
                # Clean up keys from the other strategy type
                if 'num_buy_ins' in st.session_state.strategy_configs[name]:
                    del st.session_state.strategy_configs[name]['num_buy_ins']

                # --- Convert rules to a DataFrame for the editor ---
                rules_list = current_config.get("rules", [])
                df_data = [{'threshold': r['threshold'], **r.get('tables', {})} for r in rules_list]
                
                rules_df = pd.DataFrame(df_data, columns=['threshold'] + available_stakes)
                rules_df = rules_df.sort_values(by='threshold', ascending=False).reset_index(drop=True)

                # --- Display the data editor ---
                st.data_editor(
                    rules_df,
                    key=f"rules_{name}",
                    on_change=sync_strategy_rules,
                    args=(name,),
                    num_rows="dynamic",
                    column_config={
                        "threshold": st.column_config.NumberColumn(
                            "Bankroll Threshold (€)",
                            help="The bankroll amount at which this rule applies.",
                            min_value=0,
                            format="€ %d"
                        ),
                        **{stake: st.column_config.TextColumn(
                            f"{stake} Mix",
                            help=f"Table mix for {stake}. Use percentages ('50%') or fixed counts (4)."
                           ) for stake in available_stakes}
                    }
                )

                # The conversion logic is now handled by the sync_strategy_rules callback.
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
            "Risk of Ruin (%)": res['risk_of_ruin'],
            "Target Prob (%)": res['target_prob'],
            "5th %ile BR": res['p5'],
            "P95 Max Drawdown": res['p95_max_drawdown']
        })
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df.style.format({
        "Median Final BR": "€{:.2f}", "Mode Final BR": "€{:.2f}",
        "Median Growth": "{:.2%}", "Risk of Ruin (%)": "{:.2f}%",
        "Target Prob (%)": "{:.2f}%", "5th %ile BR": "€{:.2f}",
        "P95 Max Drawdown": "€{:.2f}"
    }))

    # --- Display Comparison Plot ---
    # Use columns to constrain the width of the main comparison plot
    plot_col_left, plot_col_main, plot_col_right = st.columns([1.0, 2.0, 1.0])
    with plot_col_main:
        st.pyplot(engine.plot_median_progression_comparison(all_results, config))

    # --- Display Detailed Results for Each Strategy ---
    for strategy_name, result in all_results.items():
        with st.expander(f"Detailed Analysis for: {strategy_name}"):
            st.subheader(f"Key Metrics for '{strategy_name}'")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Median Final Bankroll", f"€{result['median_final_bankroll']:.2f}")
            col2.metric("Risk of Ruin", f"{result['risk_of_ruin']:.2f}%")
            col3.metric("Target Probability", f"{result['target_prob']:.2f}%")
            col4.metric("Median Max Drawdown", f"€{result['median_max_drawdown']:.2f}")
            col5.metric("95th Pct. Drawdown", f"€{result['p95_max_drawdown']:.2f}")

            st.subheader("Charts")
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                fig1 = engine.plot_strategy_progression(result['bankroll_histories'], result['hands_histories'], strategy_name, config)
                st.pyplot(fig1)
            with plot_col2:
                fig2 = engine.plot_final_bankroll_distribution(result['final_bankrolls'], result, strategy_name, config)
                st.pyplot(fig2)

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
