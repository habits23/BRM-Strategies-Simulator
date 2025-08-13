import streamlit as st
import pandas as pd
import ast
import datetime
import traceback

# Import the actual simulation engine we just built
import simulation_engine as engine

st.set_page_config(layout="wide", page_title="Poker Bankroll Simulator")

st.title("Poker Bankroll Simulator")
st.write("An interactive tool to simulate poker bankroll progression based on your data and strategies. Based on the logic from `Final BR Simulator v1_5.py`.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Simulation Parameters")

# Use a form to batch inputs together
with st.sidebar.form("sim_form"):
    st.subheader("General Settings")
    config = {}
    config["STARTING_BANKROLL_EUR"] = st.number_input("Starting Bankroll (€)", value=2030, min_value=0, step=100, help="The amount of money you are starting with for the simulation.")
    config["TARGET_BANKROLL"] = st.number_input("Target Bankroll (€)", value=3000, min_value=0, step=100, help="The bankroll amount you are aiming to reach. This is used to calculate 'Target Probability'.")
    config["RUIN_THRESHOLD"] = st.number_input("Ruin Threshold (€)", value=750, min_value=0, step=50, help="If a simulation's bankroll drops to or below this value, it is considered 'ruined' and stops.")

    col1, col2 = st.columns(2)
    with col1:
        config["NUMBER_OF_SIMULATIONS"] = st.number_input("Number of Simulations", value=2000, min_value=10, max_value=50000, step=100, help="How many times to run the entire simulation from start to finish. Higher numbers give more accurate results but take longer. (e.g., 2000-10000)")
    with col2:
        config["TOTAL_SESSIONS_PER_RUN"] = st.number_input("Sessions per Simulation", value=100, min_value=1, max_value=1000, help="How many playing sessions are in a single simulation run. This determines the time horizon of the simulation.")

    st.subheader("Session Settings")
    config["HANDS_PER_TABLE_PER_SESSION"] = st.number_input("Hands per Table per Session", value=60, min_value=1, help="The average number of hands you play on a single table during one session.")
    sl_percent = st.slider("Stop-Loss per Session (%)", 0, 100, 10, help="The session ends early if you lose this percentage of your bankroll at the start of that session. Set to 100% to disable.")
    config["STOP_LOSS_PERCENTAGE_PER_SESSION"] = sl_percent / 100.0

    col3, col4 = st.columns(2)
    with col3:
        config["MIN_TABLES_PER_SESSION"] = st.number_input("Min Tables", value=3, min_value=1, help="The minimum number of tables you will play in any given session. The app will pick a random number between Min and Max for each session.")
    with col4:
        # Ensure the default value for Max Tables is never less than the current Min Tables value.
        max_tables_default = max(5, config["MIN_TABLES_PER_SESSION"])
        config["MAX_TABLES_PER_SESSION"] = st.number_input("Max Tables", value=max_tables_default, min_value=config["MIN_TABLES_PER_SESSION"], help="The maximum number of tables you will play in any given session.")

    config["TARGET_TOTAL_TABLES_FOR_PERCENTAGES"] = st.number_input("Target Tables (for % display)", value=4, min_value=1, help="Used for display purposes in the PDF report to show an example table mix for strategies that use percentages.")

    st.subheader("Rakeback")
    rb_percent = st.slider("Rakeback (%)", 0, 100, 20, help="The percentage of rake you get back from the poker site. This is added to your profit at the end of each session.")
    config["RAKEBACK_PERCENTAGE"] = rb_percent / 100.0

    st.subheader("Advanced Statistical Settings")
    config["PRIOR_SAMPLE_SIZE"] = st.number_input("Prior Sample Size (for Bayesian model)", value=100000, min_value=1000, step=1000, help="Represents the strength of the model's prior belief about win rates. A larger value means the model is more confident in its own estimates and less influenced by small sample sizes from your data.")
    config["ZERO_HANDS_INPUT_WEIGHT"] = st.slider("Weight for 0-Hand Stake Estimates", 0.0, 1.0, 0.5, 0.05, help="For stakes where you have no hands played, this slider balances between your manual win rate estimate (1.0) and the model's extrapolation from other stakes (0.0).")
    config["MIN_DEGREES_OF_FREEDOM"] = st.number_input("Min Degrees of Freedom (t-dist)", value=3, min_value=2, help="The starting 'fatness' of the tails for the t-distribution, used for small sample sizes to model higher variance. Must be > 2.")
    config["MAX_DEGREES_OF_FREEDOM"] = st.number_input("Max Degrees of Freedom (t-dist)", value=30, min_value=config["MIN_DEGREES_OF_FREEDOM"], help="The 'df' for very large sample sizes. As df increases, the t-distribution approaches a normal distribution.")
    config["HANDS_FOR_MAX_DF"] = st.number_input("Hands to Reach Max DF", value=50000, min_value=1000, step=1000, help="The number of hands required to transition from Min DF to Max DF. This controls how quickly the model gains confidence in your results.")
    config["SEED"] = st.number_input("Random Seed (for reproducibility)", value=45783, step=1, help="A fixed number that ensures the simulation produces the exact same random results every time. Change it to get a different set of random outcomes.")

    st.write("---") # Visual separator
    ready_to_run = st.checkbox("I've finished editing and am ready to run.", value=False, help="Check this box to confirm you are ready. The simulation will only run if this is checked.")

    # This is the button that will trigger the simulation
    submitted = st.form_submit_button("Run Simulation", use_container_width=True)


# --- Main Area for Data and Strategy Inputs ---
st.header("Player & Strategy Configuration")

tab1, tab2 = st.tabs(["Stakes Data", "Bankroll Management Strategies"])

with tab1:
    st.subheader("Stakes Data")
    st.write("Enter your performance statistics for each stake you play. You can add or remove rows.")
    # Default data from your script
    stakes_df = pd.DataFrame([
        {"name": "NL20", "bb_per_100": 7.96, "ev_bb_per_100": 8.29, "std_dev_per_100": 91.41, "sample_hands": 77657, "bb_size": 0.20, "win_rate_drop": 0.0, "rake_bb_per_100": 15.82},
        {"name": "NL50", "bb_per_100": 23.0, "ev_bb_per_100": 15.2, "std_dev_per_100": 85.43, "sample_hands": 5681, "bb_size": 0.50, "win_rate_drop": 1.5, "rake_bb_per_100": 10.36},
        {"name": "NL100", "bb_per_100": 2.5, "ev_bb_per_100": 2.5, "std_dev_per_100": 100.0, "sample_hands": 0, "bb_size": 1.00, "win_rate_drop": 1.0, "rake_bb_per_100": 7.0},
        {"name": "NL200", "bb_per_100": 1.5, "ev_bb_per_100": 1.5, "std_dev_per_100": 100.0, "sample_hands": 0, "bb_size": 2.00, "win_rate_drop": 0.5, "rake_bb_per_100": 6.5},
    ])
    # The data_editor provides a spreadsheet-like interface
    edited_stakes = st.data_editor(stakes_df, num_rows="dynamic", key="stakes_editor")
    config["STAKES_DATA"] = edited_stakes.to_dict('records')

with tab2:
    st.subheader("Bankroll Management Strategies")
    st.write("Define your strategies using Python dictionary syntax. The tool will simulate and compare all of them.")

    # Default strategies from your script
    default_strategies_str = """{
    "Gradual Progressive": {
        "type": "standard",
        "rules": [
            {"threshold": 20000, "tables": {"NL200": "100%"}},
            {"threshold": 18000, "tables": {"NL100": "20%", "NL200": "80%"}},
            {"threshold": 16000, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 12000, "tables": {"NL100": "80%", "NL200": "20%"}},
            {"threshold": 8000, "tables": {"NL100": "100%"}},
            {"threshold": 7000, "tables": {"NL50": "20%", "NL100": "80%"}},
            {"threshold": 6000, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 5000, "tables": {"NL50": "80%", "NL100": "20%"}},
            {"threshold": 3500, "tables": {"NL50": "100%"}},
            {"threshold": 3000, "tables": {"NL20": "20%", "NL50": "80%"}},
            {"threshold": 2500, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 2000, "tables": {"NL20": "80%", "NL50": "20%"}},
            {"threshold": 1200, "tables": {"NL20": "100%"}}
        ]
    },
    "Custom Strategy": {
        "type": "standard",
        "rules": [
            {"threshold": 3100, "tables": {"NL50": "100%"}},
            {"threshold": 2650, "tables": {"NL20": "10%-30%", "NL50": "70%-90%"}},
            {"threshold": 2250, "tables": {"NL20": "30%-50%", "NL50": "50%-70%"}},
            {"threshold": 1750, "tables": {"NL20": "50%-70%", "NL50": "30%-50%"}},
            {"threshold": 1250, "tables": {"NL20": "70%-90%", "NL50": "10%-30%"}},
            {"threshold": 675, "tables": {"NL20": "100%"}}
        ]
    },
    "Ultra Conservative": {
        "type": "standard",
        "rules": [
            {"threshold": 30000, "tables": {"NL200": "100%"}},
            {"threshold": 25000, "tables": {"NL100": "20%", "NL200": "80%"}},
            {"threshold": 20000, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 16000, "tables": {"NL100": "80%", "NL200": "20%"}},
            {"threshold": 12000, "tables": {"NL100": "100%"}},
            {"threshold": 10000, "tables": {"NL100": "100%"}},
            {"threshold": 8000, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 6000, "tables": {"NL50": "80%", "NL100": "20%"}},
            {"threshold": 5000, "tables": {"NL50": "100%"}},
            {"threshold": 4000, "tables": {"NL20": "20%", "NL50": "80%"}},
            {"threshold": 3000, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 2500, "tables": {"NL20": "80%", "NL50": "20%"}},
            {"threshold": 1600, "tables": {"NL20": "100%"}},
            {"threshold": 750, "tables": {"NL20": "100%"}}
        ]
    },
    "Granular Aggressive": {
        "type": "standard",
        "rules": [
            {"threshold": 6000, "tables": {"NL200": "100%"}},
            {"threshold": 5500, "tables": {"NL100": "20%", "NL200": "80%"}},
            {"threshold": 5000, "tables": {"NL100": "50%", "NL200": "50%"}},
            {"threshold": 4000, "tables": {"NL100": "80%", "NL200": "20%"}},
            {"threshold": 2500, "tables": {"NL100": "100%"}},
            {"threshold": 2250, "tables": {"NL50": "20%", "NL100": "80%"}},
            {"threshold": 2000, "tables": {"NL50": "50%", "NL100": "50%"}},
            {"threshold": 1750, "tables": {"NL50": "80%", "NL100": "20%"}},
            {"threshold": 1000, "tables": {"NL50": "100%"}},
            {"threshold": 900, "tables": {"NL20": "20%", "NL50": "80%"}},
            {"threshold": 750, "tables": {"NL20": "50%", "NL50": "50%"}},
            {"threshold": 600, "tables": {"NL20": "80%", "NL50": "20%"}},
            {"threshold": 400, "tables": {"NL20": "100%"}}
        ]
    },
    "Balanced": {
        "type": "standard",
        "rules": [
            {"threshold": 10000, "tables": {"NL200": "100%"}},
            {"threshold": 9000, "tables": {"NL100": "20%-40%", "NL200": "60%-80%"}},
            {"threshold": 8000, "tables": {"NL100": "40%-60%", "NL200": "40%-60%"}},
            {"threshold": 7000, "tables": {"NL100": "60%-80%", "NL200": "20%-40%"}},
            {"threshold": 5000, "tables": {"NL100": "100%"}},
            {"threshold": 4500, "tables": {"NL50": "10%-30%", "NL100": "70%-90%"}},
            {"threshold": 4000, "tables": {"NL50": "30%-50%", "NL100": "50%-70%"}},
            {"threshold": 3500, "tables": {"NL50": "50%-70%", "NL100": "30%-50%"}},
            {"threshold": 2000, "tables": {"NL50": "100%"}},
            {"threshold": 1800, "tables": {"NL20": "20%-40%", "NL50": "60%-80%"}},
            {"threshold": 1600, "tables": {"NL20": "40%-60%", "NL50": "40%-60%"}},
            {"threshold": 1400, "tables": {"NL20": "60%-80%", "NL50": "20%-40%"}},
            {"threshold": 800, "tables": {"NL20": "100%"}}
        ]
    },
    "Sticky Strategy": {
        "type": "hysteresis",
        "num_buy_ins": {
            "NL20": 25, "NL50": 30, "NL100": 40, "NL200": 40
        }
    }
}"""

    strategy_text = st.text_area("Strategy Definitions", value=default_strategies_str, height=600)

    try:
        # ast.literal_eval is a safe way to parse Python literals from a string
        strategies_to_run = ast.literal_eval(strategy_text)
        config["STRATEGIES_TO_RUN"] = strategies_to_run
        st.success("Strategies parsed successfully!")
    except (ValueError, SyntaxError) as e:
        st.error(f"Error parsing strategy rules. Please check your syntax. Details: {e}")
        strategies_to_run = None


# --- Main Logic to Run Simulation and Display Results ---
if submitted:
    if not strategies_to_run:
        st.error("Cannot run simulation due to invalid strategy rules.")
    elif not ready_to_run:
        st.warning("Please check the 'I've finished editing and am ready to run.' box in the sidebar before running the simulation.")
    else:
        st.header("Simulation Results")
        try:
            with st.spinner("Running thousands of simulations... this may take a moment."):
                # THIS IS WHERE WE CALL OUR REFACTORED ENGINE
                all_results = engine.run_full_analysis(config)

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

        except Exception as e:
            st.error("An error occurred during the simulation.")
            st.exception(e)
