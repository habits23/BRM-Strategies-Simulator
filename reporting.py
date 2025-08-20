import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_strategy_progression(bankroll_histories, hands_histories, strategy_name, config, pdf=None):
    """Plots the median progression with a shaded area for the 25th-75th percentile range."""
    fig, ax = plt.subplots(figsize=(8, 5)) # Further reduced size for a more compact app layout
    median_history = np.median(bankroll_histories, axis=0)
    y_upper_limit = np.percentile(bankroll_histories, 95) * 1.1 # Add 10% for padding
    lower_percentile = np.percentile(bankroll_histories, 25, axis=0)
    upper_percentile = np.percentile(bankroll_histories, 75, axis=0)
    median_hands = np.median(hands_histories, axis=0)

    for i in range(min(50, len(bankroll_histories))):
        ax.plot(hands_histories[i], bankroll_histories[i], color='gray', alpha=0.1)

    ax.fill_between(median_hands, lower_percentile, upper_percentile,
                     color='lightblue', alpha=0.4, label='25th-75th Percentile Range')
    ax.plot(median_hands, median_history, 'b-', linewidth=3, label='Median Progression')
    ax.axhline(config['STARTING_BANKROLL_EUR'], color='black', linewidth=2, label='Starting Bankroll')
    ax.axhline(config['TARGET_BANKROLL'], color='gold', linestyle='-.', label=f"Target: €{config['TARGET_BANKROLL']}")
    ax.set_title(f'Bankroll Progression: {strategy_name}', fontsize=14)
    ax.set_xlabel('Total Hands Played', fontsize=12)
    ax.set_ylabel('Bankroll (EUR)', fontsize=12)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0, top=y_upper_limit)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_final_bankroll_comparison(all_results, config, color_map=None, pdf=None):
    """
    Creates an overlapping density plot to compare the final bankroll distributions of all strategies.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    # --- Dynamically determine the x-axis range for a clearer plot --- #
    # Combine all results to find a global range that fits all strategies well. #
    all_final_bankrolls = np.concatenate([res['final_bankrolls'] for res in all_results.values() if len(res['final_bankrolls']) > 0])

    # Filter out ruined runs to focus the plot on the distribution of successful outcomes.
    successful_runs = all_final_bankrolls[all_final_bankrolls > config['RUIN_THRESHOLD']]

    if len(successful_runs) > 5: # Need a few points to calculate percentiles robustly
        # Use the user-defined percentile to control the plot's "zoom".
        # This makes the zoom symmetrical (e.g., 95 shows 5th-95th, 99 shows 1st-99th).
        upper_percentile = config.get('PLOT_PERCENTILE_LIMIT', 99)
        lower_percentile = 100 - upper_percentile

        x_min = np.percentile(successful_runs, lower_percentile)
        x_max = np.percentile(successful_runs, upper_percentile)

        # Ensure the starting bankroll is always visible for context.
        x_min = min(x_min, config['STARTING_BANKROLL_EUR'])
        x_max = max(x_max, config['STARTING_BANKROLL_EUR'])

        # Add some visual padding to the limits.
        padding = (x_max - x_min) * 0.05
        x_min -= padding
        x_max += padding
    else:
        # Fallback for cases with very few successful runs (e.g., high risk of ruin)
        x_min = 0
        x_max = config.get('TARGET_BANKROLL', 5000) * 1.5

    x_grid = np.linspace(x_min, x_max, 1000)

    for strategy_name, result in all_results.items():
        final_bankrolls = result['final_bankrolls']
        color = color_map.get(strategy_name)
        if len(final_bankrolls) > 1:
            try:
                # FIX: Filter the data before calculating the KDE.
                # This prevents extreme outliers (the top 1% we've already
                # excluded from the x-axis) from skewing the density plot.
                # The result is a plot that better represents the main body of the distribution.
                filtered_bankrolls = final_bankrolls[(final_bankrolls >= x_min) & (final_bankrolls <= x_max) & (final_bankrolls > config['RUIN_THRESHOLD'])]
                if len(filtered_bankrolls) < 2: # Need at least 2 points for KDE
                    ax.hist(final_bankrolls, bins=50, density=True, alpha=0.5, label=f"{strategy_name} (hist)", color=color)
                    continue

                kde = gaussian_kde(filtered_bankrolls)
                density = kde(x_grid)
                ax.plot(x_grid, density, label=strategy_name, color=color, linewidth=2)
                ax.fill_between(x_grid, density, color=color, alpha=0.1)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback for datasets that are not suitable for KDE
                ax.hist(final_bankrolls, bins=50, density=True, alpha=0.5, label=f"{strategy_name} (hist)", color=color)

    ax.set_title('Comparison of Final Bankroll Distributions', fontsize=16)
    ax.set_xlabel('Final Bankroll (EUR)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=x_min, right=x_max)
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_final_bankroll_distribution(final_bankrolls, result, strategy_name, config, pdf=None, color_map=None):
    """
    Plots the final bankroll distribution for a single strategy, highlighting median and mode.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use the strategy's assigned color if a map is provided
    plot_color = 'skyblue'
    if color_map and strategy_name in color_map:
        plot_color = color_map[strategy_name]

    # Filter out ruined runs to focus the plot on the distribution of successful outcomes.
    successful_runs = final_bankrolls[final_bankrolls > config['RUIN_THRESHOLD']]

    if len(successful_runs) > 5:
        upper_percentile = config.get('PLOT_PERCENTILE_LIMIT', 99)
        lower_percentile = 100 - upper_percentile

        x_min = np.percentile(successful_runs, lower_percentile)
        x_max = np.percentile(successful_runs, upper_percentile)

        x_min = min(x_min, config['STARTING_BANKROLL_EUR'])
        x_max = max(x_max, config['STARTING_BANKROLL_EUR'])

        padding = (x_max - x_min) * 0.05
        x_min -= padding
        x_max += padding
    else:
        x_min = 0
        x_max = config.get('TARGET_BANKROLL', 5000) * 1.5

    if len(successful_runs) > 0:
        # Filter the data for the histogram to match the x-axis limits for a cleaner plot
        filtered_for_hist = successful_runs[(successful_runs >= x_min) & (successful_runs <= x_max)]
        ax.hist(filtered_for_hist, bins=50, color=plot_color, edgecolor='black', alpha=0.7)

    # Add median and mode lines
    median_br = result.get('median_final_bankroll')
    mode_br = result.get('final_bankroll_mode')

    if median_br is not None:
        ax.axvline(median_br, color='darkgreen', linestyle='--', linewidth=2, label=f'Median: €{median_br:,.0f}')
    if mode_br is not None:
        ax.axvline(mode_br, color='darkred', linestyle=':', linewidth=2, label=f'Mode: €{mode_br:,.0f}')

    ax.set_title(f'Final Bankroll Distribution for {strategy_name}', fontsize=16)
    ax.set_xlabel('Final Bankroll (EUR)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Simulations)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(left=x_min, right=x_max)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_median_progression_comparison(all_results, config, color_map=None, pdf=None):
    """Compares the median bankroll progression for all strategies on a single plot."""
    fig, ax = plt.subplots(figsize=(8, 5)) # Made consistent with other compact plots
    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    for strategy_name, result in all_results.items():
        color = color_map.get(strategy_name)
        ax.plot(result['hands_history'], result['median_history'], label=strategy_name, linewidth=2.5, color=color)

    ax.axhline(config['STARTING_BANKROLL_EUR'], color='gray', linestyle='--', label='Starting Bankroll')
    ax.axhline(config['TARGET_BANKROLL'], color='gold', linestyle='-.', label=f"Target: €{config['TARGET_BANKROLL']}")

    ax.set_title('Median Bankroll Progression Comparison Across All Strategies', fontsize=16)
    ax.set_xlabel('Total Hands Played', fontsize=12)
    ax.set_ylabel('Bankroll (EUR)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_max_downswing_distribution(max_downswings, result, strategy_name, pdf=None, color_map=None):
    """Creates a histogram of maximum downswings with key metrics highlighted."""
    if max_downswings is None or len(max_downswings) == 0:
        return plt.figure() # Return an empty figure if no data

    # Use the strategy's assigned color if a map is provided
    hist_color = 'salmon'
    if color_map and strategy_name in color_map:
        hist_color = color_map[strategy_name]

    # Filter out extreme outliers for better visibility
    max_x_limit = np.percentile(max_downswings, 99.0)
    filtered_downswings = max_downswings[max_downswings <= max_x_limit]

    median_downswing = result['median_max_downswing']
    p95_downswing = result['p95_max_downswing']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(filtered_downswings, bins=50, color=hist_color, edgecolor='black', alpha=0.7)

    ax.axvline(median_downswing, color='darkred', linestyle='dashed', linewidth=2, label=f'Median Downswing: €{median_downswing:,.2f}')
    ax.axvline(p95_downswing, color='purple', linestyle=':', linewidth=2, label=f'95th Pct. Downswing: €{p95_downswing:,.2f}')

    ax.set_title(f'Maximum Downswing Distribution for {strategy_name}')
    ax.set_xlabel('Maximum Downswing (EUR)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_time_underwater_comparison(all_results, config, color_map=None, pdf=None):
    """
    Creates a bar chart comparing the median percentage of time each strategy
    spends 'underwater' (below a previous bankroll peak).
    """
    strategy_names = list(all_results.keys())
    underwater_pcts = [res.get('median_time_underwater_pct', 0) for res in all_results.values()]

    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    # Get colors in the correct order for the plot
    plot_colors = [color_map.get(name) for name in strategy_names]

    if not strategy_names:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Horizontal bar chart for better readability of strategy names
    bars = ax.barh(strategy_names, underwater_pcts, color=plot_colors)

    ax.set_xlabel('Median Time Spent "Underwater" (%)', fontsize=12)
    ax.set_title('Psychological Cost: Time Spent Below Bankroll Peak', fontsize=16)
    ax.invert_yaxis()  # Puts the first strategy at the top
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add percentage labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center')

    ax.set_xlim(right=max(underwater_pcts) * 1.15 if underwater_pcts else 100)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_total_withdrawn_comparison(all_results, config, color_map=None, pdf=None):
    """
    Creates a bar chart comparing the median total amount withdrawn for each strategy.
    Returns None if withdrawals were not enabled, so the UI can skip rendering it.
    """
    # Only show this plot if withdrawals were enabled for the simulation run.
    if not config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
        return None

    strategy_names = list(all_results.keys())
    withdrawn_amounts = [res.get('median_total_withdrawn', 0) for res in all_results.values()]

    if color_map is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    plot_colors = [color_map.get(name) for name in strategy_names]

    if not strategy_names:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(strategy_names, withdrawn_amounts, color=plot_colors)

    ax.set_xlabel('Median Total Withdrawn (€)', fontsize=12)
    ax.set_title('Income Generation: Median Total Withdrawn', fontsize=16)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        ax.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'€{width:,.0f}', va='center', ha='left')

    ax.set_xlim(right=max(withdrawn_amounts) * 1.2 if withdrawn_amounts and max(withdrawn_amounts) > 0 else 100)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_assigned_wr_distribution(avg_assigned_wr_per_sim, median_run_assigned_wr, average_input_wr, strategy_name, pdf=None):
    """
    Plots the distribution of the weighted average 'Assigned WR' for all simulations.
    This visualizes the 'luck' distribution and highlights where the median-outcome run falls.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter out extreme outliers for better visualization
    p1 = np.percentile(avg_assigned_wr_per_sim, 1)
    p99 = np.percentile(avg_assigned_wr_per_sim, 99)
    filtered_data = avg_assigned_wr_per_sim[(avg_assigned_wr_per_sim >= p1) & (avg_assigned_wr_per_sim <= p99)]

    ax.hist(filtered_data, bins=50, color='c', edgecolor='black', alpha=0.6, label='Distribution of All Runs\' Luck')

    ax.axvline(average_input_wr, color='blue', linestyle='--', linewidth=2,
               label=f'Your Avg. Input WR: {average_input_wr:.2f}')

    ax.axvline(median_run_assigned_wr, color='red', linestyle='-', linewidth=2.5,
               label=f'Luck of Median-Outcome Run: {median_run_assigned_wr:.2f}')

    ax.set_title(f'Distribution of Assigned Luck (WR) for {strategy_name}', fontsize=14)
    ax.set_xlabel('Weighted Average Assigned Win Rate (bb/100)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Simulations)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- Ensure vertical lines are always visible ---
    # Get the current limits and expand them if the lines are outside the visible range
    xmin, xmax = ax.get_xlim()
    new_xmin = min(xmin, median_run_assigned_wr, average_input_wr)
    new_xmax = max(xmax, median_run_assigned_wr, average_input_wr)
    padding = (new_xmax - new_xmin) * 0.05 # Add 5% padding
    ax.set_xlim(new_xmin - padding, new_xmax + padding)


    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_risk_reward_scatter(all_results, config, color_map=None, pdf=None):
    """
    Creates a scatter plot to visualize the risk vs. reward trade-off for each strategy.
    """
    strategy_names = list(all_results.keys())

    # Define the metrics for the axes
    # X-axis: Risk (lower is better)
    risk_metric_key = 'risk_of_ruin'
    risk_values = [res.get(risk_metric_key, 0) for res in all_results.values()]
    risk_label = 'Risk of Ruin (%)'

    # Y-axis: Reward (higher is better)
    reward_metric_key = 'median_final_bankroll'
    reward_values = [res.get(reward_metric_key, 0) for res in all_results.values()]
    reward_label = f"Median Final Bankroll (€)"

    if not strategy_names or len(strategy_names) < 2:
        return plt.figure() # Don't plot if there's nothing to compare

    fig, ax = plt.subplots(figsize=(8, 6))

    if color_map is None:
        # Fallback for generating colors internally if no map is provided
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    plot_colors = [color_map.get(name) for name in strategy_names]

    ax.scatter(risk_values, reward_values, s=150, c=plot_colors, alpha=0.7, edgecolors='w', zorder=10)

    # Annotate each point with the strategy name
    for i, name in enumerate(strategy_names):
        ax.text(risk_values[i], reward_values[i] + np.std(reward_values)*0.03, name, fontsize=9, ha='center')

    # Add interpretation quadrants based on the average
    avg_risk = np.mean(risk_values)
    avg_reward = np.mean(reward_values)
    ax.axvline(avg_risk, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(avg_reward, color='gray', linestyle='--', linewidth=0.8)

    ax.set_xlabel(risk_label, fontsize=12)
    ax.set_ylabel(reward_label, fontsize=12)
    ax.set_title('Risk vs. Reward Analysis', fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.5)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def get_initial_table_mix_string(strategy, config):
    """Helper function to describe the starting table mix as a string."""
    rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    if not rule:
        return "No Play"
    mix_parts = [f"{stake}: {value}" for stake, value in sorted(rule.items())]
    return ", ".join(mix_parts) if mix_parts else "No Play"