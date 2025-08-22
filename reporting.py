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
    # Dynamic height to prevent legend from squishing the plot with many strategies
    num_strategies = len(all_results)
    # A small increase in height per strategy to accommodate the legend
    fig_height = max(5, 4.0 + num_strategies * 0.25)
    fig, ax = plt.subplots(figsize=(8, fig_height))

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
    p5_br = result.get('p5')
    p95_br = result.get('p95') # Use pre-calculated percentile for consistency
    target_br = config.get('TARGET_BANKROLL')

    if median_br is not None:
        ax.axvline(median_br, color='darkgreen', linestyle='--', linewidth=2, label=f'Median: €{median_br:,.0f}')
    if mode_br is not None:
        ax.axvline(mode_br, color='darkred', linestyle=':', linewidth=2, label=f'Mode: €{mode_br:,.0f}')
    if p5_br is not None:
        ax.axvline(p5_br, color='orangered', linestyle='--', linewidth=1.5, label=f'5th %ile: €{p5_br:,.0f}')
    if p95_br is not None:
        ax.axvline(p95_br, color='purple', linestyle='--', linewidth=1.5, label=f'95th %ile: €{p95_br:,.0f}')
    if target_br is not None:
        ax.axvline(target_br, color='gold', linestyle='-.', linewidth=2, label=f'Target: €{target_br:,.0f}')

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
    # Dynamic height to prevent legend from squishing the plot with many strategies
    num_strategies = len(all_results)
    # A small increase in height per strategy to accommodate the legend
    fig_height = max(5, 4.0 + num_strategies * 0.25)
    fig, ax = plt.subplots(figsize=(8, fig_height))
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

    # Dynamic height to ensure bars are not too squished with many strategies
    num_strategies = len(strategy_names)
    fig_height = max(4, 2.0 + num_strategies * 0.7) # Base height + per-strategy height, with a minimum
    fig, ax = plt.subplots(figsize=(8, fig_height))

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

    # Dynamic height to ensure bars are not too squished with many strategies
    num_strategies = len(strategy_names)
    # Use the same dynamic height formula for consistency with the other h-bar chart
    fig_height = max(4, 2.0 + num_strategies * 0.7)
    fig, ax = plt.subplots(figsize=(8, fig_height))
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

    # Make height dynamic to match the "Time Spent Underwater" plot it's paired with.
    # This ensures visual uniformity in the UI layout.
    num_strategies = len(strategy_names)
    fig_height = max(4, 2.0 + num_strategies * 0.7)
    fig, ax = plt.subplots(figsize=(8, fig_height))

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

def plot_hands_distribution_table(result, strategy_name, pdf=None):
    """
    Creates a table summarizing the hands distribution and win rates per stake
    and saves it to the PDF.
    """
    hands_dist = result.get('hands_distribution_pct')
    avg_wr = result.get('avg_assigned_wr_per_stake')
    trust = result.get('trust_per_stake')

    if not hands_dist or not avg_wr or not trust:
        return None # Not enough data to plot

    # Sort stakes by their 'bb_size' from the config for a logical order
    stakes_data = result.get('stakes_data_for_report', [])
    stake_order = {s['name']: s['bb_size'] for s in stakes_data}
    stake_names = sorted(hands_dist.keys(), key=lambda s: stake_order.get(s, float('inf')))

    if not stake_names:
        return None

    # Prepare data for the table
    cell_text = []
    for stake in stake_names:
        row = [
            f"{hands_dist.get(stake, 0):.1f}%",
            f"{avg_wr.get(stake, 0):.2f}",
            f"{trust.get(stake, 0):.1f}%"
        ]
        cell_text.append(row)

    columns = ['% of Hands', 'Avg. Assigned WR', 'Input Trust']
    rows = stake_names

    fig, ax = plt.subplots(figsize=(8, max(1.5, len(rows) * 0.5))) # Dynamic height
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5) # Adjust column width and row height

    ax.set_title(f'Hands Distribution & Trust for {strategy_name}', pad=20, fontsize=14)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_percentile_wr_analysis_table(result, strategy_name, pdf=None):
    """
    Creates a table showing the win rate breakdown for key percentile runs
    and saves it to the PDF.
    """
    percentile_wrs = result.get('percentile_win_rates')
    if not percentile_wrs:
        return None

    # Define the order of rows (percentiles)
    row_order = [p for p in ['p5', 'p25', 'p50_median', 'p75', 'p95'] if p in percentile_wrs]
    if not row_order:
        return None

    # Prepare data for the table
    cell_text = []
    for percentile_key in row_order:
        data = percentile_wrs[percentile_key]
        row = [
            f"{data.get('assigned_wr', 0):.2f}",
            f"{data.get('play_wr', 0):.2f}",
            f"{data.get('rakeback_wr', 0):.2f}",
            f"{data.get('variance_impact', 0):.2f}"
        ]
        cell_text.append(row)

    columns = ['Assigned WR', 'Play WR', 'Rakeback WR', 'Variance Impact']
    row_labels = {
        'p5': '5th Percentile', 'p25': '25th Percentile', 'p50_median': 'Median Run',
        'p75': '75th Percentile', 'p95': '95th Percentile'
    }
    rows = [row_labels.get(key, key) for key in row_order]

    fig, ax = plt.subplots(figsize=(8, max(1.5, len(rows) * 0.5))) # Dynamic height
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title(f'Percentile Win Rate Analysis (bb/100) for {strategy_name}', pad=20, fontsize=14)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def add_text_page(text, pdf, title=""):
    """
    Creates a matplotlib figure with wrapped text and adds it to the PDF.
    Useful for adding analysis summaries or other text blocks.
    """
    fig = plt.figure(figsize=(8.27, 11.69)) # A4 size in inches
    fig.clf()
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)

    # Place text with wrapping
    fig.text(0.05, 0.9, text, va='top', ha='left', wrap=True, fontsize=10)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_summary_table(all_results, strategy_page_map, config, pdf=None):
    """
    Creates a comprehensive summary table as an image, mirroring the UI's detail,
    and saves it to the PDF.
    """
    header = [
        'Strategy', 'Page', 'Median Final BR', 'Median Growth', 'Median Hands',
        'Median Rakeback', 'RoR (%)', 'Target Prob (%)', '5th %ile BR', 'P95 Downswing'
    ]
    cell_text = []

    for strategy_name, res in all_results.items():
        row = [
            strategy_name,
            str(strategy_page_map.get(strategy_name, '-')),
            f"€{res.get('median_final_bankroll', 0):,.0f}",
            f"{res.get('growth_rate', 0):.2%}",
            f"{res.get('median_hands_played', 0):,.0f}",
            f"€{res.get('median_rakeback_eur', 0):,.2f}",
            f"{res.get('risk_of_ruin', 0):.2f}%",
            f"{res.get('target_prob', 0):.2f}%",
            f"€{res.get('p5', 0):,.0f}",
            f"€{res.get('p95_max_downswing', 0):,.0f}"
        ]
        cell_text.append(row)

    if not cell_text:
        return None

    fig, ax = plt.subplots(figsize=(11.69, 8.27)) # A4 landscape
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=header, cellLoc='center', loc='center')
    table.automake_col_widths()
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    ax.set_title('Simulation Summary', pad=20, fontsize=16)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_risk_of_demotion_table(result, strategy_name, pdf=None):
    """
    Creates a table showing the risk of demotion from each stake level.
    """
    risk_data = result.get('risk_of_demotion')
    if not risk_data:
        return None

    # Sort stakes by their 'bb_size' from the config for a logical order
    stakes_data = result.get('stakes_data_for_report', [])
    stake_order = {s['name']: s['bb_size'] for s in stakes_data}
    stake_names = sorted(risk_data.keys(), key=lambda s: stake_order.get(s, float('inf')))

    if not stake_names:
        return None

    cell_text = []
    for stake in stake_names:
        data = risk_data[stake]
        row = [
            f"{int(data.get('reached_count', 0))}",
            f"{data.get('prob', 0):.1f}%"
        ]
        cell_text.append(row)

    columns = ['Sims Reaching Stake', 'Demotion Probability']
    rows = stake_names

    fig, ax = plt.subplots(figsize=(8, max(1.5, len(rows) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title(f'Risk of Demotion Analysis for {strategy_name}', pad=20, fontsize=14)

    if pdf:
        pdf.savefig(fig)
        plt.close(fig)
    return fig

def plot_sanity_check_analysis(result, config, pdf=None):
    """
    Creates a text-based plot comparing analytical expectations vs. simulation results
    for the single-stake sanity check.
    """
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

    text_content = (
        "This report compares the simulation's output against known mathematical formulas for a simple, single-stake scenario.\n"
        "The 'Actual' values from the simulation should be very close to the 'Expected' values calculated analytically.\n\n"
        "--- Expected (Analytical) ---\n"
        f"Expected Final Median Bankroll: €{expected_final_br:,.2f}\n"
        f"Expected Std. Dev. of Final Bankroll: €{expected_std_dev_eur:,.2f}\n\n"
        "--- Actual (from Simulation) ---\n"
        f"Actual Final Median Bankroll: €{actual_median_br:,.2f}\n"
        f"Actual Std. Dev. of Final Bankroll: €{actual_std_dev_br:,.2f}\n\n"
        "--- Comparison ---\n"
        f"Difference in Median: {median_diff_pct:+.2f}%\n"
        f"Difference in Std. Dev.: {std_dev_diff_pct:+.2f}%"
    )

    # Use the generic text page function to render this content
    return add_text_page(text_content, pdf, title="Sanity Check Analysis")

def get_initial_table_mix_string(strategy, config):
    """Helper function to describe the starting table mix as a string."""
    rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    if not rule:
        return "No Play"
    mix_parts = [f"{stake}: {value}" for stake, value in sorted(rule.items())]
    return ", ".join(mix_parts) if mix_parts else "No Play"