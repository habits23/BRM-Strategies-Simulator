import numpy as np
from scipy.stats import gaussian_kde
from collections import defaultdict

def calculate_binned_mode(data, ruin_threshold):
    """Calculates the mode of a continuous distribution using Kernel Density Estimation (KDE)."""
    successful_runs = data[data > ruin_threshold]
    if len(successful_runs) < 2:
        return np.median(data)

    p99 = np.percentile(successful_runs, 99)
    filtered_data = successful_runs[successful_runs <= p99]
    if len(filtered_data) < 2:
        return np.median(successful_runs)

    try:
        kde = gaussian_kde(filtered_data)
        grid = np.linspace(min(filtered_data), max(filtered_data), 1000)
        density = kde(grid)
        return grid[np.argmax(density)]
    except (np.linalg.LinAlgError, ValueError):
        return np.median(filtered_data)

def _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, total_withdrawn_histories, config, bb_size_map):
    """Calculates assigned and realized win rates for simulations closest to key percentiles."""
    percentile_win_rates = {}
    percentiles_to_find = {'2.5th': 2.5, '5th': 5, '25th': 25, 'Median': 50, '75th': 75, '95th': 95, '97.5th': 97.5}
    final_withdrawn = total_withdrawn_histories[:, -1]
    final_rakeback = rakeback_histories[:, -1]

    for name, p_val in sorted(percentiles_to_find.items(), key=lambda item: item[1]):
        percentile_bankroll = np.percentile(final_bankrolls, p_val)
        closest_sim_index = np.argmin(np.abs(final_bankrolls - percentile_bankroll))

        stake_wrs = {'p_val': p_val}
        total_hands_for_sim, assigned_weighted_wr_sum, weighted_bb_size_sum = 0, 0, 0

        for stake_name, wr_array in all_win_rates.items():
            exact_wr = wr_array[closest_sim_index]
            stake_wrs[stake_name] = f"{exact_wr:.2f}"
            hands_at_stake = hands_per_stake_histories[stake_name][closest_sim_index, -1]
            total_hands_for_sim += hands_at_stake
            assigned_weighted_wr_sum += exact_wr * hands_at_stake
            weighted_bb_size_sum += hands_at_stake * bb_size_map[stake_name]

        if total_hands_for_sim > 0:
            stake_wrs['Assigned WR'] = f"{assigned_weighted_wr_sum / total_hands_for_sim:.2f}"
            avg_bb_size = weighted_bb_size_sum / total_hands_for_sim

            if avg_bb_size > 0:
                # CRITICAL: Total profit must include money that was withdrawn.
                # Total value generated = (Final BR - Start BR) + Total Withdrawn
                total_profit_eur = (final_bankrolls[closest_sim_index] + final_withdrawn[closest_sim_index]) - config['STARTING_BANKROLL_EUR']
                profit_from_play_eur = total_profit_eur - final_rakeback[closest_sim_index]
                realized_wr_val = (profit_from_play_eur / avg_bb_size) / (total_hands_for_sim / 100)
                assigned_wr_val = assigned_weighted_wr_sum / total_hands_for_sim
                variance_impact_val = realized_wr_val - assigned_wr_val

                stake_wrs['Realized WR (Play)'] = f"{realized_wr_val:.2f}"
                stake_wrs['Rakeback (bb/100)'] = f"{(final_rakeback[closest_sim_index] / avg_bb_size) / (total_hands_for_sim / 100):.2f}"
                stake_wrs['Variance Impact'] = f"{variance_impact_val:+.2f}" # Use + to show positive/negative impact
            else: # Handle case where avg_bb_size is 0
                stake_wrs['Realized WR (Play)'] = "N/A"
                stake_wrs['Rakeback (bb/100)'] = "N/A"
                stake_wrs['Variance Impact'] = "N/A"

        percentile_win_rates[f"{name} Percentile"] = stake_wrs
    return percentile_win_rates

def analyze_strategy_results(strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng, peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, stop_loss_triggers, underwater_hands_count, integrated_drawdown, total_withdrawn_histories, downswing_depth_exceeded, downswing_duration_exceeded, config):
    """Takes the raw simulation output and calculates all the necessary metrics and analytics."""
    bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    total_hands_histories = np.sum(list(hands_per_stake_histories.values()), axis=0)
    final_withdrawn = total_withdrawn_histories[:, -1]
    final_bankrolls = bankroll_histories[:, -1]

    # --- Calculate Weighted Assigned WR for each simulation ---
    # This is crucial for understanding the distribution of "luck" (assigned win rates)
    final_hands_per_stake = {name: history[:, -1] for name, history in hands_per_stake_histories.items()}
    total_hands_per_sim = np.sum(list(final_hands_per_stake.values()), axis=0)

    weighted_assigned_wr_sum = np.zeros(config['NUMBER_OF_SIMULATIONS'])
    for stake_name, wr_array in all_win_rates.items():
        weighted_assigned_wr_sum += wr_array * final_hands_per_stake[stake_name]

    # Avoid division by zero for sims with no hands played
    avg_assigned_wr_per_sim = np.divide(
        weighted_assigned_wr_sum,
        total_hands_per_sim,
        out=np.zeros_like(weighted_assigned_wr_sum),
        where=total_hands_per_sim != 0
    )

    # Find the Assigned WR for the run that resulted in the median final bankroll
    median_final_bankroll_val = np.percentile(final_bankrolls, 50)
    median_sim_index = np.argmin(np.abs(final_bankrolls - median_final_bankroll_val))
    median_run_assigned_wr = avg_assigned_wr_per_sim[median_sim_index]

    average_assigned_win_rates = {name: np.mean(wr_array) for name, wr_array in all_win_rates.items()}

    total_hands_per_stake = {name: np.sum(history[:, -1]) for name, history in hands_per_stake_histories.items()}
    grand_total_hands = sum(total_hands_per_stake.values())
    hands_distribution_pct = {name: (total / grand_total_hands) * 100 for name, total in total_hands_per_stake.items()} if grand_total_hands > 0 else {}

    risk_of_demotion = {}
    for level, stake_name in stake_name_map.items():
        if level > 0:
            sims_that_reached_peak = np.sum(peak_stake_levels >= level)
            if sims_that_reached_peak > 0:
                sims_demoted_from_peak = np.sum(demotion_flags[level])
                demotion_prob = (sims_demoted_from_peak / sims_that_reached_peak) * 100
                risk_of_demotion[stake_name] = {'prob': demotion_prob, 'reached_count': sims_that_reached_peak}

    final_stake_counts = defaultdict(int)
    final_highest_stake_counts = defaultdict(int)
    stake_order_map = {stake['name']: i for i, stake in enumerate(sorted(config['STAKES_DATA'], key=lambda s: s['bb_size']))}

    for br in final_bankrolls:
        table_mix = strategy_obj.get_table_mix(br)
        # Sort by string representation of keys to prevent TypeErrors with mixed-type keys (e.g., 'NL50' and NaN from empty rows)
        sorted_items = sorted(table_mix.items(), key=lambda item: str(item[0])) if table_mix else []
        mix_str = ", ".join(f"{s}: {v}" for s, v in sorted_items) if sorted_items else "No Play"
        final_stake_counts[mix_str] += 1

        # This is the more direct calculation for the UI summary.
        if not table_mix:
            final_highest_stake_counts["No Play"] += 1
        else:
            # Filter the mix to only include stakes that are actually played.
            # This is critical for the Hysteresis strategy, which includes all stakes in its
            # rules, with most set to "0%". The original code would incorrectly find the
            # highest stake overall (e.g., NL200) instead of the one with a "100%" allocation.
            active_stakes_in_mix = {
                stake: value for stake, value in table_mix.items()
                if (isinstance(value, str) and float(value.replace('%','').split('-')[0]) > 0) or (isinstance(value, int) and value > 0)
            }

            if not active_stakes_in_mix:
                final_highest_stake_counts["No Play"] += 1
            else:
                highest_stake = max(active_stakes_in_mix.keys(), key=lambda s: stake_order_map.get(s, -1))
                final_highest_stake_counts[highest_stake] += 1

    final_stake_distribution = {mix_str: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for mix_str, count in final_stake_counts.items()}
    final_highest_stake_distribution = {stake: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for stake, count in final_highest_stake_counts.items()}

    percentile_win_rates = _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, total_withdrawn_histories, config, bb_size_map)

    median_max_downswing = np.median(max_drawdowns)
    p95_max_downswing = np.percentile(max_drawdowns, 95)

    # Calculate median rakeback
    final_rakeback = rakeback_histories[:, -1]
    median_rakeback_eur = np.median(final_rakeback)

    # Calculate median profit from play
    # This now correctly includes withdrawn amounts as part of the total profit generated.
    total_profit_per_sim = (final_bankrolls + final_withdrawn) - config['STARTING_BANKROLL_EUR']
    profit_from_play_per_sim = total_profit_per_sim - final_rakeback
    median_profit_from_play_eur = np.median(profit_from_play_per_sim)

    # Calculate median stop-losses
    median_stop_losses = np.median(stop_loss_triggers) if stop_loss_triggers is not None else 0

    # Calculate median hands played
    median_hands_played = np.median(total_hands_per_sim)

    # Calculate median time spent underwater
    time_underwater_pct = np.divide(
        underwater_hands_count,
        total_hands_per_sim,
        out=np.zeros_like(underwater_hands_count, dtype=float),
        where=total_hands_per_sim != 0
    ) * 100
    median_time_underwater_pct = np.median(time_underwater_pct)

    # Calculate median integrated drawdown
    median_integrated_drawdown = np.median(integrated_drawdown)

    final_bankroll_mode = calculate_binned_mode(final_bankrolls, config['RUIN_THRESHOLD'])
    target_achieved_count = np.sum(np.any(bankroll_histories >= config['TARGET_BANKROLL'], axis=1))
    busted_runs = np.sum(np.any(bankroll_histories <= config['RUIN_THRESHOLD'], axis=1))
    risk_of_ruin_percent = (busted_runs / config['NUMBER_OF_SIMULATIONS']) * 100
    percentiles = {p: np.percentile(final_bankrolls, p) for p in [2.5, 5, 25, 50, 75, 95, 97.5]}
    median_growth_rate = (percentiles[50] - config['STARTING_BANKROLL_EUR']) / config['STARTING_BANKROLL_EUR'] if config['STARTING_BANKROLL_EUR'] > 0 else 0.0

    # --- Calculate Withdrawal Metrics ---
    median_total_withdrawn = np.median(final_withdrawn)
    p95_total_withdrawn = np.percentile(final_withdrawn, 95)

    # --- Calculate Total Return ---
    # Total return is the change in bankroll plus everything that was taken out.
    total_return_per_sim = (final_bankrolls - config['STARTING_BANKROLL_EUR']) + final_withdrawn
    median_total_return = np.median(total_return_per_sim)


    return {
        'final_bankrolls': final_bankrolls, 'median_final_bankroll': percentiles[50],
        'final_bankroll_mode': final_bankroll_mode, 'growth_rate': median_growth_rate,
        'risk_of_ruin': risk_of_ruin_percent, 'p5': percentiles[5], 'p2_5': percentiles[2.5],
        'bankroll_histories': bankroll_histories, 'hands_histories': total_hands_histories,
        'median_history': np.median(bankroll_histories, axis=0),
        'hands_history': np.mean(total_hands_histories, axis=0),
        'target_prob': (target_achieved_count / config['NUMBER_OF_SIMULATIONS']) * 100,
        'above_threshold_hit_counts': {rule["threshold"]: np.sum(np.any(bankroll_histories >= rule["threshold"], axis=1))
                                      for rule in strategy_obj.rules if rule["threshold"] > config['STARTING_BANKROLL_EUR']},
        'below_threshold_drop_counts': {rule["threshold"]: np.sum(np.any(bankroll_histories <= rule["threshold"], axis=1))
                                       for rule in strategy_obj.rules if rule["threshold"] < config['STARTING_BANKROLL_EUR']},
        'percentile_win_rates': percentile_win_rates, 'risk_of_demotion': risk_of_demotion,
        'hands_distribution_pct': hands_distribution_pct,
        'final_stake_distribution': final_stake_distribution,
        'final_highest_stake_distribution': final_highest_stake_distribution,
        'median_max_downswing': median_max_downswing,
        'p95_max_downswing': p95_max_downswing,
        'max_downswings': max_drawdowns,
        'median_rakeback_eur': median_rakeback_eur,
        'median_profit_from_play_eur': median_profit_from_play_eur,
        'median_hands_played': median_hands_played,
        'median_stop_losses': median_stop_losses,
        'median_time_underwater_pct': median_time_underwater_pct,
        'median_integrated_drawdown': median_integrated_drawdown,
        'average_assigned_win_rates': average_assigned_win_rates,
        'avg_assigned_wr_per_sim': avg_assigned_wr_per_sim,
        'median_run_assigned_wr': median_run_assigned_wr,
        'median_total_withdrawn': median_total_withdrawn,
        'p95_total_withdrawn': p95_total_withdrawn,
        'median_total_return': median_total_return,
        'total_withdrawn_histories': total_withdrawn_histories,
        'downswing_analysis': (lambda: {
            'depth_probabilities': {
                int(t): p for t, p in zip(
                    config.get('DOWNSWING_DEPTH_THRESHOLDS_BB', []),
                    np.mean(downswing_depth_exceeded, axis=0) * 100
                )
            } if downswing_depth_exceeded is not None and len(config.get('DOWNSWING_DEPTH_THRESHOLDS_BB', [])) > 0 else {},
            'duration_probabilities': {
                int(t): p for t, p in zip(
                    config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', []),
                    np.mean(downswing_duration_exceeded, axis=0) * 100
                )
            } if downswing_duration_exceeded is not None and len(config.get('DOWNSWING_DURATION_THRESHOLDS_HANDS', [])) > 0 else {}
        })(),
    }