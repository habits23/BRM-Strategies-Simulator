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

def _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, config, bb_size_map):
    """Calculates assigned and realized win rates for simulations closest to key percentiles."""
    percentile_win_rates = {}
    percentiles_to_find = {'2.5th': 2.5, '5th': 5, '25th': 25, 'Median': 50, '75th': 75, '95th': 95, '97.5th': 97.5}
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
                total_profit_eur = final_bankrolls[closest_sim_index] - config['STARTING_BANKROLL_EUR']
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

def _calculate_assigned_wr_metrics(final_bankrolls, all_win_rates, hands_per_stake_histories, config):
    """Calculates metrics related to the assigned win rates (long-term luck)."""
    final_hands_per_stake = {name: history[:, -1] for name, history in hands_per_stake_histories.items()}
    total_hands_per_sim = np.sum(list(final_hands_per_stake.values()), axis=0)

    weighted_assigned_wr_sum = np.zeros(config['NUMBER_OF_SIMULATIONS'])
    for stake_name, wr_array in all_win_rates.items():
        weighted_assigned_wr_sum += wr_array * final_hands_per_stake[stake_name]

    avg_assigned_wr_per_sim = np.divide(
        weighted_assigned_wr_sum,
        total_hands_per_sim,
        out=np.zeros_like(weighted_assigned_wr_sum),
        where=total_hands_per_sim != 0
    )

    median_final_bankroll_val = np.percentile(final_bankrolls, 50)
    median_sim_index = np.argmin(np.abs(final_bankrolls - median_final_bankroll_val))
    median_run_assigned_wr = avg_assigned_wr_per_sim[median_sim_index]

    average_assigned_win_rates = {name: np.mean(wr_array) for name, wr_array in all_win_rates.items()}

    return {
        'average_assigned_win_rates': average_assigned_win_rates,
        'avg_assigned_wr_per_sim': avg_assigned_wr_per_sim,
        'median_run_assigned_wr': median_run_assigned_wr,
    }

def _calculate_hands_distribution(hands_per_stake_histories):
    """Calculates the percentage distribution of hands played across all stakes."""
    total_hands_per_stake = {name: np.sum(history[:, -1]) for name, history in hands_per_stake_histories.items()}
    grand_total_hands = sum(total_hands_per_stake.values())
    hands_distribution_pct = {name: (total / grand_total_hands) * 100 for name, total in total_hands_per_stake.items()} if grand_total_hands > 0 else {}
    return {'hands_distribution_pct': hands_distribution_pct}

def _calculate_final_stake_distributions(final_bankrolls, strategy_obj, config):
    """Calculates the distribution of final stakes and highest stakes reached."""
    final_stake_counts = defaultdict(int)
    final_highest_stake_counts = defaultdict(int)
    stake_order_map = {stake['name']: i for i, stake in enumerate(sorted(config['STAKES_DATA'], key=lambda s: s['bb_size']))}

    for br in final_bankrolls:
        table_mix = strategy_obj.get_table_mix(br)
        mix_str = ", ".join(f"{s}: {v}" for s, v in sorted(table_mix.items())) if table_mix else "No Play"
        final_stake_counts[mix_str] += 1

        if not table_mix:
            final_highest_stake_counts["No Play"] += 1
        else:
            active_stakes_in_mix = {
                stake: value for stake, value in table_mix.items()
                if (isinstance(value, str) and float(value.replace('%','').split('-')[0]) > 0) or (isinstance(value, int) and value > 0)
            }
            if not active_stakes_in_mix:
                final_highest_stake_counts["No Play"] += 1
            else:
                highest_stake = max(active_stakes_in_mix.keys(), key=lambda s: stake_order_map.get(s, -1))
                final_highest_stake_counts[highest_stake] += 1

    return {
        'final_stake_distribution': {mix_str: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for mix_str, count in final_stake_counts.items()},
        'final_highest_stake_distribution': {stake: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for stake, count in final_highest_stake_counts.items()}
    }

def _calculate_risk_of_demotion(peak_stake_levels, demotion_flags, stake_name_map):
    """Calculates the probability of being demoted from each stake level."""
    risk_of_demotion = {}
    for level, stake_name in stake_name_map.items():
        if level > 0:
            sims_that_reached_peak = np.sum(peak_stake_levels >= level)
            if sims_that_reached_peak > 0:
                sims_demoted_from_peak = np.sum(demotion_flags[level])
                risk_of_demotion[stake_name] = {
                    'prob': (sims_demoted_from_peak / sims_that_reached_peak) * 100,
                    'reached_count': sims_that_reached_peak
                }
    return {'risk_of_demotion': risk_of_demotion}

def _calculate_performance_metrics(max_drawdowns, rakeback_histories, final_bankrolls, stop_loss_triggers, hands_per_stake_histories, underwater_hands_count, config):
    """Calculates various performance metrics like downswings, rakeback, profit, etc."""
    final_rakeback = rakeback_histories[:, -1]
    total_profit_per_sim = final_bankrolls - config['STARTING_BANKROLL_EUR']
    profit_from_play_per_sim = total_profit_per_sim - final_rakeback

    final_hands_per_stake = {name: history[:, -1] for name, history in hands_per_stake_histories.items()}
    total_hands_per_sim = np.sum(list(final_hands_per_stake.values()), axis=0)
    median_hands_played = np.median(total_hands_per_sim)

    time_underwater_pct = np.divide(
        underwater_hands_count,
        total_hands_per_sim,
        out=np.zeros_like(underwater_hands_count, dtype=float),
        where=total_hands_per_sim != 0
    ) * 100

    return {
        'median_max_downswing': np.median(max_drawdowns),
        'p95_max_downswing': np.percentile(max_drawdowns, 95),
        'max_downswings': max_drawdowns,
        'median_rakeback_eur': np.median(final_rakeback),
        'median_profit_from_play_eur': np.median(profit_from_play_per_sim),
        'median_stop_losses': np.median(stop_loss_triggers) if stop_loss_triggers is not None else 0,
        'median_hands_played': median_hands_played,
        'median_time_underwater_pct': np.median(time_underwater_pct),
    }

def _calculate_summary_stats(final_bankrolls, bankroll_histories, strategy_obj, config):
    """Calculates high-level summary statistics like RoR, target probability, and percentiles."""
    final_bankroll_mode = calculate_binned_mode(final_bankrolls, config['RUIN_THRESHOLD'])
    target_achieved_count = np.sum(np.any(bankroll_histories >= config['TARGET_BANKROLL'], axis=1))
    busted_runs = np.sum(np.any(bankroll_histories <= config['RUIN_THRESHOLD'], axis=1))
    percentiles = {p: np.percentile(final_bankrolls, p) for p in [2.5, 5, 25, 50, 75, 95, 97.5]}

    return {
        'median_final_bankroll': percentiles[50],
        'final_bankroll_mode': final_bankroll_mode,
        'growth_rate': (percentiles[50] - config['STARTING_BANKROLL_EUR']) / config['STARTING_BANKROLL_EUR'] if config['STARTING_BANKROLL_EUR'] > 0 else 0.0,
        'risk_of_ruin': (busted_runs / config['NUMBER_OF_SIMULATIONS']) * 100,
        'p5': percentiles[5],
        'p2_5': percentiles[2.5],
        'target_prob': (target_achieved_count / config['NUMBER_OF_SIMULATIONS']) * 100,
        'above_threshold_hit_counts': {rule["threshold"]: np.sum(np.any(bankroll_histories >= rule["threshold"], axis=1))
                                      for rule in strategy_obj.rules if rule["threshold"] > config['STARTING_BANKROLL_EUR']},
        'below_threshold_drop_counts': {rule["threshold"]: np.sum(np.any(bankroll_histories <= rule["threshold"], axis=1))
                                       for rule in strategy_obj.rules if rule["threshold"] < config['STARTING_BANKROLL_EUR']},
    }

def analyze_strategy_results(strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng, peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, stop_loss_triggers, underwater_hands_count, config):
    """Takes the raw simulation output and calculates all the necessary metrics and analytics by calling helper functions."""
    # --- Initial Setup ---
    bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    total_hands_histories = np.sum(list(hands_per_stake_histories.values()), axis=0)
    final_bankrolls = bankroll_histories[:, -1]

    # --- Call Helper Functions to Calculate Metrics ---
    assigned_wr_metrics = _calculate_assigned_wr_metrics(final_bankrolls, all_win_rates, hands_per_stake_histories, config)
    hands_dist_metrics = _calculate_hands_distribution(hands_per_stake_histories)
    final_stake_metrics = _calculate_final_stake_distributions(final_bankrolls, strategy_obj, config)
    demotion_metrics = _calculate_risk_of_demotion(peak_stake_levels, demotion_flags, stake_name_map)
    performance_metrics = _calculate_performance_metrics(max_drawdowns, rakeback_histories, final_bankrolls, stop_loss_triggers, hands_per_stake_histories, underwater_hands_count, config)
    summary_stats = _calculate_summary_stats(final_bankrolls, bankroll_histories, strategy_obj, config)
    percentile_win_rates = _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, config, bb_size_map)

    # --- Assemble Final Results Dictionary ---
    results = {
        'final_bankrolls': final_bankrolls,
        'bankroll_histories': bankroll_histories,
        'hands_histories': total_hands_histories,
        'median_history': np.median(bankroll_histories, axis=0),
        'hands_history': np.mean(total_hands_histories, axis=0),
        'percentile_win_rates': percentile_win_rates,
        **assigned_wr_metrics,
        **hands_dist_metrics,
        **final_stake_metrics,
        **demotion_metrics,
        **performance_metrics,
        **summary_stats,
    }
    return results