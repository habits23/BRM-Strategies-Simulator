import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from scipy.stats import t, gaussian_kde
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import io

# =================================================================================
#   BANKROLL MANAGEMENT STRATEGY CLASS
# =================================================================================

class BankrollManagementStrategy:
    """
    A class to represent a bankroll management strategy with defined thresholds.
    """
    def __init__(self, name, strategy_rules, stakes_data):
        """
        Initializes the strategy with a name, a list of rules, and the stakes data for validation.
        """
        self.name = name
        self.rules = sorted(strategy_rules, key=lambda x: x["threshold"], reverse=True)
        self._validate_rules(stakes_data)

    def _validate_rules(self, stakes_data):
        """
        Central validation method to run all checks on the strategy rules.
        """
        self._validate_sorted_thresholds()
        self._validate_rule_content(stakes_data)

    def _validate_sorted_thresholds(self):
        """
        Ensures that the thresholds in the rules are sorted in descending order.
        """
        for i in range(len(self.rules) - 1):
            if self.rules[i]["threshold"] < self.rules[i+1]["threshold"]:
                raise ValueError(
                    f"Strategy '{self.name}' rules are not sorted correctly. "
                    "Thresholds must be in descending order."
                )

    def _validate_rule_content(self, stakes_data):
        """
        Checks that each rule's content is valid.
        """
        valid_stakes = {stake["name"] for stake in stakes_data}
        for rule in self.rules:
            if "threshold" not in rule or "tables" not in rule:
                raise ValueError(f"Strategy '{self.name}' rule is missing 'threshold' or 'tables'.")
            for stake_name, count in rule["tables"].items():
                if stake_name not in valid_stakes:
                    raise ValueError(f"Strategy '{self.name}' references invalid stake: '{stake_name}'. Valid: {valid_stakes}")
                if isinstance(count, str):
                    sanitized_val = count.replace('%', '').replace(' ', '')
                    if "-" in sanitized_val:
                        parts = sanitized_val.split('-')
                        if len(parts) != 2: raise ValueError(f"Invalid percentage range '{count}' in '{self.name}'.")
                        try:
                            low, high = map(float, parts)
                            if not (0 <= low <= high <= 100): raise ValueError(f"Percentage range '{count}' invalid in '{self.name}'.")
                        except ValueError: raise ValueError(f"Invalid percentage range format '{count}' in '{self.name}'.")
                    else:
                        try:
                            if not (0 <= float(sanitized_val) <= 100): raise ValueError(f"Percentage '{count}' invalid in '{self.name}'.")
                        except ValueError: raise ValueError(f"Invalid fixed percentage format '{count}' in '{self.name}'.")
                elif not (isinstance(count, int) and count >= 0):
                     raise ValueError(f"Strategy '{self.name}' has invalid table count for '{stake_name}'. Must be non-negative int or percentage string.")

    def get_table_mix(self, bankroll):
        """
        Finds the correct table mix for a given bankroll.
        If the bankroll is below the lowest threshold, it defaults to the mix of the lowest threshold rule.
        """
        for rule in self.rules:
            if bankroll >= rule["threshold"]:
                return rule["tables"]
        # If no rule is met (bankroll is below the lowest threshold),
        # return the table mix of the lowest threshold rule.
        if self.rules:
            return self.rules[-1]["tables"]
        return {}

    def get_rules_as_vectors(self):
        """
        Returns the thresholds and table mixes as separate lists, sorted.
        """
        thresholds = [rule["threshold"] for rule in self.rules]
        tables = [rule["tables"] for rule in self.rules]
        return thresholds, tables

class HysteresisStrategy(BankrollManagementStrategy):
    """
    A specialized strategy class for hysteresis-based ("sticky") bankroll management.
    """
    def __init__(self, name, num_buy_ins, stakes_data):
        self.num_buy_ins = num_buy_ins
        self.stakes_data = sorted(stakes_data, key=lambda x: x["bb_size"])
        # Generate rules before calling super().__init__ so they are ready for validation
        rules = self._generate_rules()
        super().__init__(name, rules, stakes_data)

    def _generate_rules(self):
        """
        Generate rules for moving between stakes based on buy-in requirements.
        """
        stake_thresholds = []
        for i in range(len(self.stakes_data) - 1, -1, -1):
            current_stake = self.stakes_data[i]
            if isinstance(self.num_buy_ins, dict):
                # Use the specific value, or a default of 40 if a stake is somehow missing.
                buy_ins_for_stake = self.num_buy_ins.get(current_stake['name'], 40)
            else:
                # Use the global integer value.
                buy_ins_for_stake = self.num_buy_ins
            threshold = buy_ins_for_stake * 100 * current_stake["bb_size"]
            tables_config = {s["name"]: "0%" for s in self.stakes_data}
            tables_config[current_stake["name"]] = "100%"
            stake_thresholds.append({
                "threshold": threshold,
                "tables": tables_config,
                "stake_name": current_stake["name"]
            })
        return sorted(stake_thresholds, key=lambda x: x["threshold"], reverse=True)

# =================================================================================
#   SIMULATION LOGIC AND HELPER FUNCTIONS
# =================================================================================

def setup_simulation_parameters(config, seed):
    """Initializes and returns common parameters needed for simulations."""
    rng = np.random.default_rng(seed)
    all_win_rates = {}
    # This is the critical fix: Generate one "luck factor" per simulation run.
    luck_factor = rng.normal(loc=0.0, scale=1.0, size=config['NUMBER_OF_SIMULATIONS'])

    for i, stake in enumerate(config['STAKES_DATA']):
        name = stake["name"]
        ev_bb_per_100 = stake.get("ev_bb_per_100", stake["bb_per_100"])
        std_dev_per_100 = stake["std_dev_per_100"]
        sample_hands = stake["sample_hands"]
        win_rate_drop = stake["win_rate_drop"]

        if i > 0:
            previous_stake_name = config['STAKES_DATA'][i-1]["name"]
            # The prior is the entire array of win rates from the previous stake.
            # This is CRITICAL to preserve the per-simulation luck correlation across stakes.
            prior_win_rate = all_win_rates[previous_stake_name] - win_rate_drop
        else:
            prior_win_rate = ev_bb_per_100

        all_win_rates[name] = calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, luck_factor, prior_win_rate, config)

    return all_win_rates, rng


def calculate_hand_block_outcome(current_bankrolls, proportions_per_stake, all_win_rates, rng, active_mask, config):
    """Calculates the profit and hands played for a block of hands across all simulations."""
    session_profits_eur = np.zeros_like(current_bankrolls)
    hands_per_stake_this_session = {stake["name"]: np.zeros_like(current_bankrolls, dtype=int) for stake in config['STAKES_DATA']}

    for stake in config['STAKES_DATA']:
        name = stake["name"]
        bb_size = stake["bb_size"]
        std_dev_per_100 = stake["std_dev_per_100"]

        # Define the mask based on which simulations are active AND have a proportion for this stake
        proportions_mask = active_mask & (proportions_per_stake[name] > 0)
        if np.any(proportions_mask):
            proportions = proportions_per_stake[name][proportions_mask]
            hands_for_stake = (proportions * config['HANDS_PER_CHECK']).astype(int)
            hands_per_stake_this_session[name][proportions_mask] = hands_for_stake
            
            num_100_hand_blocks = hands_for_stake / 100.0

            # Separate EV and Variance components
            # EV scales linearly with volume
            profit_from_ev_bb = all_win_rates[name][proportions_mask] * num_100_hand_blocks

            # Variance scales with sqrt(volume)
            random_noise_component = rng.normal(loc=0.0, scale=1.0, size=np.sum(proportions_mask))
            profit_from_variance_bb = random_noise_component * std_dev_per_100 * np.sqrt(num_100_hand_blocks)

            total_profit_bb = profit_from_ev_bb + profit_from_variance_bb
            session_profits_eur[proportions_mask] += total_profit_bb * bb_size

    total_rakeback_eur = np.zeros_like(current_bankrolls)
    if config['RAKEBACK_PERCENTAGE'] > 0:
        for stake in config['STAKES_DATA']:
            if "rake_bb_per_100" in stake and np.any(hands_per_stake_this_session[stake["name"]] > 0):
                rake_paid_eur = (hands_per_stake_this_session[stake["name"]] / 100.0) * stake["rake_bb_per_100"] * stake["bb_size"]
                total_rakeback_eur += rake_paid_eur * config['RAKEBACK_PERCENTAGE']

    final_session_profit = session_profits_eur + total_rakeback_eur
    return final_session_profit, hands_per_stake_this_session, total_rakeback_eur

def calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, luck_factor, prior_win_rate, config):
    """Adjusts the observed EV win rate using a Bayesian-inspired 'shrinkage' method."""
    if sample_hands > 0:
        data_weight = sample_hands / (sample_hands + config['PRIOR_SAMPLE_SIZE'])
        shrunk_win_rate = (data_weight * ev_bb_per_100) + ((1 - data_weight) * prior_win_rate)
    else:
        model_extrapolation = prior_win_rate
        user_estimate = ev_bb_per_100
        shrunk_win_rate = (config['ZERO_HANDS_INPUT_WEIGHT'] * user_estimate) + ((1 - config['ZERO_HANDS_INPUT_WEIGHT']) * model_extrapolation)

    # For a true sanity check, we can bypass the luck factor entirely to get a perfect match.
    if config.get('PRIOR_SAMPLE_SIZE') >= 10_000_000:
        # The issue is that for the first stake, shrunk_win_rate is a float.
        # We must ensure it's an array of the correct size for all simulations.
        if isinstance(shrunk_win_rate, (int, float)):
            return np.full_like(luck_factor, shrunk_win_rate)
        return shrunk_win_rate # It's already an array for subsequent stakes

    effective_sample_size_for_variance = sample_hands + config['PRIOR_SAMPLE_SIZE']
    N_blocks = max(1.0, effective_sample_size_for_variance / 100.0)
    std_error = std_dev_per_100 / np.sqrt(N_blocks)
    # Apply the same luck factor to each stake, scaled by the stake's specific standard error.
    adjustment = luck_factor * std_error
    return shrunk_win_rate + adjustment

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

def sample_percentage_range(range_str, rng):
    """Given a range string like "40-60", return a random float in that range."""
    parts = range_str.split('-')
    low, high = map(float, parts)
    return rng.uniform(low, high)

def normalize_percentages(percentages):
    """Normalizes a dictionary of percentages so they sum to 1."""
    total = sum(percentages.values())
    if total == 0:
        return {k: 1.0 / len(percentages) for k in percentages}
    return {k: v / total for k, v in percentages.items()}

def resolve_proportions(rule, rng):
    """Resolves a strategy rule into a dictionary of proportions (floats summing to 1)."""
    percentages = {}
    fixed_ratios = {}
    for stake, val in rule.items():
        if isinstance(val, str):
            sanitized_val = val.replace('%', '').replace(' ', '')
            if "-" in sanitized_val:
                percentages[stake] = sample_percentage_range(sanitized_val, rng)
            else:
                try:
                    percentages[stake] = float(sanitized_val)
                except ValueError:
                    pass
        elif isinstance(val, int) and val > 0:
            fixed_ratios[stake] = val

    if fixed_ratios:
        return normalize_percentages(fixed_ratios)
    if percentages:
        return normalize_percentages(percentages)
    return {}

def get_initial_table_mix_string(strategy, config):
    """Helper function to describe the starting table mix as a string."""
    rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    if not rule:
        return "No Play"
    mix_parts = [f"{stake}: {value}" for stake, value in sorted(rule.items())]
    return ", ".join(mix_parts) if mix_parts else "No Play"

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
                stake_wrs['Realized WR (Play)'] = f"{(profit_from_play_eur / avg_bb_size) / (total_hands_for_sim / 100):.2f}"
                stake_wrs['Rakeback (bb/100)'] = f"{(final_rakeback[closest_sim_index] / avg_bb_size) / (total_hands_for_sim / 100):.2f}"
            else: # Handle case where avg_bb_size is 0
                stake_wrs['Realized WR (Play)'] = "N/A"
                stake_wrs['Rakeback (bb/100)'] = "N/A"
        
        percentile_win_rates[f"{name} Percentile"] = stake_wrs
    return percentile_win_rates

def analyze_strategy_results(strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng, peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, stop_loss_triggers, underwater_hands_count, config):
    """Takes the raw simulation output and calculates all the necessary metrics and analytics."""
    bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    total_hands_histories = np.sum(list(hands_per_stake_histories.values()), axis=0)
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
        # This part is for the detailed PDF report, so it's still needed.
        mix_str = ", ".join(f"{s}: {v}" for s, v in sorted(table_mix.items())) if table_mix else "No Play"
        final_stake_counts[mix_str] += 1

        # This is the more direct calculation for the UI summary.
        if not table_mix:
            final_highest_stake_counts["No Play"] += 1
        else:
            highest_stake = max(table_mix.keys(), key=lambda s: stake_order_map.get(s, -1))
            final_highest_stake_counts[highest_stake] += 1

    final_stake_distribution = {mix_str: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for mix_str, count in final_stake_counts.items()}
    final_highest_stake_distribution = {stake: (count / config['NUMBER_OF_SIMULATIONS']) * 100 for stake, count in final_highest_stake_counts.items()}

    percentile_win_rates = _calculate_percentile_win_rates(final_bankrolls, all_win_rates, hands_per_stake_histories, rakeback_histories, config, bb_size_map)

    median_max_downswing = np.median(max_drawdowns)
    p95_max_downswing = np.percentile(max_drawdowns, 95)

    # Calculate median rakeback
    final_rakeback = rakeback_histories[:, -1]
    median_rakeback_eur = np.median(final_rakeback)

    # Calculate median profit from play
    total_profit_per_sim = final_bankrolls - config['STARTING_BANKROLL_EUR']
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

    final_bankroll_mode = calculate_binned_mode(final_bankrolls, config['RUIN_THRESHOLD'])
    target_achieved_count = np.sum(np.any(bankroll_histories >= config['TARGET_BANKROLL'], axis=1))
    busted_runs = np.sum(np.any(bankroll_histories <= config['RUIN_THRESHOLD'], axis=1))
    risk_of_ruin_percent = (busted_runs / config['NUMBER_OF_SIMULATIONS']) * 100
    percentiles = {p: np.percentile(final_bankrolls, p) for p in [2.5, 5, 25, 50, 75, 95, 97.5]}
    median_growth_rate = (percentiles[50] - config['STARTING_BANKROLL_EUR']) / config['STARTING_BANKROLL_EUR'] if config['STARTING_BANKROLL_EUR'] > 0 else 0.0

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
        'average_assigned_win_rates': average_assigned_win_rates,
        'avg_assigned_wr_per_sim': avg_assigned_wr_per_sim,
        'median_run_assigned_wr': median_run_assigned_wr,
    }
def run_multiple_simulations_vectorized(strategy, all_win_rates, rng, stake_level_map, config):
    """
    Runs all simulations at once using vectorized NumPy operations for speed.
    This version dynamically resolves table mixes for each session.
    """
    num_sims = config['NUMBER_OF_SIMULATIONS']
    num_checks = int(np.ceil(config['TOTAL_HANDS_TO_SIMULATE'] / config['HANDS_PER_CHECK']))
    bankroll_history = np.full((num_sims, num_checks + 1), config['STARTING_BANKROLL_EUR'], dtype=float)
    hands_per_stake_histories = {stake['name']: np.zeros((num_sims, num_checks + 1), dtype=int) for stake in config['STAKES_DATA']}
    rakeback_histories = np.zeros((num_sims, num_checks + 1), dtype=float)

    # --- Maximum Drawdown Initialization ---
    peak_bankrolls_so_far = np.full(num_sims, config['STARTING_BANKROLL_EUR'], dtype=float)
    max_drawdowns_so_far = np.zeros(num_sims, dtype=float)

    # --- Stop-Loss Initialization ---
    # These must be initialized unconditionally to avoid NameError if stop-loss is disabled.
    is_stopped_out = np.zeros(num_sims, dtype=bool)
    stop_loss_triggers = np.zeros(num_sims, dtype=int)

    # --- Underwater Time Initialization ---
    underwater_hands_count = np.zeros(num_sims, dtype=int)

    # --- Demotion Tracking Initialization ---
    initial_rule = strategy.get_table_mix(config['STARTING_BANKROLL_EUR'])
    initial_stakes_with_tables = [stake for stake, count in initial_rule.items() if (isinstance(count, int) and count > 0) or (isinstance(count, str) and float(count.replace('%','').split('-')[0]) > 0)]
    initial_level = max([stake_level_map[s] for s in initial_stakes_with_tables]) if initial_stakes_with_tables else -1
    peak_stake_levels = np.full(num_sims, initial_level, dtype=int)
    demotion_flags = {level: np.zeros(num_sims, dtype=bool) for level in stake_level_map.values()}

    stake_bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    thresholds, rules = strategy.get_rules_as_vectors()

    for i in range(num_checks):
        current_bankrolls = bankroll_history[:, i]
        
        # Store previous peak levels to detect demotions
        previous_peak_levels = peak_stake_levels.copy()

        # Determine the table mix for each simulation based on its current bankroll
        proportions_per_stake = {stake["name"]: np.zeros(num_sims, dtype=float) for stake in config['STAKES_DATA']}
        
        remaining_mask = np.ones_like(current_bankrolls, dtype=bool)
        for threshold, rule in zip(thresholds, rules):
            current_mask = (current_bankrolls >= threshold) & remaining_mask
            if not np.any(current_mask):
                continue
            
            indices = np.where(current_mask)[0]
            for sim_idx in indices:
                resolved_proportions = resolve_proportions(rule, rng)
                for stake_name, prop in resolved_proportions.items():
                    proportions_per_stake[stake_name][sim_idx] = prop
            remaining_mask[current_mask] = False

        # Handle simulations with bankroll below the lowest threshold by applying the lowest rule
        if np.any(remaining_mask) and rules:
            lowest_rule = rules[-1] # The last rule is the lowest threshold
            indices = np.where(remaining_mask)[0]
            for sim_idx in indices:
                resolved_proportions = resolve_proportions(lowest_rule, rng)
                for stake_name, prop in resolved_proportions.items():
                    proportions_per_stake[stake_name][sim_idx] = prop

        # --- Demotion Tracking Logic for the current block ---
        current_levels = np.full(config['NUMBER_OF_SIMULATIONS'], -1, dtype=int)
        for stake_name, level in stake_level_map.items():
            has_tables_mask = proportions_per_stake[stake_name] > 0
            current_levels[has_tables_mask] = np.maximum(current_levels[has_tables_mask], level)

        demotion_from_peak_mask = current_levels < previous_peak_levels
        for level in stake_level_map.values():
            if level > 0:
                demoted_this_session_mask = (previous_peak_levels == level) & demotion_from_peak_mask
                demotion_flags[level][demoted_this_session_mask] = True

        # Update peak levels for the next session
        peak_stake_levels = np.maximum(previous_peak_levels, current_levels)

        total_proportions = sum(proportions_per_stake.values())
        # Determine who is active this block (not ruined, has a table mix, and is not stopped out)
        active_mask = (current_bankrolls >= config['RUIN_THRESHOLD']) & (total_proportions > 0) & ~is_stopped_out

        # Reset the stopped out flag for the next loop. Any sim that was stopped out can now play again.
        is_stopped_out.fill(False)

        if not np.any(active_mask):
            bankroll_history[:, i+1:] = bankroll_history[:, i][:, np.newaxis]
            for stake_name in hands_per_stake_histories:
                hands_per_stake_histories[stake_name][:, i+1:] = hands_per_stake_histories[stake_name][:, i][:, np.newaxis]
            break

        block_profits_eur, hands_per_stake_this_block, block_rakeback_eur = calculate_hand_block_outcome(
            current_bankrolls, proportions_per_stake, all_win_rates, rng, active_mask, config
        )

        # --- Stop-Loss Logic ---
        if config.get("STOP_LOSS_BB", 0) > 0:
            # Find the bb_size of the highest stake played in this block for each sim
            highest_bb_size = np.zeros(num_sims)
            for stake_name, bb_size in stake_bb_size_map.items():
                played_this_stake_mask = proportions_per_stake[stake_name] > 0
                highest_bb_size[played_this_stake_mask] = np.maximum(highest_bb_size[played_this_stake_mask], bb_size)
            
            stop_loss_eur = config["STOP_LOSS_BB"] * highest_bb_size
            
            # Only trigger for active simulations that have a valid stop-loss amount
            valid_stop_loss_mask = active_mask & (stop_loss_eur > 0)
            
            # Calculate profit from play only (excluding rakeback) for the stop-loss check.
            profit_from_play = block_profits_eur - block_rakeback_eur
            triggered_mask = (profit_from_play < -stop_loss_eur) & valid_stop_loss_mask
            
            if np.any(triggered_mask):
                is_stopped_out[triggered_mask] = True
                stop_loss_triggers[triggered_mask] += 1

        new_bankrolls = current_bankrolls + block_profits_eur
        bankroll_history[:, i+1] = np.where(active_mask, new_bankrolls, current_bankrolls)

        # --- Underwater Time Calculation ---
        underwater_mask = (bankroll_history[:, i+1] < peak_bankrolls_so_far) & active_mask
        underwater_hands_count[underwater_mask] += config['HANDS_PER_CHECK']

        # --- Maximum Drawdown Calculation ---
        peak_bankrolls_so_far = np.maximum(peak_bankrolls_so_far, bankroll_history[:, i+1])
        current_drawdowns = peak_bankrolls_so_far - bankroll_history[:, i+1]
        max_drawdowns_so_far = np.maximum(max_drawdowns_so_far, current_drawdowns)

        rakeback_histories[:, i+1] = rakeback_histories[:, i] + np.where(active_mask, block_rakeback_eur, 0)
        for stake_name, hands_array in hands_per_stake_this_block.items():
            hands_per_stake_histories[stake_name][:, i+1] = hands_per_stake_histories[stake_name][:, i] + np.where(active_mask, hands_array, 0)

    return bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns_so_far, stop_loss_triggers, underwater_hands_count

def run_sticky_simulation_vectorized(strategy, all_win_rates, rng, stake_level_map, config):
    """
    Runs a simulation with a specific 'sticky' bankroll management strategy.
    This version correctly handles multiple stakes by implementing a proper state machine.
    """
    num_sims = config['NUMBER_OF_SIMULATIONS']
    num_checks = int(np.ceil(config['TOTAL_HANDS_TO_SIMULATE'] / config['HANDS_PER_CHECK']))
    bankroll_history = np.full((num_sims, num_checks + 1), config['STARTING_BANKROLL_EUR'], dtype=float)
    hands_per_stake_histories = {stake['name']: np.zeros((num_sims, num_checks + 1), dtype=int) for stake in config['STAKES_DATA']}
    rakeback_histories = np.zeros((num_sims, num_checks + 1), dtype=float)

    # --- Maximum Drawdown Initialization ---
    peak_bankrolls_so_far = np.full(num_sims, config['STARTING_BANKROLL_EUR'], dtype=float)
    max_drawdowns_so_far = np.zeros(num_sims, dtype=float)

    # --- Stop-Loss Initialization ---
    # These must be initialized unconditionally to avoid NameError if stop-loss is disabled.
    is_stopped_out = np.zeros(num_sims, dtype=bool)
    stop_loss_triggers = np.zeros(num_sims, dtype=int)

    # --- Underwater Time Initialization ---
    underwater_hands_count = np.zeros(num_sims, dtype=int)

    # --- Demotion Tracking Initialization ---
    stake_rules = sorted(strategy.rules, key=lambda r: r['threshold'])
    rule_index_to_level = {i: stake_level_map[rule['stake_name']] for i, rule in enumerate(stake_rules)}

    current_stake_indices = np.zeros(num_sims, dtype=int)
    for idx in range(len(stake_rules) - 1, -1, -1):
        threshold = stake_rules[idx]['threshold']
        current_stake_indices[config['STARTING_BANKROLL_EUR'] >= threshold] = idx

    initial_levels = np.vectorize(rule_index_to_level.get)(current_stake_indices)
    peak_stake_levels = initial_levels.copy()
    demotion_flags = {level: np.zeros(num_sims, dtype=bool) for level in stake_level_map.values()}
    stake_bb_size_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}

    for i in range(num_checks):
        current_bankrolls = bankroll_history[:, i]

        previous_peak_levels = peak_stake_levels.copy()

        for idx in range(len(stake_rules) - 1):
            move_up_threshold = stake_rules[idx+1]['threshold']
            can_move_up_mask = (current_stake_indices == idx) & (current_bankrolls >= move_up_threshold)
            current_stake_indices[can_move_up_mask] = idx + 1

        for j in range(len(stake_rules) - 1, 0, -1):
            # Hysteresis move-down: A player at stake 'j' only moves down to 'j-1'
            # if their bankroll drops below the entry threshold for stake 'i-1'.
            # This creates a "sticky" buffer zone.
            move_down_threshold = stake_rules[j-1]['threshold']
            must_move_down_mask = (current_stake_indices == j) & (current_bankrolls < move_down_threshold)
            current_stake_indices[must_move_down_mask] = j - 1

        # --- Demotion Tracking Logic ---
        current_levels = np.vectorize(rule_index_to_level.get)(current_stake_indices)

        demotion_from_peak_mask = current_levels < previous_peak_levels
        for level in stake_level_map.values():
            demoted_this_block_mask = (previous_peak_levels == level) & demotion_from_peak_mask
            demotion_flags[level][demoted_this_block_mask] = True

        # Update peak levels for the next session
        peak_stake_levels = np.maximum(previous_peak_levels, current_levels)

        proportions_per_stake = {stake["name"]: np.zeros(num_sims, dtype=float) for stake in config['STAKES_DATA']}

        for rule_idx in range(len(stake_rules)):
            at_this_stake_mask = (current_stake_indices == rule_idx)
            if not np.any(at_this_stake_mask):
                continue
            
            rule = stake_rules[rule_idx]['tables']
            resolved_proportions = resolve_proportions(rule, rng)
            for stake_name, prop in resolved_proportions.items():
                proportions_per_stake[stake_name][at_this_stake_mask] = prop

        total_proportions = sum(proportions_per_stake.values())
        # Determine who is active this block (not ruined, has a table mix, and is not stopped out)
        active_mask = (current_bankrolls >= config['RUIN_THRESHOLD']) & (total_proportions > 0) & ~is_stopped_out

        # Reset the stopped out flag for the next loop. Any sim that was stopped out can now play again.
        is_stopped_out.fill(False)
        
        if not np.any(active_mask):
            bankroll_history[:, i+1:] = bankroll_history[:, i][:, np.newaxis]
            for stake_name in hands_per_stake_histories:
                hands_per_stake_histories[stake_name][:, i+1:] = hands_per_stake_histories[stake_name][:, i][:, np.newaxis]
            break

        block_profits_eur, hands_per_stake_this_block, block_rakeback_eur = calculate_hand_block_outcome(
            current_bankrolls, proportions_per_stake, all_win_rates, rng, active_mask, config
        )

        # --- Stop-Loss Logic ---
        if config.get("STOP_LOSS_BB", 0) > 0:
            # Find the bb_size of the highest stake played in this block for each sim
            highest_bb_size = np.zeros(num_sims)
            for stake_name, bb_size in stake_bb_size_map.items():
                played_this_stake_mask = proportions_per_stake[stake_name] > 0
                highest_bb_size[played_this_stake_mask] = np.maximum(highest_bb_size[played_this_stake_mask], bb_size)
            
            stop_loss_eur = config["STOP_LOSS_BB"] * highest_bb_size
            
            # Only trigger for active simulations that have a valid stop-loss amount
            valid_stop_loss_mask = active_mask & (stop_loss_eur > 0)
            
            # Calculate profit from play only (excluding rakeback) for the stop-loss check.
            profit_from_play = block_profits_eur - block_rakeback_eur
            triggered_mask = (profit_from_play < -stop_loss_eur) & valid_stop_loss_mask
            
            if np.any(triggered_mask):
                is_stopped_out[triggered_mask] = True
                stop_loss_triggers[triggered_mask] += 1

        new_bankrolls = current_bankrolls + block_profits_eur
        bankroll_history[:, i+1] = np.where(active_mask, new_bankrolls, current_bankrolls)

        # --- Underwater Time Calculation ---
        underwater_mask = (bankroll_history[:, i+1] < peak_bankrolls_so_far) & active_mask
        underwater_hands_count[underwater_mask] += config['HANDS_PER_CHECK']

        # --- Maximum Drawdown Calculation ---
        peak_bankrolls_so_far = np.maximum(peak_bankrolls_so_far, bankroll_history[:, i+1])
        current_drawdowns = peak_bankrolls_so_far - bankroll_history[:, i+1]
        max_drawdowns_so_far = np.maximum(max_drawdowns_so_far, current_drawdowns)

        rakeback_histories[:, i+1] = rakeback_histories[:, i] + np.where(active_mask, block_rakeback_eur, 0)
        for stake_name, hands_array in hands_per_stake_this_block.items():
            hands_per_stake_histories[stake_name][:, i+1] = hands_per_stake_histories[stake_name][:, i] + np.where(active_mask, hands_array, 0)

    return bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns_so_far, stop_loss_triggers, underwater_hands_count
# =================================================================================
#   PLOTTING AND REPORTING FUNCTIONS
# =================================================================================

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

def plot_median_progression_comparison(all_results, config, pdf=None):
    """Compares the median bankroll progression for all strategies on a single plot."""
    fig, ax = plt.subplots(figsize=(8, 5)) # Made consistent with other compact plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for i, (strategy_name, result) in enumerate(all_results.items()):
        ax.plot(result['hands_history'], result['median_history'], label=strategy_name, linewidth=2.5, color=colors[i])

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

def plot_final_bankroll_distribution(final_bankrolls, result, strategy_name, config, pdf=None):
    """Creates a histogram of final bankrolls with key metrics highlighted."""
    # Filter out bankrolls that are too high to improve the visibility of the main data cluster.
    max_x_limit = np.percentile(final_bankrolls, 99.0)
    filtered_bankrolls = final_bankrolls[final_bankrolls <= max_x_limit]

    median_val = result['median_final_bankroll']
    # We only need the 5th and 95th percentiles for this cleaner plot.
    percentiles = {p: np.percentile(final_bankrolls, p) for p in [5, 95]}

    fig, ax = plt.subplots(figsize=(8, 5)) # Further reduced size for a more compact app layout
    ax.hist(filtered_bankrolls, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    # Key metrics
    ax.axvline(median_val, color='red', linestyle='dashed', linewidth=2, label=f'Median: €{median_val:,.2f}')
    ax.axvline(percentiles[5], color='darkred', linestyle=':', linewidth=2, label=f'5th Percentile: €{percentiles[5]:,.2f}')
    ax.axvline(percentiles[95], color='green', linestyle=':', linewidth=2, label=f'95th Percentile: €{percentiles[95]:,.2f}')

    # Contextual lines
    ax.axvline(config["STARTING_BANKROLL_EUR"], color='black', linewidth=2, label=f'Starting: €{config["STARTING_BANKROLL_EUR"]:,.0f}')
    ax.axvline(config["TARGET_BANKROLL"], color='gold', linestyle='-.', linewidth=2, label=f'Target: €{config["TARGET_BANKROLL"]:,.0f}')
    ax.set_title(f'Final Bankroll Distribution for {strategy_name}')
    ax.set_xlabel('Final Bankroll (EUR)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True)
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

def plot_final_bankroll_comparison(all_results, config, pdf=None):
    """
    Creates an overlapping density plot to compare the final bankroll distributions of all strategies.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

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

    for i, (strategy_name, result) in enumerate(all_results.items()):
        final_bankrolls = result['final_bankrolls']
        if len(final_bankrolls) > 1:
            try:
                # FIX: Filter the data before calculating the KDE.
                # This prevents extreme outliers (the top 1% we've already
                # excluded from the x-axis) from skewing the density plot.
                # The result is a plot that better represents the main body of the distribution.
                filtered_bankrolls = final_bankrolls[(final_bankrolls >= x_min) & (final_bankrolls <= x_max)]
                if len(filtered_bankrolls) < 2: # Need at least 2 points for KDE
                    ax.hist(final_bankrolls, bins=50, density=True, alpha=0.5, label=f"{strategy_name} (hist)")
                    continue

                kde = gaussian_kde(filtered_bankrolls)
                density = kde(x_grid)
                ax.plot(x_grid, density, label=strategy_name, color=colors[i], linewidth=2)
                ax.fill_between(x_grid, density, color=colors[i], alpha=0.1)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback for datasets that are not suitable for KDE
                ax.hist(final_bankrolls, bins=50, density=True, alpha=0.5, label=f"{strategy_name} (hist)")

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

def plot_max_downswing_distribution(max_downswings, result, strategy_name, pdf=None):
    """Creates a histogram of maximum downswings with key metrics highlighted."""
    if max_downswings is None or len(max_downswings) == 0:
        return plt.figure() # Return an empty figure if no data

    # Filter out extreme outliers for better visibility
    max_x_limit = np.percentile(max_downswings, 99.0)
    filtered_downswings = max_downswings[max_downswings <= max_x_limit]

    median_downswing = result['median_max_downswing']
    p95_downswing = result['p95_max_downswing']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(filtered_downswings, bins=50, color='salmon', edgecolor='black', alpha=0.7)

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

def plot_time_underwater_comparison(all_results, config, pdf=None):
    """
    Creates a bar chart comparing the median percentage of time each strategy
    spends 'underwater' (below a previous bankroll peak).
    """
    strategy_names = list(all_results.keys())
    underwater_pcts = [res.get('median_time_underwater_pct', 0) for res in all_results.values()]

    if not strategy_names:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Horizontal bar chart for better readability of strategy names
    bars = ax.barh(strategy_names, underwater_pcts, color=plt.cm.cividis(np.linspace(0.4, 0.9, len(strategy_names))))
    
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

def plot_risk_reward_scatter(all_results, config, pdf=None):
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
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategy_names)))
    
    ax.scatter(risk_values, reward_values, s=150, c=colors, alpha=0.7, edgecolors='w', zorder=10)

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

def write_analysis_report_to_pdf(pdf, analysis_report):
    """Writes the qualitative analysis report to a PDF page."""
    fig = plt.figure(figsize=(11, 8.5))
    
    lines = analysis_report.split('\n')
    y_pos = 0.90
    
    for line in lines:
        line = line.strip()
        if not line: # Add vertical space for empty lines
            y_pos -= 0.02
            continue

        font_size = 11
        font_weight = 'normal'
        x_pos = 0.05
        
        # Simple markdown parsing for PDF
        if line.startswith('### '):
            font_size = 16
            font_weight = 'bold'
            line = line[4:]
            y_pos -= 0.02 # Extra space before header
        elif line.startswith('- '):
            x_pos = 0.07
            line = f"• {line[2:]}" # Use a bullet point character
        
        # Strip emoji and bold markers for cleaner text
        line = line.replace('**', '').replace('🏆','').replace('📉','').replace('🛡️','').replace('🎲','').replace('🚀','').replace('😌','').replace('🎢','').replace('💰','').replace('⚠️','').replace('🧠','')

        fig.text(x_pos, y_pos, line, transform=fig.transFigure, size=font_size, weight=font_weight, va='top', ha='left', wrap=True)
        y_pos -= 0.04 # Fixed spacing between lines

    pdf.savefig(fig)
    plt.close(fig)

def create_title_page(pdf, timestamp):
    """Creates a title page for the PDF report."""
    fig = plt.figure(figsize=(11, 8.5))
    title = "Poker Bankroll Management Simulation Report"
    subtitle = f"Generated on: {timestamp}"
    fig.text(0.5, 0.6, title, ha='center', va='center', size=20, weight='bold')
    fig.text(0.5, 0.5, subtitle, ha='center', va='center', size=14)
    fig.text(0.5, 0.1, "Simulator: Final BR Simulator v1.5", ha='center', va='center', size=10, color='gray')
    pdf.savefig(fig)
    plt.close(fig)

def get_strategy_report_lines(strategy_name, result, strategy_obj, config):
    """Gathers all the text lines for a strategy's detailed report."""
    report_lines = [
        f"--- Detailed Report for: {strategy_name} ---",
        f"Initial Table Mix: {get_initial_table_mix_string(strategy_obj, config)}",
        ""
    ]
    # Helper functions for different sections of the report
    def _write_threshold_analysis(res):
        lines = ["--- Strategy Threshold Analysis ---"]
        if 'above_threshold_hit_counts' in res and res['above_threshold_hit_counts']:
            lines.append("Probability of hitting upper thresholds:")
            for threshold, count in sorted(res['above_threshold_hit_counts'].items(), reverse=True):
                lines.append(f"   - €{threshold:,.0f}: {(count / config['NUMBER_OF_SIMULATIONS']) * 100:.2f}%")
        if 'below_threshold_drop_counts' in res and res['below_threshold_drop_counts']:
            lines.append("Probability of dropping to lower thresholds:")
            for threshold, count in sorted(res['below_threshold_drop_counts'].items(), reverse=True):
                lines.append(f"   - €{threshold:,.0f}: {(count / config['NUMBER_OF_SIMULATIONS']) * 100:.2f}%")
        return lines

    def _write_hands_distribution(res):
        lines = []
        if 'hands_distribution_pct' in res and res['hands_distribution_pct']:
            lines.extend(["", "--- Hands Played Distribution ---", "Percentage of total hands played at each stake (averaged over all simulations):"])
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            sorted_stakes = sorted(res['hands_distribution_pct'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')))
            for stake_name, pct in sorted_stakes:
                if pct > 0.01:
                    lines.append(f"   - {stake_name}: {pct:.2f}%")
        return lines

    def _write_demotion_analysis(res):
        lines = []
        if 'risk_of_demotion' in res and res['risk_of_demotion']:
            lines.extend(["", "--- Risk of Demotion Analysis ---"])
            lines.append("Demotion Risk: Percentage of simulations that dropped to a lower stake after reaching a peak.")
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            sorted_demotions = sorted(res['risk_of_demotion'].items(), key=lambda item: stake_order_map.get(item[0], float('inf')), reverse=True)
            for stake_name, data in sorted_demotions:
                lines.append(f"   - From {stake_name}: {data['prob']:.2f}% (based on {data['reached_count']} simulations reaching this peak)")
        return lines

    def _write_downswing_analysis(res):
        lines = []
        if 'median_max_downswing' in res:
            lines.extend(["", "--- Maximum Downswing Analysis ---"])
            lines.append("A downswing is the largest single peak-to-trough drop in bankroll during a simulation.")
            median_downswing = res['median_max_downswing']
            p95_downswing = res['p95_max_downswing']
            lines.append(f"   - Median Downswing: €{median_downswing:.2f}")
            lines.append(f"   - 95th Percentile Downswing: €{p95_downswing:.2f} (5% of runs had a worse downswing)")
        return lines

    def _write_final_stake_distribution(res):
        lines = []
        if 'final_stake_distribution' in res and res['final_stake_distribution']:
            lines.extend(["", "--- Final Stake Distribution ---", "Percentage of simulations ending at each stake/table mix:"])
            sorted_dist = sorted(res['final_stake_distribution'].items(), key=lambda item: item[1], reverse=True)
            for mix_str, pct in sorted_dist:
                if pct > 0.01:
                    lines.append(f"   - {mix_str}: {pct:.2f}%")
        return lines

    def _write_final_highest_stake_distribution(res):
        lines = []
        if 'final_highest_stake_distribution' in res and res['final_highest_stake_distribution']:
            lines.extend(["", "--- Final Highest Stake Played ---", "Percentage of simulations ending with this as their highest active stake:"])
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            sorted_dist = sorted(res['final_highest_stake_distribution'].items(), key=lambda item: stake_order_map.get(item[0], -1), reverse=True)
            for stake, pct in sorted_dist:
                if pct > 0.01:
                    lines.append(f"   - {stake}: {pct:.2f}%")
        return lines

    def _write_final_bankroll_metrics(res):
        p2_5 = res.get('p2_5', 0.0)
        p5 = res.get('p5', 0.0)
        median = res.get('median_final_bankroll', 0.0)
        p95 = np.percentile(res.get('final_bankrolls', [0]), 95)
        p97_5 = np.percentile(res.get('final_bankrolls', [0]), 97.5)
        return [
            "", "--- Final Bankroll Metrics ---",
            f"Risk of Ruin: {res.get('risk_of_ruin', 0.0):.2f}% (Percentage of runs that dropped to or below €{config['RUIN_THRESHOLD']})",
            f"Probability of reaching €{config['TARGET_BANKROLL']}: {res.get('target_prob', 0.0):.2f}%",
            f"2.5th Percentile: €{p2_5:.2f} (97.5% of runs finished above this value)",
            f"5th Percentile:   €{p5:.2f} (95% of runs finished above this value)",
            f"50th Percentile:  €{median:.2f} (Median)",
            f"95th Percentile:  €{p95:.2f} (5% of runs finished above this value)",
            f"97.5th Percentile:€{p97_5:.2f} (2.5% of runs finished above this value)"
        ]

    def _write_win_rate_analysis(res):
        lines = ["", "--- Effective Win Rate Analysis (bb/100) ---", "Win rates for simulations ending at specific percentiles:"]
        for percentile_name, wr_data in sorted(res['percentile_win_rates'].items(), key=lambda item: item[1]['p_val']):
            play_wr = wr_data.get('Realized WR (Play)', 'N/A')
            rb_wr = wr_data.get('Rakeback (bb/100)', 'N/A')
            assigned_wr = wr_data.get('Assigned WR', 'N/A')
            primary_wr_str = f"Realized (Play): {play_wr}, Rakeback: {rb_wr}, Assigned: {assigned_wr}"
            stake_wr_string = ", ".join([f"{stake}: {wr}" for stake, wr in wr_data.items() if stake not in ['Realized WR (Play)', 'Rakeback (bb/100)', 'Assigned WR', 'p_val']])
            lines.extend([f"   - {percentile_name}:", f"     {primary_wr_str}", f"     (Assigned Stake Breakdown: {stake_wr_string})" if stake_wr_string else ""])
        return lines

    report_lines.extend(_write_threshold_analysis(result))
    report_lines.extend(_write_hands_distribution(result))
    report_lines.extend(_write_demotion_analysis(result))
    report_lines.extend(_write_downswing_analysis(result))
    report_lines.extend(_write_final_bankroll_metrics(result))
    report_lines.extend(_write_final_stake_distribution(result))
    report_lines.extend(_write_final_highest_stake_distribution(result))
    if 'percentile_win_rates' in result and result['percentile_win_rates']:
        report_lines.extend(_write_win_rate_analysis(result))
    return report_lines

def write_strategy_report_to_pdf(pdf, report_lines, start_page_num=None):
    """Writes the detailed text report for a single strategy to a PDF page, handling pagination."""
    lines_per_page = 45
    pages_of_lines = [report_lines[i:i + lines_per_page] for i in range(0, len(report_lines), lines_per_page)]

    for i, page_lines in enumerate(pages_of_lines):
        report_text = "\n".join(page_lines)
        fig = plt.figure(figsize=(11, 8.5))
        if start_page_num is not None:
            fig.text(0.95, 0.05, f"Page {start_page_num + i}", transform=fig.transFigure, size=8, va='bottom', ha='right', color='gray')
        fig.text(0.05, 0.95, report_text, transform=fig.transFigure, size=10, va='top', ha='left', fontfamily='monospace')
        pdf.savefig(fig)
        plt.close(fig)

def save_summary_table_to_pdf(pdf, all_results, strategy_page_map, config):
    """Creates a table of the main summary results and saves it to a PDF page."""
    header = ['Strategy', 'Page', 'Median Final BR', 'Mode Final BR', 'Median Growth', 'Median Rakeback', 'RoR (%)', 'Target Prob (%)', '5th %ile', '2.5th %ile']
    
    # Conditionally add the stop-loss header
    if config.get("STOP_LOSS_BB", 0) > 0:
        header.insert(6, 'Median SL') # Insert after Median Rakeback

    cell_text = []
    for strategy_name, result in all_results.items():
        row = [
            strategy_name,
            str(strategy_page_map.get(strategy_name, '-')),
            f"€{result['median_final_bankroll']:,.2f}",
            f"€{result['final_bankroll_mode']:,.2f}",
            f"{result['growth_rate']:.2%}",
            f"€{result.get('median_rakeback_eur', 0.0):,.2f}",
            f"{result['risk_of_ruin']:.2f}",
            f"{result['target_prob']:.2f}",
            f"€{result['p5']:,.2f}",
            f"€{result['p2_5']:,.2f}"
        ]
        
        # Conditionally add the stop-loss value to the row
        if config.get("STOP_LOSS_BB", 0) > 0:
            row.insert(6, f"{result.get('median_stop_losses', 0):.1f}")
        
        cell_text.append(row)

    if not cell_text: return

    col_widths = [len(h) for h in header]
    for row in cell_text:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    total_width = sum(col_widths)
    col_widths_ratio = [w / total_width for w in col_widths]

    fig, ax = plt.subplots(figsize=(16, 2 + len(cell_text) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=cell_text, colLabels=header, loc='center', cellLoc='center', colWidths=col_widths_ratio)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.5)
    ax.set_title('Strategy Comparison Summary', fontweight="bold", pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def generate_pdf_report(all_results, analysis_report, config, timestamp_str):
    """
    Generates the entire multi-page PDF report, writing to an in-memory buffer.
    """
    pdf_buffer = io.BytesIO()
    strategy_page_map = {}
    
    # Page counting: Title(1) + Summary(1) + Analysis(1) + CompPlots(4) = 7 pages before details
    page_counter_for_map = 7
    lines_per_page = 45

    for strategy_name, result in all_results.items():
        strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
        strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])
        
        # The page this strategy's report starts on
        strategy_page_map[strategy_name] = page_counter_for_map + 1
        
        report_lines = get_strategy_report_lines(strategy_name, result, strategy_obj, config)
        num_text_pages = (len(report_lines) + lines_per_page - 1) // lines_per_page # Ceiling division
        num_plot_pages = 4
        page_counter_for_map += num_text_pages + num_plot_pages

    # Calculate a representative input win rate for the plot's label
    total_sample_hands = sum(s['sample_hands'] for s in config['STAKES_DATA'])
    if total_sample_hands > 0:
        weighted_input_wr = sum(s['ev_bb_per_100'] * s['sample_hands'] for s in config['STAKES_DATA']) / total_sample_hands
    else:
        # Fallback if no sample hands are provided
        weighted_input_wr = config['STAKES_DATA'][0]['ev_bb_per_100'] if config['STAKES_DATA'] else 1.5

    with PdfPages(pdf_buffer) as pdf:
        create_title_page(pdf, timestamp_str)
        save_summary_table_to_pdf(pdf, all_results, strategy_page_map, config)
        
        if analysis_report:
            write_analysis_report_to_pdf(pdf, analysis_report)

        plot_median_progression_comparison(all_results, config, pdf=pdf)
        plot_final_bankroll_comparison(all_results, config, pdf=pdf)
        plot_time_underwater_comparison(all_results, config, pdf=pdf)
        plot_risk_reward_scatter(all_results, config, pdf=pdf)

        for strategy_name, result in all_results.items():
            strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
            strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])
            page_num = strategy_page_map.get(strategy_name, 0)
            report_lines_for_writing = get_strategy_report_lines(strategy_name, result, strategy_obj, config)
            write_strategy_report_to_pdf(pdf, report_lines_for_writing, start_page_num=page_num)
            plot_strategy_progression(result['bankroll_histories'], result['hands_histories'], strategy_name, config, pdf=pdf)
            plot_final_bankroll_distribution(result['final_bankrolls'], result, strategy_name, config, pdf=pdf)
            plot_assigned_wr_distribution(
                result['avg_assigned_wr_per_sim'],
                result['median_run_assigned_wr'],
                weighted_input_wr,
                strategy_name,
                pdf=pdf)
            plot_max_downswing_distribution(result['max_downswings'], result, strategy_name, pdf=pdf)

    pdf_buffer.seek(0)
    return pdf_buffer
# =================================================================================
#   QUALITATIVE ANALYSIS FUNCTIONS
# =================================================================================

def generate_qualitative_analysis(all_results, config):
    """Generates a human-readable analysis comparing the performance of different strategies."""
    insights = []
    num_strategies = len(all_results)

    if num_strategies < 2:
        insights.append("### Overall Summary")
        insights.append("Run at least two strategies to generate a comparative analysis.")
        return "\n".join(insights)

    def find_best_worst_with_ties(metric_key, higher_is_better=True):
        """Helper to find the best and worst performing strategies, handling ties for 'best'."""
        valid_results = {name: res for name, res in all_results.items() if metric_key in res}
        if not valid_results: return [], None
        
        sorted_strategies = sorted(valid_results.items(), key=lambda item: item[1][metric_key], reverse=higher_is_better)
        
        best_value = sorted_strategies[0][1][metric_key]
        best_strategies = [name for name, res in sorted_strategies if res[metric_key] == best_value]
        
        worst_name = sorted_strategies[-1][0]
        
        # If all strategies have the same value, there is no "worst"
        if best_value == sorted_strategies[-1][1][metric_key]:
            return best_strategies, None
            
        return best_strategies, worst_name

    insights.append("### Automated Strategy Analysis")
    insights.append("This analysis compares your strategies based on key performance indicators from the simulation.")
    
    best_medians, worst_median = find_best_worst_with_ties('median_final_bankroll', higher_is_better=True)
    if best_medians:
        names = f"'{best_medians[0]}'" if len(best_medians) == 1 else f"'{', '.join(best_medians)}'"
        verb = "achieved" if len(best_medians) == 1 else "were tied for"
        insights.append(f"\n**🏆 Best Typical Outcome:** The **{names}** strategy {verb} the highest median final bankroll (€{all_results[best_medians[0]]['median_final_bankroll']:,.0f}). This suggests it provides the most consistent growth for the average simulation run.")
    if worst_median and worst_median not in best_medians:
         insights.append(f"\n**📉 Worst Typical Outcome:** The **'{worst_median}'** strategy had the lowest median result (€{all_results[worst_median]['median_final_bankroll']:,.0f}). Check its Risk of Ruin and Downswing metrics to understand why.")

    best_rors, worst_ror = find_best_worst_with_ties('risk_of_ruin', higher_is_better=False)
    if best_rors:
        names = f"'{best_rors[0]}'" if len(best_rors) == 1 else f"'{', '.join(best_rors)}'"
        verb = "was" if len(best_rors) == 1 else "were tied for"
        insights.append(f"\n**🛡️ Safest Strategy:** With a Risk of Ruin of only {all_results[best_rors[0]]['risk_of_ruin']:.2f}%, **{names}** {verb} the least likely to go broke. This is ideal for risk-averse players.")
    if worst_ror and worst_ror not in best_rors:
        insights.append(f"\n**🎲 Riskiest Strategy:** **'{worst_ror}'** had the highest Risk of Ruin at {all_results[worst_ror]['risk_of_ruin']:.2f}%. This strategy is significantly more volatile.")

    best_targets, _ = find_best_worst_with_ties('target_prob', higher_is_better=True)
    if best_targets:
        names = f"'{best_targets[0]}'" if len(best_targets) == 1 else f"'{', '.join(best_targets)}'"
        verb = "gave" if len(best_targets) == 1 else "were tied for giving"
        insights.append(f"\n**🚀 Highest Upside:** If your main goal is to reach the target bankroll, the **{names}** strategy {verb} the best chance at {all_results[best_targets[0]]['target_prob']:.2f}%. This often comes with higher risk, so check its RoR.")

    best_downswings, worst_downswing = find_best_worst_with_ties('median_max_downswing', higher_is_better=False)
    if best_downswings:
        names = f"'{best_downswings[0]}'" if len(best_downswings) == 1 else f"'{', '.join(best_downswings)}'"
        verb = "had" if len(best_downswings) == 1 else "were tied for having"
        insights.append(f"\n**😌 Smoothest Ride:** The **{names}** strategy {verb} the smallest median downswing (€{all_results[best_downswings[0]]['median_max_downswing']:,.0f}), making it the least stressful to play.")
    if worst_downswing and worst_downswing not in best_downswings:
        insights.append(f"\n**🎢 Rollercoaster Ride:** Be prepared for significant swings with the **'{worst_downswing}'** strategy, which had the largest median downswing of €{all_results[worst_downswing]['median_max_downswing']:,.0f}.")

    # Add insight for highest rakeback earner
    if config.get("RAKEBACK_PERCENTAGE", 0) > 0:
        best_rb_strats, _ = find_best_worst_with_ties('median_rakeback_eur', higher_is_better=True)
        if best_rb_strats:
            names = f"'{best_rb_strats[0]}'" if len(best_rb_strats) == 1 else f"'{', '.join(best_rb_strats)}'"
            verb = "generated" if len(best_rb_strats) == 1 else "were tied for generating"
            insights.append(f"\n**💰 Highest Rakeback Earner:** The **{names}** strategy {verb} the most rakeback (€{all_results[best_rb_strats[0]]['median_rakeback_eur']:,.0f}), often by playing more hands at higher stakes.")

    # Add insight for stop-loss triggers
    if config.get("STOP_LOSS_BB", 0) > 0:
        most_sl_strats, _ = find_best_worst_with_ties('median_stop_losses', higher_is_better=True)
        if most_sl_strats and all_results[most_sl_strats[0]]['median_stop_losses'] > 0:
            names = f"'{most_sl_strats[0]}'" if len(most_sl_strats) == 1 else f"'{', '.join(most_sl_strats)}'"
            verb = "triggered" if len(most_sl_strats) == 1 else "were tied for triggering"
            median_sl_val = all_results[most_sl_strats[0]]['median_stop_losses']
            insights.append(f"\n**⚠️ Session Volatility:** The **{names}** strategy {verb} the stop-loss most often (median of {median_sl_val:.1f} times). This indicates it was more prone to large, single-session losses.")

    # Add insight for Time Underwater
    most_underwater_strats, _ = find_best_worst_with_ties('median_time_underwater_pct', higher_is_better=True)
    if most_underwater_strats:
        strat_name = most_underwater_strats[0]
        underwater_pct = all_results[strat_name]['median_time_underwater_pct']
        if underwater_pct > 60: # Threshold for a "high" amount of time underwater
            insights.append(f"\n**🧠 Psychological Cost:** The **'{strat_name}'** strategy spent the most time 'underwater' ({underwater_pct:.0f}% of hands). While potentially profitable, this indicates a psychologically demanding journey with long periods of grinding back to a previous peak.")

    insights.append("\n### Why Did They Perform This Way?")

    if worst_ror and best_targets and worst_ror in best_targets:
        insights.append(f"- The **'{worst_ror}'** strategy is a classic high-risk, high-reward approach. It achieved the highest probability of reaching the target, but also came with the highest Risk of Ruin. This is a trade-off between upside potential and safety.")
    
    # Analyze the contribution of rakeback to the best strategy's success
    if config.get("RAKEBACK_PERCENTAGE", 0) > 0 and best_medians:
        best_median_strat_name = best_medians[0]
        best_median_res = all_results[best_median_strat_name]
        median_profit = best_median_res['median_final_bankroll'] - config['STARTING_BANKROLL_EUR']
        median_rakeback = best_median_res['median_rakeback_eur']
        if median_profit > 0 and median_rakeback > 0:
            rb_contribution = (median_rakeback / median_profit) * 100
            if rb_contribution > 30:
                insights.append(f"- Rakeback was a critical factor for the top-performing **'{best_median_strat_name}'** strategy, accounting for **{rb_contribution:.0f}%** of its median profit. This strategy's success may be highly dependent on the rakeback deal.")
    elif config.get("RAKEBACK_PERCENTAGE", 0) == 0:
        insights.append("- With **0% rakeback**, all strategies are handicapped. This significantly reduces profitability and increases risk, especially for aggressive strategies that rely on moving up to high-rake environments.")
    
    if worst_median:
        worst_median_res = all_results[worst_median]
        hands_dist = worst_median_res.get('hands_distribution_pct', {})
        if hands_dist:
            stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
            played_stakes = [s for s, p in hands_dist.items() if p > 5]
            if played_stakes:
                highest_played_stake = max(played_stakes, key=lambda s: stake_order_map.get(s, -1))
                demotion_risks = worst_median_res.get('risk_of_demotion', {})
                if highest_played_stake in demotion_risks and demotion_risks[highest_played_stake]['prob'] > 40:
                    insights.append(f"- The **'{worst_median}'** strategy likely underperformed due to instability. It had a high **Risk of Demotion of {demotion_risks[highest_played_stake]['prob']:.1f}%** from {highest_played_stake}. This 'yo-yo effect' of moving up and down frequently is inefficient and can hurt long-term growth.")

    hysteresis_strats = [name for name in all_results if 'Hysteresis' in name or 'Sticky' in name]
    if hysteresis_strats:
        h_strat_name = hysteresis_strats[0]
        demotion_risks = all_results[h_strat_name].get('risk_of_demotion', {})
        if demotion_risks:
            avg_demotion_risk = np.mean([v['prob'] for v in demotion_risks.values()]) if demotion_risks else 100
            if avg_demotion_risk < 20:
                insights.append(f"- The **'{h_strat_name}'** strategy likely performed well by minimizing demotions. Its 'sticky' nature prevented players from dropping stakes during minor downswings, leading to more stable growth at higher limits.")

    # Insight for overly conservative strategies
    stake_order_map = {stake['name']: stake['bb_size'] for stake in config['STAKES_DATA']}
    conservative_strats = [name for name, res in all_results.items() if 'Conservative' in name]
    if conservative_strats:
        c_strat_name = conservative_strats[0]
        c_res = all_results[c_strat_name]
        if c_res['target_prob'] < np.mean([res['target_prob'] for res in all_results.values()]) - 10: # If target prob is >10% below average
            hands_dist = c_res.get('hands_distribution_pct', {})
            if hands_dist:
                lowest_stake_played = min(hands_dist.keys(), key=lambda s: stake_order_map.get(s, float('inf'))) if hands_dist else None
                if lowest_stake_played:
                    percent_at_lowest = hands_dist.get(lowest_stake_played, 0)
                    if percent_at_lowest > 80: # If it spends >80% of time at the lowest stake
                        display_percent = "over 99" if percent_at_lowest > 99 else f"over {percent_at_lowest:.0f}"
                        insights.append(f"- The **'{c_strat_name}'** strategy may be too conservative. It spent {display_percent}% of its time at {lowest_stake_played}, which significantly limited its growth and ability to reach the target.")

    return "\n".join(insights)

# =================================================================================
#   MAIN CONTROLLER FUNCTIONS
# =================================================================================

def initialize_strategy(strategy_name, strategy_config, stakes_data):
    """Initializes a strategy object from its configuration."""
    strategy_type = strategy_config.get("type", "standard")
    if strategy_type == "hysteresis":
        return HysteresisStrategy(strategy_name, strategy_config["num_buy_ins"], stakes_data)
    elif strategy_type == "standard":
        return BankrollManagementStrategy(strategy_name, strategy_config["rules"], stakes_data)
    else:
        raise ValueError(f"Unknown strategy type '{strategy_type}' for '{strategy_name}'")

def run_full_analysis(config):
    """
    The main entry point for running the entire simulation analysis.
    This function is called by the Streamlit app.
    """
    all_results = {}
    stake_level_map = {stake['name']: i for i, stake in enumerate(sorted(config['STAKES_DATA'], key=lambda s: s['bb_size']))}
    stake_name_map = {v: k for k, v in stake_level_map.items()}

    # Create a master RNG to generate unique, deterministic seeds for each strategy.
    # This ensures each strategy gets its own "deck of cards" while the overall
    # simulation remains reproducible from the main seed.
    master_rng = np.random.default_rng(config['SEED'])

    for strategy_name, strategy_config in config['STRATEGIES_TO_RUN'].items():
        strategy_seed = master_rng.integers(1, 1_000_000_000)
        all_win_rates, rng = setup_simulation_parameters(config, strategy_seed)
        strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])

        # Determine which simulation function to use based on the class type
        if isinstance(strategy_obj, HysteresisStrategy):
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns, stop_loss_triggers, underwater_hands_count = run_sticky_simulation_vectorized(strategy_obj, all_win_rates, rng, stake_level_map, config)
        else:
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns, stop_loss_triggers, underwater_hands_count = run_multiple_simulations_vectorized(strategy_obj, all_win_rates, rng, stake_level_map, config)

        # Analyze the results and store them
        all_results[strategy_name] = analyze_strategy_results(
            strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng,
            peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, stop_loss_triggers, underwater_hands_count, config
        )

    # Generate the final qualitative analysis report
    analysis_report = generate_qualitative_analysis(all_results, config)

    return {
        "results": all_results,
        "analysis_report": analysis_report
    }
