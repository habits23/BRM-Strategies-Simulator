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

import analysis
import reporting

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
    # This is the critical step: Generate one "long-term luck factor" per simulation run.
    long_term_luck_factors = rng.normal(loc=0.0, scale=1.0, size=config['NUMBER_OF_SIMULATIONS'])

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

        all_win_rates[name] = calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, long_term_luck_factors, prior_win_rate, config)

    return all_win_rates, rng


def calculate_hand_block_outcome(current_bankrolls, proportions_per_stake, all_win_rates, rng, active_mask, config):
    """Calculates the profit and hands played for a block of hands across all simulations."""
    # This function models SHORT-TERM LUCK (session-to-session variance).
    # It takes the pre-calculated `all_win_rates` (the Assigned WR, which includes long-term luck)
    # and adds a new random variance component for this specific block of hands.
    # The final result of the simulation, after accumulating all these short-term results,
    # is the "Play WR" or "Realized WR".

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

            # 1. Profit from Skill + Long-Term Luck (the pre-calculated Assigned WR)
            # This component is deterministic for a given simulation run.
            profit_from_ev_bb = all_win_rates[name][proportions_mask] * num_100_hand_blocks

            # 2. Profit from Short-Term Variance (the "Session Luck Factor")
            # A new random number is generated for every block to simulate session-to-session luck.
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

def _calculate_withdrawal_amounts(current_bankrolls, bankroll_at_start_of_month, settings, due_mask, ruin_threshold):
    """
    Calculates the withdrawal amount for each simulation based on its strategy.
    This is a helper function for the main simulation loops.
    """
    withdrawal_amounts = np.zeros_like(current_bankrolls)

    # Safety check: only withdraw if bankroll is above the minimum threshold specified in the UI
    eligible_mask = due_mask & (current_bankrolls >= settings["min_bankroll"])
    if not np.any(eligible_mask):
        return withdrawal_amounts

    strategy = settings["strategy"]
    value = settings["value"]

    calculated_withdrawals = np.zeros_like(current_bankrolls)

    if "Fixed Amount" in strategy:
        calculated_withdrawals[eligible_mask] = value

    elif "Percentage of Profits" in strategy:
        profits = current_bankrolls - bankroll_at_start_of_month
        # Only withdraw from positive profits
        positive_profit_mask = eligible_mask & (profits > 0)
        if np.any(positive_profit_mask):
            calculated_withdrawals[positive_profit_mask] = profits[positive_profit_mask] * (value / 100.0)

    elif "Withdraw Down to Threshold" in strategy:
        amount_over_threshold = current_bankrolls - value
        # Only withdraw if bankroll is over the threshold
        over_threshold_mask = eligible_mask & (amount_over_threshold > 0)
        if np.any(over_threshold_mask):
            calculated_withdrawals[over_threshold_mask] = amount_over_threshold[over_threshold_mask]

    # Final check: Don't let withdrawal cause bankroll to go below ruin threshold.
    # This is a safety net, though min_br_for_withdrawal should usually prevent this.
    max_possible_withdrawal = current_bankrolls - ruin_threshold
    final_withdrawals = np.minimum(calculated_withdrawals, max_possible_withdrawal)
    final_withdrawals[final_withdrawals < 0] = 0 # Ensure withdrawals are not negative

    withdrawal_amounts[eligible_mask] = final_withdrawals[eligible_mask]
    return withdrawal_amounts

def calculate_effective_win_rate(ev_bb_per_100, std_dev_per_100, sample_hands, long_term_luck_factors, prior_win_rate, config):
    """Adjusts the observed EV win rate using a Bayesian-inspired 'shrinkage' method."""
    # This function calculates the "Assigned WR" which includes LONG-TERM LUCK.
    # It's called once per stake at the beginning of the simulation.
    # 1. It creates a "skill_estimate_wr" by blending the user's input with a prior, weighted by sample size.
    # 2. It calculates the uncertainty ("std_error") in that skill estimate.
    # 3. It applies the `long_term_luck_factors` (one persistent random number per simulation) scaled by the uncertainty.
    # The result is the "true" win rate for a given simulation run.

    # The `PRIOR_SAMPLE_SIZE` has two distinct effects:
    # 1. The "Shrinkage" Effect: It determines how much weight to give the user's input vs. the prior (the model's extrapolation).
    #    A high prior means the model is more skeptical of the user's input if the `sample_hands` is low.
    # 2. The "Uncertainty" Effect: It's added to the user's `sample_hands` to form a total "pool of evidence".
    #    A larger total pool results in a smaller `std_error`, meaning less long-term luck is applied.

    # --- A Note on a Common Point of Confusion ---
    # It may seem that a low prior (more trust in user input) should lead to less variance in Assigned WR.
    # This is not the case. The two effects work differently:
    # - "Trust" (Shrinkage) affects the *center* of the Assigned WR distribution. A low prior centers the distribution on the user's input.
    # - "Uncertainty" (Variance) affects the *width* of the distribution. A low prior creates a small "pool of evidence",
    #   which means the model is very uncertain about its estimate, resulting in a *wider* distribution of long-term luck.
    # Conversely, a high prior creates a large evidence pool, high certainty, and a *narrower* distribution.

    if sample_hands > 0:
        # Effect 1: Calculate the weight for the "shrinkage" effect.
        data_weight = sample_hands / (sample_hands + config['PRIOR_SAMPLE_SIZE'])
        skill_estimate_wr = (data_weight * ev_bb_per_100) + ((1 - data_weight) * prior_win_rate)
    else:
        model_extrapolation = prior_win_rate
        user_estimate = ev_bb_per_100
        # This is the model's best guess at your "true skill" for a stake with no data.
        skill_estimate_wr = (config['ZERO_HANDS_INPUT_WEIGHT'] * user_estimate) + ((1 - config['ZERO_HANDS_INPUT_WEIGHT']) * model_extrapolation)

    # For a true sanity check, we can bypass the luck factor entirely.
    if config.get('PRIOR_SAMPLE_SIZE') >= 10_000_000:
        # The issue is that for the first stake, skill_estimate_wr is a float.
        # We must ensure it's an array of the correct size for all simulations.
        if isinstance(skill_estimate_wr, (int, float)):
            return np.full_like(long_term_luck_factors, skill_estimate_wr)
        return skill_estimate_wr # It's already an array for subsequent stakes

    # Effect 2: Calculate the total "pool of evidence" to determine uncertainty.
    effective_sample_size_for_variance = sample_hands + config['PRIOR_SAMPLE_SIZE']
    N_blocks = max(1.0, effective_sample_size_for_variance / 100.0)
    std_error = std_dev_per_100 / np.sqrt(N_blocks)

    # This is the "long-term luck" component for the entire simulation run.
    long_term_luck_adjustment = long_term_luck_factors * std_error

    # The final Assigned WR is the skill estimate plus the long-term luck.
    return skill_estimate_wr + long_term_luck_adjustment

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

    # --- Withdrawal Initialization ---
    withdrawal_settings = config.get("WITHDRAWAL_SETTINGS", {"enabled": False})
    total_withdrawn_histories = np.zeros((num_sims, num_checks + 1), dtype=float)
    if withdrawal_settings.get("enabled"):
        hands_since_last_withdrawal = np.zeros(num_sims, dtype=int)
        # Store the bankroll at the start of a "month" to calculate profit-based withdrawals
        bankroll_at_last_withdrawal = np.full(num_sims, config['STARTING_BANKROLL_EUR'], dtype=float)

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
        
        # Update bankrolls with play profits first, before withdrawal calculations
        temp_bankrolls = current_bankrolls + block_profits_eur

        # --- Withdrawal Logic ---
        withdrawal_amounts_this_block = np.zeros(num_sims, dtype=float)
        if withdrawal_settings.get("enabled"):
            total_hands_this_block = np.sum(list(hands_per_stake_this_block.values()), axis=0)
            hands_since_last_withdrawal += np.where(active_mask, total_hands_this_block, 0)

            due_for_withdrawal_mask = (hands_since_last_withdrawal >= withdrawal_settings["monthly_volume"]) & active_mask
            if np.any(due_for_withdrawal_mask):
                withdrawal_amounts_this_block = _calculate_withdrawal_amounts(
                    temp_bankrolls, bankroll_at_last_withdrawal, withdrawal_settings,
                    due_for_withdrawal_mask, config['RUIN_THRESHOLD']
                )
                temp_bankrolls -= withdrawal_amounts_this_block
                hands_since_last_withdrawal[due_for_withdrawal_mask] = 0
                bankroll_at_last_withdrawal[due_for_withdrawal_mask] = temp_bankrolls[due_for_withdrawal_mask]

        total_withdrawn_histories[:, i+1] = total_withdrawn_histories[:, i] + withdrawal_amounts_this_block
        new_bankrolls = temp_bankrolls

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

    return bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns_so_far, stop_loss_triggers, underwater_hands_count, total_withdrawn_histories

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

    # --- Withdrawal Initialization ---
    withdrawal_settings = config.get("WITHDRAWAL_SETTINGS", {"enabled": False})
    total_withdrawn_histories = np.zeros((num_sims, num_checks + 1), dtype=float)
    if withdrawal_settings.get("enabled"):
        hands_since_last_withdrawal = np.zeros(num_sims, dtype=int)
        # Store the bankroll at the start of a "month" to calculate profit-based withdrawals
        bankroll_at_last_withdrawal = np.full(num_sims, config['STARTING_BANKROLL_EUR'], dtype=float)

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

        # Update bankrolls with play profits first, before withdrawal calculations
        temp_bankrolls = current_bankrolls + block_profits_eur

        withdrawal_amounts_this_block = np.zeros(num_sims, dtype=float)
        if withdrawal_settings.get("enabled"):
            total_hands_this_block = np.sum(list(hands_per_stake_this_block.values()), axis=0)
            hands_since_last_withdrawal += np.where(active_mask, total_hands_this_block, 0)

            due_for_withdrawal_mask = (hands_since_last_withdrawal >= withdrawal_settings["monthly_volume"]) & active_mask
            if np.any(due_for_withdrawal_mask):
                withdrawal_amounts_this_block = _calculate_withdrawal_amounts( # noqa
                    temp_bankrolls, bankroll_at_last_withdrawal, withdrawal_settings,
                    due_for_withdrawal_mask, config['RUIN_THRESHOLD']
                )
                temp_bankrolls -= withdrawal_amounts_this_block
                hands_since_last_withdrawal[due_for_withdrawal_mask] = 0
                bankroll_at_last_withdrawal[due_for_withdrawal_mask] = temp_bankrolls[due_for_withdrawal_mask]

        total_withdrawn_histories[:, i+1] = total_withdrawn_histories[:, i] + withdrawal_amounts_this_block
        new_bankrolls = temp_bankrolls
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

    return bankroll_history, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns_so_far, stop_loss_triggers, underwater_hands_count, total_withdrawn_histories
# =================================================================================
#   PLOTTING AND REPORTING FUNCTIONS
# =================================================================================

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
        f"Initial Table Mix: {reporting.get_initial_table_mix_string(strategy_obj, config)}",
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

    def _write_income_analysis(res):
        lines = []
        # Only add this section if withdrawals were enabled for the run
        if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
            lines.extend(["", "--- Income & Total Return Analysis ---"])
            lines.append("Metrics related to money taken out of the bankroll and overall value generated.")
            median_withdrawn = res.get('median_total_withdrawn', 0.0)
            p95_withdrawn = res.get('p95_total_withdrawn', 0.0)
            median_return = res.get('median_total_return', 0.0)
            lines.append(f"   - Median Total Withdrawn: €{median_withdrawn:,.2f}")
            lines.append(f"   - 95th Percentile Withdrawn: €{p95_withdrawn:,.2f} (5% of runs withdrew more than this)")
            lines.append(f"   - Median Total Return: €{median_return:,.2f} ((Final BR - Start BR) + Withdrawn)")
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
    report_lines.extend(_write_income_analysis(result))
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
    header = ['Strategy', 'Page', 'Median Final BR', 'Med Withdrawn', 'Med Total Return', 'Median Growth', 'Median Rakeback', 'RoR (%)', 'Target Prob (%)', '5th %ile']

    # Conditionally add the stop-loss header
    if config.get("STOP_LOSS_BB", 0) > 0:
        header.insert(7, 'Median SL') # Insert after Median Rakeback

    cell_text = []
    for strategy_name, result in all_results.items():
        row = [
            strategy_name,
            str(strategy_page_map.get(strategy_name, '-')),
            f"€{result['median_final_bankroll']:,.2f}",
            f"€{result.get('median_total_withdrawn', 0.0):,.2f}",
            f"€{result.get('median_total_return', 0.0):,.2f}",
            f"{result['growth_rate']:.2%}",
            f"€{result.get('median_rakeback_eur', 0.0):,.2f}",
            f"{result['risk_of_ruin']:.2f}",
            f"{result['target_prob']:.2f}",
            f"€{result['p5']:,.2f}"
        ]

        # Conditionally add the stop-loss value to the row
        if config.get("STOP_LOSS_BB", 0) > 0:
            row.insert(7, f"{result.get('median_stop_losses', 0):.1f}")

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

    # Page counting: Title(1) + Summary(1) + Analysis(1) + CompPlots(5) = 8 pages before details
    # The number of comparison plots increases by 1 if withdrawals are enabled.
    num_comparison_plots = 4
    if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
        num_comparison_plots += 1

    page_counter_for_map = 2 + num_comparison_plots + (1 if analysis_report else 0)
    lines_per_page = 45

    for strategy_name, result in all_results.items():
        strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
        strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA']) # noqa

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

    # Create the color map once for the entire PDF generation process
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    color_map = {name: colors[i] for i, name in enumerate(all_results.keys())}

    with PdfPages(pdf_buffer) as pdf:
        create_title_page(pdf, timestamp_str)
        save_summary_table_to_pdf(pdf, all_results, strategy_page_map, config)

        if analysis_report:
            write_analysis_report_to_pdf(pdf, analysis_report)

        reporting.plot_median_progression_comparison(all_results, config, color_map=color_map, pdf=pdf)
        reporting.plot_final_bankroll_comparison(all_results, config, color_map=color_map, pdf=pdf)
        reporting.plot_total_withdrawn_comparison(all_results, config, color_map=color_map, pdf=pdf)
        reporting.plot_time_underwater_comparison(all_results, config, color_map=color_map, pdf=pdf)
        reporting.plot_risk_reward_scatter(all_results, config, color_map=color_map, pdf=pdf)

        for strategy_name, result in all_results.items():
            strategy_config = config['STRATEGIES_TO_RUN'][strategy_name]
            strategy_obj = initialize_strategy(strategy_name, strategy_config, config['STAKES_DATA'])
            page_num = strategy_page_map.get(strategy_name, 0)
            report_lines_for_writing = get_strategy_report_lines(strategy_name, result, strategy_obj, config)
            write_strategy_report_to_pdf(pdf, report_lines_for_writing, start_page_num=page_num)
            reporting.plot_strategy_progression(result['bankroll_histories'], result['hands_histories'], strategy_name, config, pdf=pdf)
            reporting.plot_final_bankroll_distribution(result['final_bankrolls'], result, strategy_name, config, pdf=pdf, color_map=color_map)
            reporting.plot_assigned_wr_distribution(
                result['avg_assigned_wr_per_sim'],
                result['median_run_assigned_wr'],
                weighted_input_wr,
                strategy_name,
                pdf=pdf)
            reporting.plot_max_downswing_distribution(result['max_downswings'], result, strategy_name, pdf=pdf, color_map=color_map)

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

    # --- New Metric Calculation: Efficiency Score ---
    for name, res in all_results.items():
        # Calculate a risk-adjusted return score. Higher is better.
        # Use Median Total Return if withdrawals are on, otherwise use simple growth.
        if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
            total_value_generated = res.get('median_total_return', 0)
        else:
            total_value_generated = res['median_final_bankroll'] - config['STARTING_BANKROLL_EUR']

        pain = res['p95_max_downswing']
        if pain > 1: # Avoid division by zero or tiny numbers
            # We add 1 to value to avoid issues with 0 and to slightly penalize negative values.
            res['efficiency_score'] = (total_value_generated + 1) / pain
        else:
            # If there's no significant downswing, it's extremely efficient. Give it a very high score, proportional to value.
            res['efficiency_score'] = total_value_generated if total_value_generated > 0 else 1.0

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

    best_efficiency, _ = find_best_worst_with_ties('efficiency_score', higher_is_better=True)
    if best_efficiency:
        names = f"'{best_efficiency[0]}'" if len(best_efficiency) == 1 else f"'{', '.join(best_efficiency)}'"
        verb = "demonstrated" if len(best_efficiency) == 1 else "were tied for demonstrating"
        insights.append(f"\n**⚡ Most Efficient:** The **{names}** strategy {verb} the best risk-adjusted return. It generated the most 'total value' (growth + withdrawals) for the amount of 'pain' (downswing) it caused, making it a highly efficient choice.")

    # Add insight for withdrawals if enabled
    if config.get("WITHDRAWAL_SETTINGS", {}).get("enabled"):
        best_income, worst_income = find_best_worst_with_ties('median_total_withdrawn', higher_is_better=True)
        if best_income:
            names = f"'{best_income[0]}'" if len(best_income) == 1 else f"'{', '.join(best_income)}'"
            verb = "generated" if len(best_income) == 1 else "were tied for generating"
            insights.append(f"\n**💸 Best Income Generator:** The **{names}** strategy {verb} the most income through withdrawals, with a median of €{all_results[best_income[0]]['median_total_withdrawn']:,.0f}. This is a key metric for players who need to live off their winnings.")
        if worst_income and worst_income not in best_income and all_results[worst_income]['median_total_withdrawn'] < all_results[best_income[0]]['median_total_withdrawn']:
            insights.append(f"\n**🏦 Lowest Income Generator:** **'{worst_income}'** generated the least income (€{all_results[worst_income]['median_total_withdrawn']:,.0f}). This might be because it was too conservative to generate profits to withdraw, or too risky, leading to frequent downswings that prevented withdrawals.")

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

    # --- New Section: Actionable Recommendations ---
    insights.append("\n### Actionable Recommendations")
    recommendations_made = False

    # 1. Identify Dominated Strategies
    dominated_strategies = {}
    strategy_names = list(all_results.keys())
    for i in range(len(strategy_names)):
        for j in range(len(strategy_names)):
            if i == j: continue
            strat_a_name, strat_b_name = strategy_names[i], strategy_names[j]
            res_a, res_b = all_results[strat_a_name], all_results[strat_b_name]

            # Strategy B dominates A if it's better/equal on all key metrics, and strictly better on at least one.
            b_is_better = (res_b['median_final_bankroll'] >= res_a['median_final_bankroll'] and
                           res_b['risk_of_ruin'] <= res_a['risk_of_ruin'] and
                           res_b['p95_max_downswing'] <= res_a['p95_max_downswing'])
            b_is_strictly_better = (res_b['median_final_bankroll'] > res_a['median_final_bankroll'] or
                                    res_b['risk_of_ruin'] < res_a['risk_of_ruin'] or
                                    res_b['p95_max_downswing'] < res_a['p95_max_downswing'])
            if b_is_better and b_is_strictly_better:
                dominated_strategies.setdefault(strat_a_name, []).append(strat_b_name)

    if dominated_strategies:
        insights.append("\n**Consider Retiring Dominated Strategies:**")
        for dominated, dominators in dominated_strategies.items():
            dominator_str = f"'{dominators[0]}'" if len(dominators) == 1 else f"'{', '.join(dominators)}'"
            insights.append(f"- The **'{dominated}'** strategy appears to be inefficient. It is outperformed by **{dominator_str}**, which offer(s) a better combination of reward and safety. There is likely no reason to choose '{dominated}'.")
        recommendations_made = True

    # 2. Provide Prescriptive Advice
    advice_given = set() # To avoid giving the same advice twice
    if worst_ror and worst_ror not in best_rors and 'risky' not in advice_given:
        insights.append(f"\n**Tuning the Riskiest Strategy:**\n- To make the **'{worst_ror}'** strategy safer, consider increasing its bankroll thresholds. This will make it move up stakes more slowly, reducing its high Risk of Ruin ({all_results[worst_ror]['risk_of_ruin']:.2f}%).")
        advice_given.add('risky'); recommendations_made = True

    if conservative_strats and 'conservative' not in advice_given:
        c_strat_name = conservative_strats[0]
        if all_results[c_strat_name]['target_prob'] < np.mean([res['target_prob'] for res in all_results.values()]) - 10:
            insights.append(f"\n**Tuning the Most Conservative Strategy:**\n- The **'{c_strat_name}'** strategy may be too conservative and limiting your growth. Consider lowering its bankroll thresholds to allow it to move up to more profitable stakes sooner.")
            advice_given.add('conservative'); recommendations_made = True

    if not recommendations_made:
        insights.append("\nAll strategies appear to be well-balanced with clear trade-offs. Your choice depends on your personal risk tolerance.")

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
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns, stop_loss_triggers, underwater_hands_count, total_withdrawn_histories = run_sticky_simulation_vectorized(strategy_obj, all_win_rates, rng, stake_level_map, config)
        else:
            bankroll_histories, hands_per_stake_histories, rakeback_histories, peak_stake_levels, demotion_flags, max_drawdowns, stop_loss_triggers, underwater_hands_count, total_withdrawn_histories = run_multiple_simulations_vectorized(strategy_obj, all_win_rates, rng, stake_level_map, config)

        # Analyze the results and store them
        all_results[strategy_name] = analysis.analyze_strategy_results(
            strategy_name, strategy_obj, bankroll_histories, hands_per_stake_histories, rakeback_histories, all_win_rates, rng,
            peak_stake_levels, demotion_flags, stake_level_map, stake_name_map, max_drawdowns, stop_loss_triggers, underwater_hands_count, total_withdrawn_histories, config
        )

    # Generate the final qualitative analysis report
    analysis_report = generate_qualitative_analysis(all_results, config)

    return {
        "results": all_results,
        "analysis_report": analysis_report
    }