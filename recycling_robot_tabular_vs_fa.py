"""
Recycling Robot: From Tabular to Function Approximation
Sutton & Barto Example 3.3 - Complete Implementation

Author: Waruni Liyanapathirana
Date: 25.02.2026
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PART A: 2-STATE TABULAR MDP
# ============================================

@dataclass
class TabularRecyclingRobot:
    """Exact 2-state MDP from Sutton & Barto Example 3.3"""
    
    alpha: float = 0.8
    beta: float = 0.6
    gamma: float = 0.9
    r_search: float = 2.0
    r_wait: float = 1.0
    r_rescue: float = -3.0
    
    def __post_init__(self):
        self.states = ['high', 'low']
        self.actions = {
            'high': ['search', 'wait'],
            'low': ['search', 'wait', 'recharge']
        }
        self._build_transitions()
        self._compute_expected_rewards()
    
    def _build_transitions(self):
        """Build transition probability and reward matrices"""

        a = self.alpha
        b = self.beta
        
        # Transition probabilities
        self.P = {s: {} for s in self.states}
        # Rewards for transitions
        self.R = {s: {} for s in self.states}
        
        # High
        # If high and search
        #   -> high with prob alpha, reward r_search
        #   -> low  with prob 1-alpha, reward r_search
        self.P['high']['search'] = [('high', a), ('low', 1 - a)]
        self.R['high']['search'] = {'high': self.r_search, 'low': self.r_search}
        
        # If high and wait
        #   -> high with prob 1, reward r_wait
        self.P['high']['wait'] = [('high', 1.0)]
        self.R['high']['wait'] = {'high': self.r_wait}
        
        # Low
        # If low and search
        #   -> low  with prob beta, reward r_search
        #   -> high with prob 1-beta, reward r_rescue
        self.P['low']['search'] = [('low', b), ('high', 1 - b)]
        self.R['low']['search'] = {'low': self.r_search, 'high': self.r_rescue}
        
        # If low and wait
        #   -> low with prob 1, reward r_wait
        self.P['low']['wait'] = [('low', 1.0)]
        self.R['low']['wait'] = {'low': self.r_wait}
        
        # If low and recharge
        #   -> high with prob 1, reward 0
        self.P['low']['recharge'] = [('high', 1.0)]
        self.R['low']['recharge'] = {'high': 0.0}
    
    def _compute_expected_rewards(self):
        """Compute r(s,a) = Σ p(s'|s,a) * r(s,a,s')"""

        self.r_sa = {s: {} for s in self.states}
        
        for s in self.states:
            for a in self.actions[s]:
                exp_r = 0.0
                for s_next, p in self.P[s][a]:
                    exp_r += p * self.R[s][a][s_next]
                self.r_sa[s][a] = exp_r
    
    def value_iteration(self, theta=1e-6, max_iter=1000):
        """Implement tabular value iteration"""

        # initialize values
        V = {s: 0.0 for s in self.states}
        delta_history = []
        
        for it in range(1, max_iter + 1):
            delta = 0.0
            V_new = {}
            
            for s in self.states:
                best_q = -np.inf
                
                for a in self.actions[s]:
                    q = 0.0
                    for s_next, p in self.P[s][a]:
                        r = self.R[s][a][s_next]
                        q += p * (r + self.gamma * V[s_next])
                    
                    best_q = max(best_q, q)
                
                V_new[s] = best_q
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            delta_history.append(delta)
            
            if delta < theta:
                break
        
        policy = self.extract_policy(V)
        return V, policy, it, delta_history
    
    def extract_policy(self, V):
        """Extract greedy policy from value function"""
        
        policy = {}
        
        for s in self.states:
            best_a = None
            best_q = -np.inf
            
            for a in self.actions[s]:
                q = 0.0
                for s_next, p in self.P[s][a]:
                    r = self.R[s][a][s_next]
                    q += p * (r + self.gamma * V[s_next])
                
                if q > best_q:
                    best_q = q
                    best_a = a
            
            policy[s] = best_a
        
        return policy

# ============================================
# PART B: CONTINUOUS MDP WITH CANS
# ============================================

class ContinuousRecyclingEnv:
    """
    Continuous battery level b ∈ [0, 2]
    Stochastic can arrivals, deterministic transitions
    """
    
    def __init__(self, 
                 gamma=0.9,
                 p_can=0.4,
                 can_lambda=1.5,
                 can_value=1.0,
                 rescue_threshold=0.2,
                 rescue_penalty=-3.0,
                 max_steps=50):
        
        self.gamma = gamma
        self.p_can = p_can
        self.can_lambda = can_lambda
        self.can_value = can_value
        self.rescue_threshold = rescue_threshold
        self.rescue_penalty = rescue_penalty
        self.max_steps = max_steps
        
        # Battery consumption rates
        self.battery_delta = {
            'search': -0.2,
            'wait': -0.05,
            'recharge': 0.4
        }
        
        # Base rewards
        self.base_reward = {
            'search': 2.0,
            'wait': 1.0,
            'recharge': 0.0
        }
        
        # Collection efficiency
        self.efficiency = {
            'search': 1.0,
            'wait': 0.3,
            'recharge': 0.0
        }
        
        self.reset()
    
    def reset(self):
        """Start new episode"""
        self.battery = 2.0
        self.steps = 0
        self.total_reward = 0
        self.rescues = 0
        self.cans_collected = 0
        return self.battery
    
    def get_available_actions(self, battery):
        """Return actions available at current battery level"""
        return ['search', 'wait', 'recharge']
    
    def generate_cans(self):
        """Generate cans according to Poisson process"""
        if np.random.rand() < self.p_can:
            return int(np.random.poisson(self.can_lambda))
        return 0
    
    def get_next_state(self, battery, action):
        """Compute deterministic next state"""
        b = float(battery)

        # Rescue condition
        if action == 'search' and b < self.rescue_threshold:
            return 0.5, True

        delta = self.battery_delta[action]
        b_next = b + delta

        # clip to [0,2]
        b_next = max(0.0, min(2.0, b_next))
        return b_next, False
    
    def get_expected_reward(self, battery, action):
        """
        Compute expected reward for (b, a)
        E[r] = base_reward + E[collection] + E[rescue_penalty]
        """
        b = float(battery)

        base = self.base_reward[action]

        # Expected cans per step
        expected_cans = self.p_can * self.can_lambda
        expected_collection = expected_cans * self.can_value * self.efficiency[action]

        rescue = 0.0
        if action == 'search' and b < self.rescue_threshold:
            rescue = self.rescue_penalty

        return base + expected_collection + rescue
    
    def step(self, action):
        """Execute action, return (next_state, reward, done, info)"""
        assert action in self.get_available_actions(self.battery), "Invalid action"

        # stochastic cans
        n_cans = self.generate_cans()
        collection_reward = n_cans * self.can_value * self.efficiency[action]

        # deterministic transition + rescue
        b_next, rescued = self.get_next_state(self.battery, action)

        # reward this step
        reward = self.base_reward[action] + collection_reward
        if rescued:
            reward += self.rescue_penalty

        # update episode stats
        self.steps += 1
        self.total_reward += reward
        self.cans_collected += n_cans
        if rescued:
            self.rescues += 1

        self.battery = b_next
        done = self.steps >= self.max_steps

        info = {"cans": n_cans, "rescued": rescued}
        return b_next, reward, done, info


class DiscretizedTabularSolver:
    """
    Discretize continuous MDP into N bins
    Provides ground truth for comparison
    """
    
    def __init__(self, n_bins=1000, gamma=0.9):
        self.n_bins = n_bins
        self.gamma = gamma
        self.bins = np.linspace(0, 2, n_bins)
        self.bin_width = 2.0 / n_bins
        self.env = ContinuousRecyclingEnv(gamma=gamma)
        self.actions = ['search', 'wait', 'recharge']
        self._precompute_transitions()
        

    def _state_to_index(self, b):
        """Map continuous b in [0,2] to nearest bin index"""
        idx = int(np.round(b / 2.0 * (self.n_bins - 1)))
        return int(np.clip(idx, 0, self.n_bins - 1))
    
    def _precompute_transitions(self):
        """Precompute next state indices for all bins"""
        nA = len(self.actions)
        self.next_idx = np.zeros((self.n_bins, nA), dtype=int)
        self.exp_r = np.zeros((self.n_bins, nA), dtype=float)

        for i, b in enumerate(self.bins):
            for a_i, a in enumerate(self.actions):
                b_next, _rescued = self.env.get_next_state(b, a)
                self.next_idx[i, a_i] = self._state_to_index(b_next)
                self.exp_r[i, a_i] = self.env.get_expected_reward(b, a)
    
    def value_iteration(self, theta=1e-4, max_iter=1000):
        """Run tabular value iteration on discretized MDP"""
        V = np.zeros(self.n_bins, dtype=float)
        delta_history = []

        for it in range(1, max_iter + 1):
            V_new = np.empty_like(V)

            for i in range(self.n_bins):
                q_vals = self.exp_r[i, :] + self.gamma * V[self.next_idx[i, :]]
                V_new[i] = np.max(q_vals)

            delta = float(np.max(np.abs(V_new - V)))
            delta_history.append(delta)
            V = V_new

            if delta < theta:
                break

        policy = self.extract_policy(V)
        return V, policy, it, delta_history
    
    def extract_policy(self, V):
        """Extract greedy policy from value function"""
        policy_idx = np.zeros(self.n_bins, dtype=int)
        for i in range(self.n_bins):
            q_vals = self.exp_r[i, :] + self.gamma * V[self.next_idx[i, :]]
            policy_idx[i] = int(np.argmax(q_vals))
        return policy_idx


# ============================================
# PART C: TILE CODING FUNCTION APPROXIMATION
# ============================================

class TileCoder1D:
    """
    Tile coding for 1D continuous state space [0, 2]
    
    Architecture:
    - n_tilings: number of overlapping tilings
    - n_tiles: number of tiles per tiling
    - Exactly n_tilings active features per state
    """
    
    def __init__(self, n_tilings=8, n_tiles=10, state_range=(0, 2)):
        """Initialize tile coder with offsets"""
        self.n_tilings = int(n_tilings)
        self.n_tiles = int(n_tiles)
        self.low = float(state_range[0])
        self.high = float(state_range[1])
        self.range = self.high - self.low
        self.tile_width = self.range / self.n_tiles  # 2.0/10 = 0.2
        self.offsets = np.linspace(0.0, self.tile_width, self.n_tilings, endpoint=False)
        self.feature_dim = self.n_tilings * self.n_tiles

    def _tile_index(self, state, tiling_id):
        s = float(np.clip(state, self.low, self.high))
        shifted = (s - self.low) + self.offsets[tiling_id]
        idx = int(np.floor(shifted / self.tile_width))
        return int(np.clip(idx, 0, self.n_tiles - 1))
    
    def get_features(self, state):
        """Convert continuous state to binary feature vector"""
        x = np.zeros(self.feature_dim, dtype=float)

        for t in range(self.n_tilings):
            tile = self._tile_index(state, t)
            feat_idx = t * self.n_tiles + tile
            x[feat_idx] = 1.0

        return x
    
    def get_feature_matrix(self, states):
        """Convert array of states to feature matrix X"""
        states = np.asarray(states, dtype=float)
        X = np.zeros((len(states), self.feature_dim), dtype=float)
        for i, s in enumerate(states):
            X[i, :] = self.get_features(float(s))
        return X


class LinearValueFunction:
    """
    Linear function approximation: V(b) = w^T · x(b)
    Uses batch least squares update
    """
    
    def __init__(self, tile_coder):
        self.tc = tile_coder
        self.w = np.zeros(tile_coder.feature_dim, dtype=float)
    
    def predict(self, state):
        """V(b) = w·x(b)"""
        x = self.tc.get_features(state)
        return float(self.w @ x)
    
    def predict_batch(self, states):
        """Vectorized prediction for many states"""
        X = self.tc.get_feature_matrix(states)
        return X @ self.w
    
    def batch_update(self, states, targets):
        X = self.tc.get_feature_matrix(states)
        y = np.asarray(targets, dtype=float)

        XtX = X.T @ X
        XtY = X.T @ y

        # Use pseudo-inverse to handle singular matrices
        self.w = np.linalg.pinv(XtX) @ XtY


class FAValueIteration:
    """
    Value iteration with linear function approximation
    Batch least squares at each iteration
    """
    
    def __init__(self, env, tile_coder):
        self.env = env
        self.tc = tile_coder
        self.vf = LinearValueFunction(tile_coder)
        self.gamma = env.gamma
        self.actions = ['search', 'wait', 'recharge']
    
    def compute_targets(self, states):
        """
        For each state b, compute:
        y(b) = max_a [ r(b,a) + γ * V(b') ]
        """
        states = np.asarray(states, dtype=float)
        targets = np.zeros_like(states, dtype=float)

        for i, b in enumerate(states):
            best_q = -np.inf
            for a in self.actions:
                r = self.env.get_expected_reward(float(b), a)
                b_next, _rescued = self.env.get_next_state(float(b), a)
                q = r + self.gamma * self.vf.predict(float(b_next))
                if q > best_q:
                    best_q = q
            targets[i] = best_q

        return targets
    
    def iterate(self, n_iterations=100, n_samples=500, theta=1e-4, seed=0):
        """
        Run value iteration with function approximation
        Returns:
            V_history: list of value functions at each iteration
            w_history: list of weight vectors
            delta_history: max change in V
            time_history: time per iteration
            sample_states: states used for evaluation
        """
        rng = np.random.default_rng(seed)

        # evaluation grid for tracking V curves and ΔV
        eval_states = np.linspace(0.0, 2.0, 300)
        V_prev = self.vf.predict_batch(eval_states)

        V_history = []
        w_history = []
        delta_history = []
        time_history = []

        for k in range(1, n_iterations + 1):
            t0 = time()

            # sample states uniformly from [0,2]
            sample_states = rng.uniform(0.0, 2.0, size=n_samples)

            # Bellman targets using current vf
            targets = self.compute_targets(sample_states)

            # batch least squares fit
            self.vf.batch_update(sample_states, targets)

            # evaluate on grid
            V_now = self.vf.predict_batch(eval_states)

            # max change across grid
            delta = float(np.max(np.abs(V_now - V_prev)))

            V_history.append(V_now.copy())
            w_history.append(self.vf.w.copy())
            delta_history.append(delta)
            time_history.append(time() - t0)

            V_prev = V_now

            # convergence check
            if delta < theta:
                break

        return V_history, w_history, delta_history, time_history, eval_states
    
def evaluate_policy(env, policy_fn, n_episodes=1000, seed=0):
    rng = np.random.default_rng(seed)
    returns = []
    rescues = []
    cans = []

    for ep in range(n_episodes):
        env.reset()
        done = False
        while not done:
            b = env.battery
            a = policy_fn(float(b))
            _, _, done, _ = env.step(a)

        returns.append(env.total_reward)
        rescues.append(1 if env.rescues > 0 else 0)
        cans.append(env.cans_collected)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "rescue_rate": float(np.mean(rescues)),
        "mean_cans": float(np.mean(cans)),
    }
    

# ============================================
# MAIN EXPERIMENT
# ============================================

def run_experiment():
    """Run complete experiment"""
    # part A
    print("=" * 60)
    print("PART A: 2-State Tabular MDP")
    print("=" * 60)
    
    robot = TabularRecyclingRobot()
    V_A, pi_A, it_A, deltas_A = robot.value_iteration()

    print("\nOptimal Values:")
    print(f"V*(high) = {V_A['high']:.6f}")
    print(f"V*(low)  = {V_A['low']:.6f}")
    print("\nOptimal Policy:")
    print(pi_A)
    print(f"Iterations to converge: {it_A}")

    # convergence plot 
    plt.figure()
    plt.plot(range(1, len(deltas_A) + 1), deltas_A)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Max ΔV (log scale)")
    plt.title("Tabular Value Iteration Convergence - 2-State MDP")
    plt.grid(True)
    plt.tight_layout()

    # Part A final results table
    print("\n[Table] Part A Final Results")
    print(f"{'State':<8} {'V*(state)':<12} {'pi*(state)':<12} {'Iterations':<10}")
    print(f"{'high':<8} {V_A['high']:<12.6f} {pi_A['high']:<12} {it_A:<10}")
    print(f"{'low':<8} {V_A['low']:<12.6f} {pi_A['low']:<12} {'':<10}")

    # part B
    
    print("\n" + "=" * 60)
    print("PART B: Continuous MDP - Discretized Baseline")
    print("=" * 60)
    
    solver = DiscretizedTabularSolver(n_bins=1000)
    V_true, pi_true, it_B, deltas_B = solver.value_iteration()

    print(f"Discretized iterations: {it_B}")
    print(f"Value range: {V_true.min():.6f} → {V_true.max():.6f}")
    
    # part C
    
    print("\n" + "=" * 60)
    print("PART C: Function Approximation with Tile Coding")
    print("=" * 60)
    
    env = ContinuousRecyclingEnv()
    tc = TileCoder1D()
    fa = FAValueIteration(env, tc)

    V_hist, w_hist, deltas_fa, times_fa, eval_states = fa.iterate(
        n_iterations=100,
        n_samples=500,
        theta=1e-4
    )

    print(f"FA iterations: {len(deltas_fa)}")
    print(f"Final ΔV: {deltas_fa[-1]:.6e}")
    print(f"Mean time/iteration: {np.mean(times_fa):.4f}s")
    print(f"Total time: {np.sum(times_fa):.4f}s")

    # Evaluation

    print("\n" + "=" * 60)
    print("EVALUATION AND VISUALIZATION")
    print("=" * 60)

    # Approximation Quality
    V_true_interp = np.interp(eval_states, solver.bins, V_true)
    V_fa_final = V_hist[-1]

    mse = float(np.mean((V_fa_final - V_true_interp) ** 2))
    max_abs_err = float(np.max(np.abs(V_fa_final - V_true_interp)))

    print("\nApproximation Quality:")
    print(f"MSE: {mse:.12e}")
    print(f"Max Absolute Error: {max_abs_err:.12e}")

    mse_history = [float(np.mean((V_k - V_true_interp) ** 2)) for V_k in V_hist]

    # Policy functions
    action_names = ['search', 'wait', 'recharge']
    action_to_int = {a: i for i, a in enumerate(action_names)}

    def policy_true_fn(b):
        idx = int(np.argmin(np.abs(solver.bins - b)))
        return action_names[int(pi_true[idx])]

    def policy_fa_fn(b):
        best_a, best_q = None, -1e18
        for a in action_names:
            r = env.get_expected_reward(b, a)
            b_next, _ = env.get_next_state(b, a)
            q = r + env.gamma * fa.vf.predict(b_next)
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    # Policy agreement
    rng = np.random.default_rng(1)
    test_states = rng.uniform(0.0, 2.0, size=1000)
    agreement = float(np.mean([policy_true_fn(b) == policy_fa_fn(b) for b in test_states]))
    print(f"Policy Agreement (1000 samples): {agreement:.4f}")

    # Policy evaluation 
    true_metrics = evaluate_policy(env, policy_true_fn, n_episodes=1000, seed=0)
    fa_metrics   = evaluate_policy(env, policy_fa_fn,   n_episodes=1000, seed=1)

    print("\nPolicy Quality (1000 episodes):")
    print(f"{'Policy':<12} {'MeanReturn':>12} {'Std':>10} {'RescueRate':>12} {'MeanCans':>10}")
    print(f"{'GroundTruth':<12} {true_metrics['mean_return']:>12.3f} {true_metrics['std_return']:>10.3f} {true_metrics['rescue_rate']:>12.3f} {true_metrics['mean_cans']:>10.3f}")
    print(f"{'FA':<12} {fa_metrics['mean_return']:>12.3f} {fa_metrics['std_return']:>10.3f} {fa_metrics['rescue_rate']:>12.3f} {fa_metrics['mean_cans']:>10.3f}")

    # Plot 1: Value Function Comparison
    
    plt.figure()
    plt.plot(solver.bins, V_true, label="Ground truth (discretized, N=1000)")
    for k in [1, 10, 25, 50, 100]:
        if k <= len(V_hist):
            plt.plot(eval_states, V_hist[k-1], label=f"FA iter {k}")
    plt.xlabel("Battery level b")
    plt.ylabel("Value V(b)")
    plt.title("Value Function: Tabular vs Function Approximation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Plot 2: Convergence Analysis

    plt.figure()
    plt.plot(range(1, len(deltas_B)+1), deltas_B, label="Max ΔV (tabular discretized)")
    plt.plot(range(1, len(deltas_fa)+1), deltas_fa, label="Max ΔV (FA)")
    plt.plot(range(1, len(mse_history)+1), mse_history, label="MSE vs ground truth (FA)")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Metric (log scale)")
    plt.title("Convergence Speed and Approximation Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Plot 3: Policy Visualization
    
    fa_policy_int = np.array([action_to_int[policy_fa_fn(float(b))] for b in solver.bins], dtype=int)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(solver.bins, pi_true)
    plt.yticks([0, 1, 2], action_names)
    plt.xlabel("b")
    plt.ylabel("Action")
    plt.title("Ground Truth Policy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(solver.bins, fa_policy_int)
    plt.yticks([0, 1, 2], action_names)
    plt.xlabel("b")
    plt.ylabel("Action")
    plt.title("FA Policy")
    plt.grid(True)

    plt.suptitle("Optimal Policy Comparison")
    plt.tight_layout()

    # Plot 4: Computational Efficiency
    
    t0 = time()
    _ = solver.value_iteration()
    tab_total = time() - t0
    tab_mean = tab_total / max(1, it_B)

    fa_mean = float(np.mean(times_fa))
    fa_total = float(np.sum(times_fa))

    plt.figure()
    plt.bar(["Discretized mean/iter", "FA mean/iter"], [tab_mean, fa_mean])
    plt.ylabel("Seconds")
    plt.title("Computational Cost Comparison - Time per Iteration")
    plt.grid(True, axis="y")
    plt.tight_layout()

    plt.figure()
    plt.bar(["Discretized total", "FA total"], [tab_total, fa_total])
    plt.ylabel("Seconds")
    plt.title("Computational Cost Comparison - Total Time")
    plt.grid(True, axis="y")
    plt.tight_layout()

    plt.show()

    print("\nExperiment complete!")

run_experiment()

"""
BRIEF COMMENTARY
=======================================

Convergence Speed:
The tabular value iteration converges smoothly with exponential decay in ΔV,
requiring 137 iterations in the 2-state MDP and 98 iterations in the
discretized continuous case. The function approximation (FA) method also
converges in 98 iterations with stable reduction in ΔV. Although FA updates
are based on sampled states and least-squares fitting, its convergence trend
closely follows the tabular solution.

Approximation Quality:
The FA value function matches the discretized ground truth extremely closely,
with near-zero MSE (~2e-24) and maximum absolute error around 1e-12.
Policy agreement over 1000 sampled states is 100%, and episode-based
performance metrics (mean return, rescue rate, cans collected) are nearly
identical. This indicates that tile coding with linear approximation
accurately represents the value structure of the environment.

Computational Trade-offs:
The discretized tabular method is computationally faster per iteration and
in total runtime. In contrast, FA incurs higher cost due to repeated batch
least-squares updates. This illustrates the trade-off between efficiency and
generalisation in reinforcement learning.
"""
    
    