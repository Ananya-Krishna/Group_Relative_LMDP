import numpy as np
import matplotlib.pyplot as plt
from agent import lsvi_ucb
from linearmdp import linMDP

# Define a custom GridWorldMDP
class GridWorldMDP(linMDP):
    def __init__(self, grid_size, horizon, feature_type="standard"):
        self.grid_size = grid_size
        self.S = grid_size * grid_size  # number of states
        self.A = 4  # up, down, left, right
        self.H = horizon
        self.feature_type = feature_type

        if feature_type == "standard":
            self.d = self.S * self.A
        elif feature_type == "group":
            self.d = self.S  # share across actions
        else:
            raise ValueError("Unknown feature type")

        # dummy to satisfy superclass
        self.num_chain = 1
        super(linMDP, self).__init__()
        
        self.dim = self.d 
        self.horizon = self.H
        self.num_state = self.S
        self.num_action = self.A

        self.generate_linearMDP()

    def generate_linearMDP(self):
        """
        Generate a structured GridWorld Linear MDP,
        matching the assumptions in Group Relative Linear MDPs,
        with small step rewards and goal reward.
        """

        # === 1. Feature map φ(s,a)
        self.phi = np.zeros((self.H, self.S, self.A, self.d))

        if self.feature_type == "standard":
            # Standard features: one-hot over (state, action)
            for h in range(self.H):
                for s in range(self.S):
                    for a in range(self.A):
                        idx = s * self.A + a  # unique id for (s,a)
                        if idx < self.d:  # Safety: avoid overflow if d < S*A
                            self.phi[h, s, a, idx] = 1.0
            print("Generated standard one-hot phi features for GridWorldMDP.")

        elif self.feature_type == "group":
            # Group-invariant features: depend only on state
            for h in range(self.H):
                for s in range(self.S):
                    for a in range(self.A):
                        idx = s  # depend only on state
                        if idx < self.d:
                            self.phi[h, s, a, idx] = 1.0
            print("Generated group-invariant phi features for GridWorldMDP.")

        else:
            raise ValueError(f"Unknown feature_type {self.feature_type}")

        # === 2. True reward parameters θ
        self.theta = np.zeros((self.H, self.d))

        goal_state = self.S - 1  # e.g., bottom-right corner is goal

        for h in range(self.H):
            for i in range(self.d):
                if i == goal_state % self.d:
                    self.theta[h, i] = 1.0  # Big reward for goal-related features
                else:
                    self.theta[h, i] = 0.0  # Zero elsewhere
                    
        # === 3. True transition parameters μ
        self.mu = np.zeros((self.H, self.d, self.S))

        grid_len = int(np.sqrt(self.S))  # Grid dimension (assume square)
        
        for h in range(self.H):
            for i in range(self.d):
                s = i % self.S  # corresponding state
                possible_moves = []

                # Define moves: up, down, left, right
                if s >= grid_len:
                    possible_moves.append(s - grid_len)  # up
                if s < self.S - grid_len:
                    possible_moves.append(s + grid_len)  # down
                if s % grid_len != 0:
                    possible_moves.append(s - 1)  # left
                if (s + 1) % grid_len != 0:
                    possible_moves.append(s + 1)  # right

                # If no move possible (corner case), stay
                if not possible_moves:
                    possible_moves = [s]

                for p in possible_moves:
                    self.mu[h, i, p] = 1.0 / len(possible_moves)

                # Small noise everywhere else
                for p in range(self.S):
                    if p not in possible_moves:
                        self.mu[h, i, p] = 1e-4

                self.mu[h, i] /= self.mu[h, i].sum()  # Normalize

        # === 4. Build true transition and reward functions
        self.tran_func = np.einsum('hsad,hdp->hsap', self.phi, self.mu)
        self.tran_func = np.clip(self.tran_func, 0, None)
        row_sums = self.tran_func.sum(axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.tran_func /= row_sums

        self.reward_func = np.zeros((self.H, self.S, self.A))

        for h in range(self.H):
            for s in range(self.S):
                for a in range(self.A):
                    next_state_probs = self.tran_func[h, s, a, :]
                    expected_reward = 0.0
                    for s_prime, prob in enumerate(next_state_probs):
                        dist_now = grid_distance(s, goal_state, grid_len)
                        dist_next = grid_distance(s_prime, goal_state, grid_len)

                        # Reward if closer to goal
                        if dist_next < dist_now:
                            expected_reward += prob * 0.01  # small positive reward
                        # Reward big if reaching goal
                        if s_prime == goal_state:
                            expected_reward += prob * 1.0  # goal reward

                    self.reward_func[h, s, a] = expected_reward

        print("Finished building structured GridWorld Linear MDP.")

def grid_distance(s1, s2, grid_len):
    """
    Manhattan distance between two flattened grid states
    """
    x1, y1 = divmod(s1, grid_len)
    x2, y2 = divmod(s2, grid_len)
    return abs(x1 - x2) + abs(y1 - y2)

def create_mdp(grid_size, horizon, feature_type="standard"):
    return GridWorldMDP(grid_size, horizon, feature_type)

def run_gridworld_experiment(grid_size=4, horizon=10, num_epi=500):
    lam = 1.0
    p = 0.1
    c = 5.0
    set_beta = 0

    print(f"Running Gridworld Grid {grid_size}x{grid_size} with standard and group features")

    # Standard features
    print(f"Creating standard MDP for {grid_size}x{grid_size} grid...")
    mdp_standard = create_mdp(grid_size, horizon, feature_type="standard")
    print("Running LSVI-UCB with standard features...")
    accu_reward_standard, _, episode_accu_reward_standard = lsvi_ucb(
        mdp_standard, lam=lam, num_epi=num_epi, p=p, c=c, set_beta=set_beta
    )

    # Group features
    print(f"Creating group-invariant MDP for {grid_size}x{grid_size} grid...")
    mdp_group = create_mdp(grid_size, horizon, feature_type="group")
    print("Running LSVI-UCB with group-invariant features...")
    accu_reward_group, _, episode_accu_reward_group = lsvi_ucb(
        mdp_group, lam=lam, num_epi=num_epi, p=p, c=c, set_beta=set_beta
    )

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(episode_accu_reward_standard, label='Standard Features (dim = d)')
    plt.plot(episode_accu_reward_group, label='Group-Invariant Features (dim = d_G)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(f'Grid Size {grid_size}x{grid_size} | Horizon {horizon} | Episodes {num_epi}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'gridlock_standard_vs_group_{grid_size}x{grid_size}.png')
    plt.show()

    np.savez(f'gridlock_standard_vs_group_{grid_size}x{grid_size}.npz',
             standard=episode_accu_reward_standard,
             group=episode_accu_reward_group)

if __name__ == "__main__":
    run_gridworld_experiment(grid_size=4, horizon=10, num_epi=500)
