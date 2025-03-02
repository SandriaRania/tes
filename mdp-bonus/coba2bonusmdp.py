import itertools
import numpy as np

# Gridworld parameters
GRID_HEIGHT = 3
GRID_WIDTH = 4
NUM_STATES = 10  # 8 normal states + 1 terminal state (goal/hole)
ACTIONS = ['N', 'E', 'S', 'W']
NUM_ACTIONS = len(ACTIONS)
DISCOUNT_FACTOR = 0.9  # Gamma for discounted rewards
SLIP_PROB = 0.1  # Probability of slipping to a wrong direction
REWARD_GOAL = 1.0
REWARD_HOLE = -1.0
REWARD_STEP = -0.01

# State transition probabilities
def get_transition_probabilities():
    transitions = {}
    for s in range(NUM_STATES - 1):  # Exclude terminal state
        transitions[s] = {}
        for a, action in enumerate(ACTIONS):
            transitions[s][a] = []
            intended_s = move(s, action)
            for slip_a, slip_action in enumerate(ACTIONS):
                slip_s = move(s, slip_action)
                prob = SLIP_PROB / (NUM_ACTIONS - 1) if slip_a != a else 1 - SLIP_PROB
                transitions[s][a].append((prob, slip_s))
    return transitions

# Movement function
def move(state, action):
    if state in [8, 9]:  # Goal and Hole -> Terminal State
        return 8  # Terminal state
    row, col = divmod(state, GRID_WIDTH)
    
    if action == 'N' and row > 0:
        row -= 1
    elif action == 'S' and row < GRID_HEIGHT - 1:
        row += 1
    elif action == 'E' and col < GRID_WIDTH - 1:
        col += 1
    elif action == 'W' and col > 0:
        col -= 1

    new_state = row * GRID_WIDTH + col
    if new_state >= NUM_STATES:  # Ensure new_state is within range
        return 8  # Terminal state
    return new_state


# Compute expected value of a policy
def evaluate_policy(policy, transitions):
    V = np.zeros(NUM_STATES)
    threshold = 1e-6
    while True:
        delta = 0
        for s in range(NUM_STATES - 1):  # Exclude terminal
            a = policy[s]
            new_value = sum(prob * (REWARD_STEP + DISCOUNT_FACTOR * V[next_s]) for prob, next_s in transitions[s][a])
            delta = max(delta, abs(V[s] - new_value))
            V[s] = new_value
        if delta < threshold:
            break
    return V

# Compute the average reward of a policy
def average_reward(policy, transitions):
    R = np.zeros(NUM_STATES - 1)
    P = np.zeros((NUM_STATES - 1, NUM_STATES - 1))
    for s in range(NUM_STATES - 1):
        a = policy[s]
        for prob, next_s in transitions[s][a]:
            P[s, next_s] += prob
            R[s] += prob * REWARD_STEP
    
    try:
        avg_r = np.linalg.solve(np.eye(NUM_STATES - 1) - P + np.ones((NUM_STATES - 1, NUM_STATES - 1)), R)
        return avg_r.mean()
    except np.linalg.LinAlgError:
        return float('-inf')

# Exhaustive policy search
def find_optimal_policies():
    transitions = get_transition_probabilities()

# Debugging: Print if any state transitions to an invalid state
    for s, actions in transitions.items():
        for a, outcomes in actions.items():
            for prob, next_s in outcomes:
                if next_s >= NUM_STATES:
                    print(f"ERROR: State {s} with action {ACTIONS[a]} transitions to invalid state {next_s}")

    best_discounted_policy = None
    best_discounted_value = float('-inf')
    best_avg_policy = None
    best_avg_value = float('-inf')
    
    for policy_tuple in itertools.product(range(NUM_ACTIONS), repeat=NUM_STATES - 1):
        policy = list(policy_tuple)
        V = evaluate_policy(policy, transitions)
        avg_r = average_reward(policy, transitions)
        
        if np.sum(V) > best_discounted_value:
            best_discounted_value = np.sum(V)
            best_discounted_policy = policy[:]
        
        if avg_r > best_avg_value:
            best_avg_value = avg_r
            best_avg_policy = policy[:]
    
    return best_discounted_policy, best_avg_policy

# Run exhaustive search
best_discounted_policy, best_avg_policy = find_optimal_policies()
print("Best Discounted Policy:", best_discounted_policy)
print("Best Average Reward Policy:", best_avg_policy)
