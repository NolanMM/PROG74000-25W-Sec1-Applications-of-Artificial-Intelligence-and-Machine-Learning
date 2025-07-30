import pprint

def bellman_optimality(mdp, gamma=0.9, theta=1e-6):
    """
    Calculate the Bellman Optimality Equation corresponding to a Markov Decision Process (MDP).

    This function iteratively calculates the optimal value function V*(s) for each
    state and then extracts the optimal policy (state -> action).

    Args:
        mdp (dict): The Markov Decision Process. The structure is:
                    {
                        'state': {
                            'action': [(probability, next_state, reward), ...],
                            ...
                        },
                        ...
                    }
        gamma (float): The discount factor for future rewards.
        theta (float): A small threshold for checking convergence. The algorithm stops when the value function changes by less than theta.

    Returns:
        tuple: A tuple containing:
            - V (dict): The optimal value function {state: value}.
            - actions (dict): The optimal actions {state: action}.
    """
    # 1. Initialization - Get all states from the MDP definition
    states = mdp.keys()
    # Initialize the value function V(s) to 0 for all states
    V = {s: 0 for s in states}

    print("Starting Value Iteration...\n" + "-" * 30)

    iteration = 0
    while True:
        iteration += 1
        # Delta is used to track the maximum change in the value functionn in the current iteration. We use it to check for convergence.
        delta = 0
        
        V_temp = V.copy()

        # 2. Iteration over all states
        for s in states:
            # Store the old value of the state to calculate the change (delta)
            v_old = V[s]
            
            # Stores the expected values for taking each possible action from state 's'
            action_values = []

            # 3. Calculate Q-value for each action in the current state
            for a in mdp[s]:
                q_s_a = 0
                # The value of an action is the sum of (prob * (reward + gamma * V(s'))) for all possible outcomes of that action.
                for (prob, next_state, reward) in mdp[s][a]:
                    q_s_a += prob * (reward + gamma * V[next_state])
                action_values.append(q_s_a)

            # 4. Update the value function for the current state (The new value for the state is the maximum of the Q-values for all possible actions.
            if action_values:
                V_temp[s] = max(action_values)
            
            # Update delta with the maximum change in this iteration
            delta = max(delta, abs(v_old - V_temp[s]))
        
        # Update the main value function with the new values from this iteration
        V = V_temp

        print(f"Iteration {iteration}: Delta = {delta:.6f}")

        # 5. Check Convergence (If the maximum change (delta) is smaller than our threshold (theta), the value function has converged, and we can exit the loop.)
        if delta < theta:
            print("-" * 30 + f"\nConvergence reached after {iteration} iterations.")
            break
            
    # 6. Extracting Action (we can extract the optimal action from the optimal value function V* is found)
    policies = {s: None for s in states}
    for s in states:
        best_action = None
        max_action_value = -float('inf')

        # Find the action that maximizes the expected value from state 's'
        for a in mdp[s]:
            q_s_a = 0
            for (prob, next_state, reward) in mdp[s][a]:
                q_s_a += prob * (reward + gamma * V[next_state])
            
            if q_s_a > max_action_value:
                max_action_value = q_s_a
                best_action = a
        
        policies[s] = best_action

    return V, policies


if __name__ == '__main__':
    # MDP Definition with format: {state: {action: [(probability, next_state, reward), ...], ...}}
    mdp_definition = {
        's0': {
            'a0': [(0.6, 's1', 10), (0.4, 's0', 0)],
            'a1': [(1.0, 's1', 0)]
        },
        's1': {
            'a1': [(1.0, 's1', 0)],
            'a2': [(1.0, 's0', -10)],
            'a3': [(0.8, 's2', 0), (0.2, 's3', -30)]
        },
        's2': {
            'a0': [(0.7, 's3', 0), (0.3, 's1', -10)],
            'a2': [(0.8, 's1', 0), (0.2, 's2', 10)]
        },
        's3': {
            'a1': [(0.9, 's3', 20), (0.1, 's1', -10)]
        }
    }
    gamma = 0.9     # Discount factor
    theta = 1e-6    # Convergence threshold

    optimal_values, optimal_policy = bellman_optimality(mdp_definition, gamma, theta)

    print("\n" + "="*40 + "\nResults" + "\n" + "="*40)
    print("\nOptimal Value Function Results" + "\n" + "-"*30)
    pprint.pprint(optimal_values)

    print("\nOptimal Policies Results" + "\n" + "-"*30)
    pprint.pprint(optimal_policy)
    print("\n"+ "="*40 + "\nExplaination\n" + "="*40 + "\n")
    for state, action in optimal_policy.items():
        print(f"From state '{state}', the optimal action is '{action}' with the maximum values of reward it can reach is {optimal_values[state]:.2f}.")
    print("\n")