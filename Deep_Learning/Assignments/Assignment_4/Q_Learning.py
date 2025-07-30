import gymnasium as gym
import pandas as pd
import numpy as np
import random

# Parameter sets to test as requested
ALPHA_VALUES = [0.01, 0.001, 0.2]
INITIAL_EPSILON_VALUES = [0.2, 0.3]

# Default parameters (used when the other is being varied)
DEFAULT_ALPHA = 0.1
DEFAULT_INITIAL_EPSILON = 0.1
# Default discount factor: how important future rewards are
GAMMA = 0.9                     

EPSILON_DECAY = 0.995   # How quickly the agent stops exploring
MIN_EPSILON = 0.01      # Minimum exploration rate

# Training and testing configuration
NUM_EPISODES = 100000    # Number of games to play during training
MAX_STEPS = 20          # Max steps that taxi taken per episode before we considered a failure
NUM_TEST_EPISODES = 10  # Number of episodes to run for testing after training

def train_agent(env, alpha, initial_epsilon, gamma, epsilon_decay, min_epsilon, num_episodes, max_steps):
    """
    Trains the Q-learning agent for a given set of hyperparameters.
    
    Args:
        env: The Gym environment.
        alpha (float): The learning rate.
        initial_epsilon (float): The starting exploration rate.
        gamma (float): The discount factor.
        epsilon_decay (float): The rate at which epsilon decays.
        min_epsilon (float): The minimum value for epsilon.
        num_episodes (int): The number of episodes to train for.
        max_steps (int): The maximum steps per episode.
        
    Returns:
        numpy.ndarray: The trained Q-table.
    """
    print(f"\n" + "="*50)
    print(f" STARTING TRAINING: alpha={alpha}, epsilon={initial_epsilon} ")
    print("="*50)
    
    # Initialize a new Q-table for each training run
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    # Reset epsilon for each new training run
    epsilon = initial_epsilon  

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False

        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                # If the random number is less than epsilon, the agent can choose a random action
                action = env.action_space.sample()  
            else:
                # Otherwise, it choose the best known action from the learned Q-values
                action = np.argmax(q_table[state, :])

            # Perform the action and calculate the next state and reward
            next_state, reward, done, truncated, info = env.step(action)
            old_q_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            # Update the Q-value using the Q-learning formula
            new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_q_value

            state = next_state
            
            if done or truncated:
                break
        
        # Decay epsilon to reduce exploration over time
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 2000 == 0:
            print(f"  Running Episode {episode + 1}/{num_episodes} completed.")
            
    print("-" * 3 +" Training Finished " + "-" * 3)
    return q_table

def test_agent(q_table, num_episodes, max_steps, render=True):
    """
    Tests the performance of a trained agent.
    
    Args:
        q_table (numpy.ndarray): The trained Q-table.
        num_episodes (int): The number of episodes to test.
        max_steps (int): The maximum steps per episode.
        render (bool): If True, renders the environment for visualization.
        
    Returns:
        float: The average total reward over the test episodes.
    """
    # Create a separate environment for testing
    test_env = gym.make("Taxi-v3", render_mode="human" if render else None)
    
    total_rewards = []
    print("\n--- Testing Agent ---")
    
    for episode in range(num_episodes):
        state, _ = test_env.reset()
        done = False
        truncated = False
        episode_reward = 0

        for step in range(max_steps):
            if render:
                test_env.render()

            # During testing, always choose the best action
            action = np.argmax(q_table[state, :])
            next_state, reward, done, truncated, info = test_env.step(action)
            
            state = next_state
            episode_reward += reward

            if done or truncated:
                if render:
                    test_env.render()
                break
        
        total_rewards.append(episode_reward)
        print(f"  Test Episode {episode + 1}: Total Reward = {episode_reward}")

    test_env.close()
    
    average_reward = np.mean(total_rewards)
    print(f"--- Average Test Reward: {average_reward:.2f} ---")
    return average_reward

if __name__ == "__main__":
    # Create a single environment instance to be used for all training
    env = gym.make("Taxi-v3")
    
    results = []
    temporary_average_reward = None
    best_q_table = None

    # Experiment 1: Compare Multiple Learning Rates (alpha) Values
    for alpha_val in ALPHA_VALUES:
        q_table = train_agent(
            env=env,
            alpha=alpha_val, 
            initial_epsilon=DEFAULT_INITIAL_EPSILON, 
            gamma=GAMMA, 
            epsilon_decay=EPSILON_DECAY, 
            min_epsilon=MIN_EPSILON, 
            num_episodes=NUM_EPISODES, 
            max_steps=MAX_STEPS
        )
        # Test
        avg_reward = test_agent(q_table, NUM_TEST_EPISODES, MAX_STEPS, render=True)
        results.append({
            'alpha': alpha_val,
            'epsilon': DEFAULT_INITIAL_EPSILON,
            'Average Reward': avg_reward
        })
        if temporary_average_reward is None or avg_reward > temporary_average_reward:
            temporary_average_reward = avg_reward
            best_q_table = q_table

    # Experiment 2: Compare Multiple Exploration Factors (epsilon) Values
    for epsilon_val in INITIAL_EPSILON_VALUES:
        q_table = train_agent(
            env=env,
            alpha=DEFAULT_ALPHA, 
            initial_epsilon=epsilon_val, 
            gamma=GAMMA, 
            epsilon_decay=EPSILON_DECAY, 
            min_epsilon=MIN_EPSILON, 
            num_episodes=NUM_EPISODES, 
            max_steps=MAX_STEPS
        )
        # Test
        avg_reward = test_agent(q_table, NUM_TEST_EPISODES, MAX_STEPS, render=True)
        results.append({
            'alpha': alpha_val,
            'epsilon': DEFAULT_INITIAL_EPSILON,
            'Average Reward': avg_reward
        })
        if temporary_average_reward is None or avg_reward > temporary_average_reward:
            temporary_average_reward = avg_reward
            best_q_table = q_table

    # Final Summary 
    print("\n\n" + "="*50)
    print(" Experiment Summary ")
    print("="*50)
    for result in results:
        print(f"Parameters: {result['alpha'], result['epsilon']}, -> Average Reward: {result['Average Reward']}")

    if best_q_table is not None:
        print("\nThe best agent was trained with the following parameters:")
        best_params = max(results, key=lambda x: x['Average Reward'])
        print(f"Parameters: {best_params}")
    
    df = pd.DataFrame(results)
    df.to_csv("q_learning_metrics_results.csv", index=False)

    env.close()
