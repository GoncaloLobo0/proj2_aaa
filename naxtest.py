import numpy as np
import gymnax
import matplotlib.pyplot as plt
import jax

rng = jax.random.PRNGKey(12)

env, env_params = gymnax.make("MountainCar-v0")
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

n_position = 20  # Number of discrete positions
n_velocity = 20  # Number of discrete velocities
n_actions = env.action_space().n

def discretize_state(state, position_bins, velocity_bins):
    position, velocity = state
    position_idx = np.digitize(position, position_bins) - 1
    velocity_idx = np.digitize(velocity, velocity_bins) - 1
    return position_idx, velocity_idx

position_bins = np.linspace(env.observation_space(env_params).low[0], env.observation_space(env_params).high[0], n_position)
velocity_bins = np.linspace(env.observation_space(env_params).low[1], env.observation_space(env_params).high[1], n_velocity)

Q_table = np.zeros((n_position, n_velocity, n_actions))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial epsilon for epsilon-greedy
epsilon_decay = 0.9995  # Decay factor for epsilon
min_epsilon = 0.01  # Minimum epsilon
episodes = 50000  # Number of training episodes

# Tracking performance
rewards_per_episode = []
steps_per_episode = []

# Q-learning algorithm
for episode in range(episodes):
    obs, state = env.reset(key_reset, env_params)
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        rng, key_reset, key_act, key_step = jax.random.split(rng, 4)
        position_idx, velocity_idx = discretize_state(obs, position_bins, velocity_bins)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.choice(n_actions)  # Explore
        else:
            action = np.argmax(Q_table[position_idx, velocity_idx])  # Exploit
            
        next_obs, next_state, reward, done, _ = env.step(key_step, state, action, env_params)
        next_position_idx, next_velocity_idx = discretize_state(next_obs, position_bins, velocity_bins)
        
        # Q-learning update
        best_next_action = np.argmax(Q_table[next_position_idx, next_velocity_idx])
        td_target = reward + gamma * Q_table[next_position_idx, next_velocity_idx, best_next_action]
        td_error = td_target - Q_table[position_idx, velocity_idx, action]
        Q_table[position_idx, velocity_idx, action] += alpha * td_error
        
        state = next_state
        obs = next_obs
        total_reward += reward
        steps += 1
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}, Epsilon = {epsilon:.3f}")
        
        

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.subplot(1, 2, 2)
plt.plot(steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')

plt.tight_layout()
plt.show()

env.close()