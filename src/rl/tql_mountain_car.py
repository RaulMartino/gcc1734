from timeit import default_timer as timer
import numpy as np
import pickle
from environment import Environment

class QLearningAgentTabular:

  def __init__(self, 
               env: Environment, 
               decay_rate, 
               learning_rate, 
               gamma):
    self.env = env

    self.q_table = np.zeros((env.position_bins, env.velocity_bins, env.get_num_actions()))
    print(f"self.q_table.shape: {self.q_table.shape}")
    # self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
    self.epsilon = 1.0
    self.max_epsilon = 1.0
    self.min_epsilon = 0.01
    self.decay_rate = decay_rate
    self.learning_rate = learning_rate
    self.gamma = gamma # discount rate
    self.epsilons_ = []
    
  def choose_action(self, state_p, state_v, is_in_exploration_mode=True):
    exploration_tradeoff = np.random.uniform(0, 1)

    if is_in_exploration_mode and exploration_tradeoff < self.epsilon:
      # exploration
      action = np.random.randint(self.env.get_num_actions())    
    else:
      # exploitation (taking the biggest Q value for this state)
      action = np.argmax(self.q_table[state_p, state_v, :])
    
    return action

  def update(self, state_p, state_v, action, reward, new_state_p, new_state_v):
    '''
    Apply update rule Q(s,a):= Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    '''
    self.q_table[state_p, state_v, action] = self.q_table[state_p, state_v, action] + \
      self.learning_rate * (reward + self.gamma * \
        np.max(self.q_table[new_state_p, new_state_v, :]) - self.q_table[state_p, state_v, action])

  def train(self, num_episodes: int):
    rewards_per_episode = []

    start_time = timer()  # Record the start time

    for episode in range(num_episodes):
  
      terminated = False
      truncated = False

      state, _ = self.env.reset()
      state_p = np.digitize(state[0], self.env.position_space)
      state_v = np.digitize(state[1], self.env.velocity_space)
      # state_id = self.env.get_state_id(state)
      # state = state_id

      rewards_in_episode = []
      
      total_penalties = 0

      while not (terminated or truncated or total_penalties < -2000):
          
        # print(f"state: {state}")
        action = self.choose_action(state_p, state_v)
        
        # transição
        new_state, reward, terminated, truncated, info = self.env.step(action)
        new_state_p = np.digitize(new_state[0], self.env.position_space)
        new_state_v = np.digitize(new_state[1], self.env.velocity_space)

        if reward < 0:
            total_penalties += reward

        self.update(state_p, state_v, action, reward, new_state_p, new_state_v)

        if (terminated or truncated):
          # Reduce epsilon to decrease the exploration over time
          self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
            np.exp(-self.decay_rate * episode)
          self.epsilons_.append(self.epsilon)

        state = new_state
        state_p = new_state_p
        state_v = new_state_v
            
        rewards_in_episode.append(reward)

      sum_rewards = np.sum(rewards_in_episode)
      rewards_per_episode.append(sum_rewards)

      if episode % 100 == 0:
        end_time = timer()  # Record the end time
        execution_time = end_time - start_time
        n_actions = len(rewards_in_episode)
        print(f"Stats for episode {episode}/{num_episodes}:") 
        print(f"\tNumber of actions: {n_actions}")
        print(f"\tTotal reward: {sum_rewards:#.2f}")
        print(f"\tExecution time: {execution_time:.2f}s")
        print(f"\tTotal penalties: {total_penalties}")
        start_time = end_time

    return rewards_per_episode

  def save(self, filename):
    # open a file, where you want to store the data
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(self, file)

    # close the file
    file.close()

  @staticmethod
  def load_agent(filename):
    # open a file, where you stored the pickled data
    file = open(filename, 'rb')

    # dump information to that file
    agent = pickle.load(file)

    return agent