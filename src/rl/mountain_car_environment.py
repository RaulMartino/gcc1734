import numpy as np
from environment import Environment

class MountainCarEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        self.position_bins = 20
        self.velocity_bins = 20

        self.position_space = np.linspace(-1.2, 0.6, self.position_bins)
        self.velocity_space = np.linspace(-0.07, 0.07, self.velocity_bins)
    
    def get_num_states(self):
        return self.position_bins * self.velocity_bins
    
    def get_num_actions(self):
        return self.env.action_space.n
    
    def get_state_id(self, state):
        position, velocity = state
        position_bin = np.digitize(position, self.position_space)
        velocity_bin = np.digitize(velocity, self.velocity_space)
        return position_bin * self.velocity_bins + velocity_bin
    
    def get_random_action(self):
        return self.env.action_space.sample()