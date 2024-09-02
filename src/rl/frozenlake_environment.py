from environment import Environment

class FrozenLakeEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        self.grid_height = 4
        self.grid_width = 4

    def get_num_states(self):
        return self.grid_height * self.grid_width
    
    def get_num_actions(self):
        return self.env.action_space.n
    
    def get_state_id(self, state):
        return state
    
    def get_random_action(self):
        return self.env.action_space.sample()