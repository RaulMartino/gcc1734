from environment import Environment

class CliffWalkingEnvironment(Environment):
    def __init__(self, env):
        super().__init__(env)
        self.grid_height = 4
        self.grid_width = 12

    def get_num_states(self):
        return self.grid_height * self.grid_width

    def get_num_actions(self):
        return self.env.action_space.n  # 0: up, 1: right, 2: down, 3: left

    def get_state_id(self, state):
        return state

    def get_random_action(self):
        return self.env.action_space.sample()