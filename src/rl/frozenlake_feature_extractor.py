import numpy as np
from feature_extractor import FeatureExtractor

class Actions:
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

class FrozenLakeFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.LEFT:  np.array([1, 0, 0, 0]),
        Actions.DOWN:  np.array([0, 1, 0, 0]),
        Actions.RIGHT: np.array([0, 0, 1, 0]),
        Actions.UP:    np.array([0, 0, 0, 1])
    }

    def __init__(self, env):
        '''
        Initializes the FrozenLakeFeatureExtractor object.
        '''
        self.env = env
        self.is_slippery = True
        self.nrow, self.ncol = env.nrow, env.ncol
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.f1)
        self.features_list.append(self.f2)
        self.features_list.append(self.f3)
        self.features_list.append(self.f4)

    def get_num_features(self):
        '''
        Returns the number of features extracted by the feature extractor.
        '''
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        '''
        Returns the number of actions available in the environment.
        '''
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        '''
        Returns the one-hot encoded representation of an action.
        '''
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        '''
        Checks if the given state is a terminal state.
        '''
        row, col = self.state_to_position(state)
        return self.env.desc[row][col] in b'GH'

    def get_actions(self):
        '''
        Returns a list of available actions in the environment.
        '''
        return [Actions.LEFT, Actions.DOWN, Actions.RIGHT, Actions.UP]
    
    def get_features(self, state, action):
        '''
        Takes a state and an action as input and returns the feature vector for that state-action pair.
        '''
        feature_vector = np.zeros(len(self.features_list))
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)

        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])

        return feature_vector

    def f0(self, state, action):
        '''
        Bias term.
        '''
        return 1.0

    def f1(self, state, action):
        '''
        Distance to goal feature, considering the action.
        Returns a higher value when closer to the goal.
        '''
        next_state = self.get_next_state(state, action)
        goal_state = self.nrow * self.ncol - 1
        row_goal, col_goal = self.state_to_position(goal_state)
        row_next, col_next = self.state_to_position(next_state)
        distance = abs(row_goal - row_next) + abs(col_goal - col_next)
        return 1.0 / (distance + 1)  # Adding 1 to avoid division by zero

    def f2(self, state, action):
        '''
        Safety feature (avoiding holes), considering the action.
        Returns a lower value when the next state is likely to be a hole.
        '''
        next_state = self.get_next_state(state, action)
        row, col = self.state_to_position(next_state)
        if self.env.desc[row][col] == b'H':
            return 0.0  # Strongly discourage moving into a hole
        return 1.0  # Encourage safe moves

    def f3(self, state, action):
        '''
        are the next states closer to the goal?
        '''
        possible_next_states = self.get_possible_next_states(state, action)

        sum_distance = 0

        goal_state = self.nrow * self.ncol - 1
        
        row_goal, col_goal = self.state_to_position(goal_state)

        for next_state in possible_next_states:
            row_next, col_next = next_state
            distance = abs(row_goal - row_next) + abs(col_goal - col_next)
            sum_distance += distance
        
        mean_distance = sum_distance / len(possible_next_states)

        return 1.0 / (mean_distance + 1)  # Adding 1 to avoid division by zero

    def f4(self, state, action):
        '''
        are the next states safe?
        '''
        possible_next_states = self.get_possible_next_states(state, action)
        row, col = self.state_to_position(state)
        num_holes = 0
        for next_state in possible_next_states:
            next_row, next_col = next_state
            if self.env.desc[next_row][next_col] == b'H':
                num_holes += 1
        
        return 1.0 / (num_holes + 1)  # Adding 1 to avoid division by zero

    def get_next_state(self, state, action):
        '''
        Calculates the most likely next state given the current state and action.
        '''
        row, col = self.state_to_position(state)
        if action == Actions.LEFT:
            col = max(col - 1, 0)
        elif action == Actions.RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif action == Actions.UP:
            row = max(row - 1, 0)
        elif action == Actions.DOWN:
            row = min(row + 1, self.nrow - 1)
        return self.position_to_state((row, col))
    
    def get_possible_next_states(self, state, action):
        '''
        Returns a list of possible next states given the current state.
        '''
        row, col = self.state_to_position(state)
        if action == Actions.LEFT:
            next_states = [(row, max(col - 1, 0)), (max(row-1,0), col), (min(row+1,self.nrow-1), col)]
        elif action == Actions.RIGHT:
            next_states = [(row, min(col + 1, self.ncol - 1)), (max(row-1,0), col), (min(row+1,self.nrow-1), col)]
        elif action == Actions.UP:
            next_states = [(max(row - 1, 0), col), (row, max(col-1,0)), (row, min(col+1,self.ncol-1))]
        elif action == Actions.DOWN:
            next_states = [(min(row + 1, self.nrow - 1), col), (row, max(col-1,0)), (row, min(col+1,self.ncol-1))]
        return next_states

    def state_to_position(self, state):
        '''
        Converts a state number to a (row, col) position.
        '''
        row = state // self.ncol
        col = state % self.ncol
        return (row, col)

    def position_to_state(self, position):
        '''
        Converts a (row, col) position to a state number.
        '''
        return position[0] * self.ncol + position[1]