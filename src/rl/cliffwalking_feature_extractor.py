import numpy as np
from feature_extractor import FeatureExtractor

class Actions:
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class CliffWalkingFeatureExtractor(FeatureExtractor):
    __actions_one_hot_encoding = {
        Actions.UP:    np.array([1, 0, 0, 0]),
        Actions.RIGHT: np.array([0, 1, 0, 0]),
        Actions.DOWN:  np.array([0, 0, 1, 0]),
        Actions.LEFT:  np.array([0, 0, 0, 1])
    }

    def __init__(self, env):
        '''
        Initializes the CliffWalkingFeatureExtractor object.
        It adds feature extraction methods to the features_list attribute.
        '''
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)
        self.features_list.append(self.f1)
        self.features_list.append(self.f2)

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
        In CliffWalking, the terminal state is 47 (location [3, 11]).
        '''
        return state == 47

    def get_actions(self):
        '''
        Returns a list of available actions in the environment.
        '''
        return [Actions.UP, Actions.RIGHT, Actions.DOWN, Actions.LEFT]
    
    def get_features(self, state, action):
        '''
        Takes a state and an action as input and returns the feature vector for that state-action pair.
        It calls the feature extraction methods and constructs the feature vector.
        '''
        feature_vector = np.zeros(len(self.features_list))
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)

        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])

        return feature_vector

    def f0(self, state, action):
        '''
        This is just the bias term.
        '''
        return 1.0

    def f1(self, state, action):
        '''
        Distance to goal feature.
        '''
        goal_state = 47
        coord_state = self.state_to_position(state)
        coord_goal = self.state_to_position(goal_state)
        next_coord = coord_state
        if action == Actions.UP:
            next_coord = (coord_state[0] - 1, coord_state[1])
        elif action == Actions.RIGHT:
            next_coord = (coord_state[0], coord_state[1] + 1)
        elif action == Actions.DOWN:
            next_coord = (coord_state[0] + 1, coord_state[1])
        elif action == Actions.LEFT:
            next_coord = (coord_state[0], coord_state[1] - 1)
        distance = self.manhattan_distance(next_coord, coord_goal)
        return 1.0 / (distance + 1)  # Adding 1 to avoid division by zero
    
    def f2(self, state, action):
        '''
        Proximity to cliff feature.
        '''
        cliff_states = range(37, 47)
        coord_state = self.state_to_position(state)
        next_coord = coord_state
        if action == Actions.UP:
            next_coord = (coord_state[0] - 1, coord_state[1])
        elif action == Actions.RIGHT:
            next_coord = (coord_state[0], coord_state[1] + 1)
        elif action == Actions.DOWN:
            next_coord = (coord_state[0] + 1, coord_state[1])
        elif action == Actions.LEFT:
            next_coord = (coord_state[0], coord_state[1] - 1)
        for cliff_state in cliff_states:
            coord_cliff = self.state_to_position(cliff_state)
            if next_coord == coord_cliff:
                return -1.0
        return 0.0

    @staticmethod
    def state_to_position(state):
        '''
        Converts a state number to a (row, col) position.
        '''
        row = state // 12
        col = state % 12
        return (row, col)

    @staticmethod
    def position_to_state(position):
        '''
        Converts a (row, col) position to a state number.
        '''
        return position[0] * 12 + position[1]
    
    @staticmethod
    def state_to_position(state):
        '''
        Converts a state number to a (row, col) position.
        '''
        row = state // 12
        col = state % 12
        return (row, col)
    
    @staticmethod
    def manhattan_distance(xy1, xy2):
        '''
        Computes the Manhattan distance between two points.
        '''
        return abs(xy1[0] - xy2[0]) + abs(xy1[1]-xy2[1])