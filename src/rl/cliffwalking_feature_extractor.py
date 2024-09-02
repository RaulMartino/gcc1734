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
        """
        Initializes the CliffWalkingFeatureExtractor object.
        Adds feature extraction methods to the features_list attribute.
        """
        self.env = env
        self.features_list = []
        self.features_list.append(self.f0)  # Bias term
        self.features_list.append(self.f1)  # Penalidade por proximidade ao penhasco
        self.features_list.append(self.f2)  # Distância até a meta

    def get_num_features(self):
        """
        Returns the number of features extracted by the feature extractor.
        """
        return len(self.features_list) + self.get_num_actions()

    def get_num_actions(self):
        """
        Returns the number of actions available in the environment.
        """
        return len(self.get_actions())

    def get_action_one_hot_encoded(self, action):
        """
        Returns the one-hot encoded representation of an action.
        """
        return self.__actions_one_hot_encoding[action]

    def is_terminal_state(self, state):
        """
        Checks if the state is terminal.
        The terminal state is when the agent reaches the goal at [3, 11].
        """
        return state == 47

    def get_actions(self):
        """
        Returns a list of available actions in the environment.
        """
        return [Actions.UP, Actions.RIGHT, Actions.DOWN, Actions.LEFT]
    
    def get_features(self, state, action):
        """
        Takes a state and an action as input and returns the feature vector for that state-action pair.
        It calls the feature extraction methods and constructs the feature vector.
        """
        feature_vector = np.zeros(len(self.features_list))
        for index, feature in enumerate(self.features_list):
            feature_vector[index] = feature(state, action)

        action_vector = self.get_action_one_hot_encoded(action)
        feature_vector = np.concatenate([feature_vector, action_vector])

        return feature_vector

    def f0(self, state, action):
        """
        This is just the bias term.
        """
        return 1.0

    def f1(self, state, action):
        cliff_positions = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]  # Penhasco na linha 3, colunas 1 a 10
        penalty = 1.0 if state in cliff_positions else 0.0
        return -penalty  # Penalidade para estados no penhasco

    def f2(self, state, action):
        goal_position = (3, 11)
        agent_position = (state // 12, state % 12)
        distance_to_goal = np.linalg.norm(np.array(goal_position) - np.array(agent_position))
        max_distance = np.linalg.norm(np.array([0, 0]) - np.array(goal_position))
        return 1.0 - (distance_to_goal / max_distance)
