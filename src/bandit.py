import random
import numpy as np

class Bandit:
    """A multi-armed bandit class representing a set of arms 
    with different reward distributions.

    Attributes:
    - k (int): The number of arms 
    - times_actioned (numpy.ndarray): An array containing the number of times each arm was pulled
    - action_values (numpy.ndarray): An array containing the estimated values of each arm
    - action_preferences (numpy.ndarray): An array containing the preferences for each arm
    - distribution (str): The distribution used to sample rewards ("Gaussian" or "Bernoulli")
    - q (numpy.ndarray): The true values of the arms based on the specified distribution
    - optimalAction (int): The optimal arm with the highest true value

    Methods:
    - get_times_actioned(): Returns an array of the number of times each arm was pulled
    - get_action_values(): Returns an array of the estimated values of each arm
    - get_action_preferences(): Returns an array of the preferences for each arm
    - get_optimal_action(): Returns the index of the optimal arm
    - get_reward(action): Computes the reward obtained by pulling a specific arm
    - updateValues(action, rewards, alpha, use_preferences): Updates action values or 
    preferences based on rewards
    - reset(): Resets the bandit's attributes to their initial values
    """

    def __init__(self, k, distribution, initialActionValue):
        """
        Initializes a Bandit object.

        Parameters:
        - k (int): The number of arms
        - distribution (str): The distribution used to sample rewards ("Gaussian" or "Bernoulli")
        - initialActionValue (float): The initial value assigned to each action
        """

        # The number of arms of the bandit
        self.k = k

        # The number of times a certain lever was actioned 
        self.times_actioned = np.zeros(k, dtype = np.int64)

        # The action values (Q_t(a) -> the estimated value of action a at time t)
        self.action_values = np.zeros(k, dtype = np.float64) + np.double(initialActionValue)

        # The action preferences (H_t(a) -> the preference for an action a at time t)
        self.action_preferences = np.zeros(k, dtype = np.float64)

        # The distribution that the bandit uses to sample rewards
        # in our program can be just "Gaussian" or "Bernoulli"
        self.distribution = distribution

        # Depending on the distribution, the true values of the actions
        # were sampled differently:
        # - "Gaussian" => the true values were sampled from a normal
        # distribution with mu = 0 and var = 1
        # - "Bernoulli" => the true values were sampled from an uniform
        # distribution between 0 and 1
        if self.distribution == "Gaussian":
            self.q = np.array([random.gauss(mu = 0, sigma = 1) for _ in range(k)])
        elif self.distribution == "Bernoulli":
            self.q = np.array([random.uniform(0, 1) for _ in range(k)])

        # The optimal action is the lever that has the highest true value
        self.optimalAction = np.random.choice(np.flatnonzero(self.q == self.q.max()))
     
    def get_times_actioned(self):
        """
        Returns:
        - numpy.ndarray: an array that contains the times each actioned was executed 
        """
        return self.times_actioned
    
    def get_action_values(self):
        """
        Returns:
        - numpy.ndarray: an array that contains the action values (Q(a)) 
        """
        return self.action_values
    
    def get_action_preferences(self):
        """
        Returns:
        - numpy.ndarray: an array that contains the action preferences (H(a))
        """
        return self.action_preferences
    
    def get_optimal_action(self):
        """
        Returns:
        - (int): the optimal action (the action that has the maximum true value)
        """
        return self.optimalAction
    
    def get_reward(self, action) -> float:
        """Computes the reward returned by the bandit depending on the distribution
        
        Parameters:
        - action (int): the action taken by the agent

        Returns:
        - (float): the reward obtained as a consequence of executing 
        a certain action (by the agent)

        """
        # Gaussian -> it samples the reward from a normal 
        # distribution with mean of q(a) and variance of 1
        # Bernoulli -> it samples the reward from a binomial
        # distribution with probability of success q(a)
        if self.distribution == "Gaussian":
            return random.gauss(mu = self.q[action], sigma = 1)
        elif self.distribution == "Bernoulli":
            return np.random.binomial(1, p = self.q[action])

    def updateValues(self, action, rewards, alpha, use_preferences):
        """ Update either the action values / action preferences =

        Parameters:
        - action (int): the action that the agent took
        - rewards (list[int]): the list of the past rewards obtained until the present moment
        - alpha (float): the step size (in case of action preferences)
        - use_preferences (bool): True in case of using action preferences, otherwise False

        Returns:
        - reward (float): the reward obtained as a consequence of executing the action
        
        """

        # If the action preferences are not used, the action values are used instead
        if use_preferences == False:
            # Update the number of times the chosen action was picked
            self.times_actioned[action] += 1

            # Compute the reward obtained as a result of executing the action
            reward = self.get_reward(action)

            # Update the action value of the executed action
            self.action_values[action] += (1 / self.times_actioned[action]) * (reward - self.action_values[action])
        else: # This is the case where action preferences are used
            # Compute the reward obtained as a result of executing the action
            reward = self.get_reward(action)

            # Compute the average result using the rewards obtained until the current moment
            avgReward = (sum(rewards) + reward) / (len(rewards) + 1)

            # Compute probabilities using softmax function
            expSum = np.sum(np.exp(self.action_preferences))
            probabilities = [np.exp(self.action_preferences[a]) / expSum for a in range(self.k)]

            # Update the action preferences
            for a in range(self.k):
                if a == action:
                    self.action_preferences[a] += alpha * (reward - avgReward) * (1 - probabilities[a])
                else:
                    self.action_preferences[a] -= alpha * (reward - avgReward) * probabilities[a]
        
        return reward
    
    def reset(self):
        """
        Reset the bandit's attributes to their initial values
        """
        self.times_actioned = np.zeros(self.k, dtype = np.int64)
        self.action_values = np.zeros(self.k, dtype = np.float64)
        self.action_preferences = np.zeros(self.k, dtype = np.float64)