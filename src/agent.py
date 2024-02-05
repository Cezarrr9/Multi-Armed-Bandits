import numpy as np

class Agent:
    """ An agent class implementing different action selection policies for 
    a multi-armed bandit problem.

    Methods:
    - chooseAction(policy, action_values=None, action_preferences=None, 
    times_actioned=None, epsilon=None, c=None, use_preferences=False): chooses an action
    based on the specified policy
    - epsilonGreedy(epsilon, action_values): chooses an action 
    based on the epsilon-greedy policy
    - ucb(action_values, times_actioned, c): chooses an action based on the 
    upper-confidence bound (UCB) policy
    - softmax(use_preferences = False, action_values = None, action_preferences = None): 
    chooses an action based on the softmax policy

    """
    
    def chooseAction(self,
                     policy,
                     action_values = None,
                     action_preferences = None,
                     times_actioned = None,
                     epsilon = None,
                     c = None):
        """
        Chooses an action based on the specified policy.

        Parameters:
        - policy (str): The action selection policy ("greedy", "epsilon-greedy",
        "softmax", "UCB").
        - action_values (numpy.ndarray): Estimated values of each action (used in
        greedy and epsilon-greedy policies)
        - action_preferences (numpy.ndarray): Preferences for each action (used in
        softmax policy)
        - times_actioned (numpy.ndarray): Number of times each action
        has been taken 
        - epsilon (float): Exploration parameter for epsilon-greedy policy
        - c (float): Confidence level parameter for UCB policy

        Returns:
        - int: The chosen action.
        """

        if policy == "greedy":
            chosenAction = self.epsilonGreedy(epsilon = 0, action_values = action_values)
        elif policy == "epsilon-greedy":
            chosenAction = self.epsilonGreedy(epsilon = epsilon, action_values = action_values)
        elif policy == "softmax":
            chosenAction = self.softmax(action_preferences = action_preferences)
        elif policy == "UCB":
            chosenAction = self.ucb(action_values, times_actioned, c)
            
        return chosenAction

    def epsilonGreedy(self, epsilon, action_values):
        """ 
        Chooses an action based on the epsilon-greedy policy.

        Parameters:
        - epsilon (float): Exploration parameter
        - action_values (numpy.ndarray): Estimated values of each action.

        Returns:
        int: The chosen action.
        """

        # Find the action with the highest estimated action value
        greedyAction = np.random.choice(np.flatnonzero(action_values == action_values.max()))

        # Set the probabilities for each action to be selected:
        # -> 1-epsilon + epsilon / len(action_values) for the greedy action
        # -> epsilon / len(action_values) for every other action
        length = len(action_values)
        probabilities = [epsilon/length] * length
        probabilities[greedyAction] += 1 - epsilon

        # Choose randomly an action with the given probabilities 
        chosenAction = np.random.choice(range(length), p = probabilities)
        
        # Return the chosen action
        return chosenAction

    def ucb(self, action_values, times_actioned, c):
        """
        Chooses an action based on the upper-confidence bound (UCB) policy

        Parameters:
        - action_values (numpy.ndarray): Estimated values of each action
        - times_actioned (numpy.ndarray): Number of times each action has been taken
        - c (float): Confidence level parameter

        Returns:
        int: The chosen action.
        """

        # Compute the current step
        step = sum(times_actioned)

        # Create an empty list for storing the values of the upper bounds 
        # on the true values of the actions 
        values = np.zeros(len(action_values), dtype = np.float64)

        # Going through each action and compute the confidence bound
        for a in range(len(action_values)):
            if times_actioned[a]:
                cb = action_values[a] + c * np.sqrt(np.log(step) / times_actioned[a])
                values[a] = cb
            else: 
                # If an actioned was not taken before, 
                # it is considered to be a maximizing function, thus it is chosen by default
                return a
        
        # If all the actions were selected at least one previously,
        # select a random action from those which maximize the upper bounds
        chosenAction = np.random.choice(np.flatnonzero(values == values.max()))

        # Return the chosen action
        return chosenAction
    
    def softmax(self, action_preferences = None):
        """
        Chooses an action based on the softmax policy.

        Parameters:
        - action_preferences (numpy.ndarray): Preferences for each
        action (used if use_preferences is True)

        Returns:
        int: The chosen action.
        """

        expSum = np.sum(np.exp(action_preferences))
        probabilities = [np.exp(action_preferences[a]) / expSum for a in range(len(action_preferences))]
        chosenAction = np.random.choice(range(len(action_preferences)), p = probabilities)
        
        return chosenAction