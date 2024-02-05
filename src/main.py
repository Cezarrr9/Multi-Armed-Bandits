from bandit import Bandit
from agent import Agent

import matplotlib.pyplot as plt
import numpy as np

def plot_bandit_results(resultsGaussian, resultsBernoulli, policy_names):
    """
    Plot the results of bandit experiments for both Gaussian and Bernoulli distributions.

    Parameters:
    - results_gaussian (dict): Results of experiments for Gaussian distribution
    - results_bernoulli (dict): Results of experiments for Bernoulli distribution
    - policy_names (dict): Mapping of policy IDs to policy names for legend labeling
    """

    plt.figure(figsize=(22, 10))

    # Subplot for Average Rewards Gaussian Bandit
    plt.subplot(221)
    for policy_id, (R, A) in resultsGaussian.items():
        averageRewards = R.mean(axis=0)
        plt.plot(averageRewards, label=f'{policy_names[policy_id]}')
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Gaussian Bandit")
    plt.grid()
    plt.legend(loc = "lower right")

    # Subplot for Optimal Action Percent Gaussian Bandit
    plt.subplot(222)
    for policy_id, (R, A) in resultsGaussian.items():
        percentageOptimalAction = (A.mean(axis=0) * 100)
        plt.plot(percentageOptimalAction, label=f'{policy_names[policy_id]}')
    plt.xlabel("Step")
    plt.ylabel("Optimal Action Percent")
    plt.title("Percentage of Optimal Actions Gaussian Bandit")
    plt.grid()
    plt.legend(loc = "lower right")

    # Subplot for Average Rewards Bernoulli Bandit
    plt.subplot(223)
    for policy_id, (R, A) in resultsBernoulli.items():
        averageRewards = R.mean(axis=0)
        plt.plot(averageRewards, label=f'{policy_names[policy_id]}')
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards Bernoulli Bandit")
    plt.grid()
    plt.legend(loc = "lower right")

    # Subplot for Optimal Action Percent Bernoulli Bandit
    plt.subplot(224)
    for policy_id, (R, A) in resultsBernoulli.items():
        percentageOptimalAction = (A.mean(axis=0) * 100)
        plt.plot(percentageOptimalAction, label=f'{policy_names[policy_id]}')
    plt.xlabel("Step")
    plt.ylabel("Optimal Action Percent")
    plt.title("Percentage of Optimal Actions Bernoulli Bandit")
    plt.grid()
    plt.legend(loc = "lower right")

    plt.savefig('data/results.png', format = 'png')

def initializeBandits(k, distribution, initialActionValue, numberOfProblems):
    """
    Initializes bandit instances for experiments.

    Parameters:
    - k (int): Number of arms in the bandit
    - distribution (str): Type of distribution for bandit rewards ("Gaussian" or "Bernoulli")
    - initial_action_value (float): Initial value assigned to each action
    - number_of_problems (int): Number of bandit instances for the experiments

    Returns:
    list: List of Bandit instances.
    """
    bandits = []
    for _ in range(numberOfProblems):
        bandit = Bandit(k, distribution, initialActionValue)
        bandits.append(bandit)
    
    return bandits

def runExperiment(agent,
                  policy,
                  numberOfSteps,
                  numberOfProblems,
                  k, 
                  distribution,
                  initialActionValue = 0,
                  epsilon = None,
                  c = None,
                  alpha = None,
                  use_preferences = False):
    """
    Runs a bandit experiment with the specified parameters

    Parameters:
    - agent (Agent): An instance of the Agent class
    - policy (str): The action selection policy to use
    - numberOfSteps (int): Number of steps in the experiment
    - numOfProblems (int): Number of bandit instances to run
    - k (int): Number of arms in the bandit
    - distribution (str): Type of distribution for bandit rewards ("Gaussian" or "Bernoulli")
    - initialActionValue (float): Initial value assigned to each action
    - epsilon (float): Exploration parameter for epsilon-greedy policy
    - c (float): Confidence level parameter for UCB policy
    - alpha (float): Step size parameter for softmax policy with preferences
    - use_preferences (bool): Flag indicating whether to use action preferences 
    in the softmax policy.

    Returns:
    tuple: Tuple containing two numpy arrays - rewards and actions for each experiment and step
    """
    
    # Initialize the bandits used in the experiment
    bandits = initializeBandits(k, distribution, initialActionValue, numberOfProblems)

    # Rewards for each experiment and step (reward history)
    R = np.zeros((numberOfProblems, numberOfSteps))

    # Actions for each experiment and step (1 if optimal action was taken)  
    A = np.zeros((numberOfProblems, numberOfSteps))  

    # Iterate through all the bandits 
    for problem_idx, bandit in enumerate(bandits):
        rewards = [] # Temporary list to hold rewards for the softmax policy calculation
        optimalAction = bandit.get_optimal_action()

        # Iterate through all the steps 
        for step in range(numberOfSteps):
            # Get some useful attributes
            action_values = bandit.get_action_values()
            times_actioned = bandit.get_times_actioned()
            action_preferences = bandit.get_action_preferences()

            # At each step the agent chooses an action to execute based on a policy
            action = agent.chooseAction(policy, action_values, action_preferences, 
                                        times_actioned, epsilon, c)
            
            # Then the bandit updates its action values / action preferences 
            # The function update values also returns the reward obtained as a result 
            # of executing the action
            reward = bandit.updateValues(action, rewards, alpha, use_preferences)

            # Update the list for softmax calculation if needed
            rewards.append(reward)  

            # Store the reward received by the bandit on this step in the reward history
            R[problem_idx, step] = reward

            # Store 1 if the action taken was the optimal action, otherwise store 0 
            A[problem_idx, step] = int(action == optimalAction)
        
        # Reset the attributes of the bandit
        bandit.reset()

    # Return the reward and action history as a tuple
    return R, A

if __name__ == "__main__":

    # The hyper-parameters for each experiment
    numberOfSteps = 1000
    numberOfProblems = 2000
    k = 10

    # Dictionaries for storing the results from both bandits 
    resultsGaussian = {}
    resultsBernoulli = {}

    # Agent that executes the actions
    agent = Agent()

    # Map policy IDs to policy names for legend labeling
    policy_names = {
        0: "Greedy",
        1: "Epsilon-greedy",
        2: "Optimistic Initial Values (epsilon-greedy)",
        3: "Upper Confidence Bound",
        4: "Softmax (action-preferences)"
    }

    # Run the experiments with all the policies for both bandits

    # greedy -> Gaussian
    R, A = runExperiment(policy = "greedy",
                        agent = agent, 
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k, 
                        distribution = "Gaussian")
    resultsGaussian[0] = (R, A)

    # epsilon-greedy -> Gaussian
    R, A = runExperiment(policy = "epsilon-greedy",
                        agent = agent,
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Gaussian",
                        epsilon = 0.1)
    resultsGaussian[1] = (R, A)
    
    # optimistic initial values -> Gaussian
    R, A = runExperiment(policy = "epsilon-greedy",
                        agent = agent,
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Gaussian", 
                        epsilon = 0.1,
                        initialActionValue = 5)
    resultsGaussian[2] = (R, A)

    # upper confidence bound -> Gaussian
    R, A = runExperiment(policy = "UCB",
                        agent = agent, 
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Gaussian", 
                        c = 2)
    resultsGaussian[3] = (R, A)

    # softmax action-preferences -> Gaussian
    R, A = runExperiment(policy = "softmax",
                        agent = agent,
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Gaussian",
                        use_preferences = True, 
                        alpha = 0.1)
    resultsGaussian[4] = (R, A)

    # greedy -> Bernoulli
    R, A = runExperiment(policy = "greedy",
                        agent = agent,
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k, 
                        distribution = "Bernoulli")
    resultsBernoulli[0] = (R, A)

    # epsilon-greedy -> Bernoulli
    R, A = runExperiment(policy = "epsilon-greedy",
                        agent = agent, 
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Bernoulli",
                        epsilon = 0.1)
    resultsBernoulli[1] = (R, A)
    
    # optimistic initial values -> Bernoulli
    R, A = runExperiment(policy = "epsilon-greedy",
                        agent = agent, 
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Bernoulli", 
                        epsilon = 0.1,
                        initialActionValue=5)
    resultsBernoulli[2] = (R, A)

    # upper confidence bound -> Bernoulli
    R, A = runExperiment(policy = "UCB",
                        agent = agent,
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Bernoulli", 
                        c = 2)
    resultsBernoulli[3] = (R, A)

    # softmax action-preferences -> Bernoulli
    R, A = runExperiment(policy = "softmax",
                        agent = agent,
                        numberOfSteps = numberOfSteps,
                        numberOfProblems = numberOfProblems,
                        k = k,
                        distribution = "Bernoulli",
                        use_preferences = True, 
                        alpha = 0.1)
    resultsBernoulli[4] = (R, A)

    # Plot average rewards and optimal action % for both bandits 
    plot_bandit_results(resultsGaussian, resultsBernoulli, policy_names)