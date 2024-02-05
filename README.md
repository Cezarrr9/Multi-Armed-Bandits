# Multi-Armed-Bandits

The following code represents an implementation of several multi-armed bandit problems. There are two types of bandits:
• The Gaussian bandit: a multi-armed bandit in which the reward obtained from each action is
sampled from a normal distribution.
• The Bernoulli bandit: a multi-armed bandit in which the reward obtained from each action is
sampled from a Bernoulli distribution (each arm has probability p to return 1 and 1−p probability
to return 0).
The goal of the agent for each bandit problem is to learn an optimal policy π∗, i.e. the action that brings the maximum reward. This goal is reached through learning. I created a set of N randomly generated k-armed bandit problems for both bandit scenarios, where both k and N are parameters of your choice. For each of those problems, I trained an agent with different exploration methods:
• Greedy and -greedy
• Optimistic initial values
• Upper-Confidence Bound
• Softmax Policy with Action Preferences
N experiments are performed for each exploration method. Each experiment will consist of a number of training steps T. At the end of each training run, we expect your agent to have learned to recognize the action (or actions) that allow it to obtain the maximum possible reward. The learning performance of the agent is measured by monitoring the average reward it obtains, as well as the percentage of times the agent chooses the best action.