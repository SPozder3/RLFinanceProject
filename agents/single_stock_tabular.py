from collections import defaultdict
from enum import Enum
import random
from typing import Tuple, Sequence, Callable, Dict

import numpy as np
import pandas as pd
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from tqdm import trange


class PortfolioAction(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2


def convert_prices_to_discrete_state(prev_data_df: pd.DataFrame, current_data_df: pd.DataFrame) -> Tuple:
    # if prev_data_df['tic'].tolist() == current_data_df['tic'].tolist():
    # for x, y in zip(prev_data_df['tic'].tolist(), current_data_df['tic'].tolist()):
    #    print(x == y)
    # raise IndexError("Dfs are wrong")
    percent_diffs = ((current_data_df['open'].to_numpy() - prev_data_df['open'].to_numpy()) / prev_data_df['open'].to_numpy()) * 100
    discrete = []
    for dif in percent_diffs:
        if dif > 5:
            discrete.append(0)
        elif 5 > dif > 0:
            discrete.append(1)
        elif 0 > dif > -5:
            discrete.append(2)
        elif -5 > dif:
            discrete.append(3)
        else:
            discrete.append(4)
    return tuple(discrete)


def argmax(arr: Sequence[float]) -> int:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum, so we define this method to use in EpsilonGreedy and UCB agents.
    Args:
        arr: sequence of values
    """
    largest = max(arr)
    in_case_of_ties = []
    for i in range(len(arr)):
        if arr[i] == largest:
            in_case_of_ties.append(i)
    return random.choice(in_case_of_ties)


def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        # You can reuse code from ex1
        # Make sure to break ties arbitrarily
        a_star = argmax(Q[state])
        if np.random.random() < epsilon:
            action = random.choice(list(PortfolioAction))
        else:
            probabilities = [(1 - epsilon + (epsilon / num_actions)) if a.value == a_star else epsilon / num_actions for
                             a in list(PortfolioAction)]
            return np.random.choice(list(PortfolioAction), p=probabilities)
        return action

    return get_action


def update_single_stock_percent(perc, act, aggresive: bool = False):
    if not aggresive:
        if act == PortfolioAction.BUY.value:
            perc += 0.1
        elif act == PortfolioAction.SELL.value:
            perc -= 0.1
        elif act == PortfolioAction.HOLD.value:
            perc = perc

        if perc > 1:
            perc = 1
        if perc < 0:
            perc = 0
        return perc
    if aggresive:
        if act == PortfolioAction.BUY.value:
            perc = 1
        elif act == PortfolioAction.SELL.value:
            perc = 0
        elif act == PortfolioAction.HOLD.value:
            perc = perc
        return perc


def softmax_normalization(actions):
    numerator = np.exp(actions)
    denominator = np.sum(np.exp(actions))
    softmax_output = numerator / denominator
    return softmax_output


def sarsa_single_stock(env: StockPortfolioEnv, num_episodes: int, gamma: float, epsilon: float, step_size: float,
                       stock: int = 0, q: Dict = None, aggressive: bool = False):
    """SARSA algorithm."""
    if q is None:
        Q = defaultdict(lambda: np.zeros(len(PortfolioAction)))
    else:
        Q = q
    policy = create_epsilon_policy(Q, epsilon)
    episodes = []
    for _ in trange(num_episodes, desc="Episode", leave=False):
        env.reset()
        # take initial step without investing anything to get a price to compare with
        env.step(np.zeros(28))

        previous_data = env.data
        env.step(np.zeros(28))
        current_data = env.data
        # create discrete state
        S = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)
        # given state which includes all stocks return buy sell hold for the single stock we are looking at
        A = policy(S)
        A = A.value

        episode = []
        percent = 0
        while True:
            # need to convert buy sell hold into a percentage for env step
            percent = update_single_stock_percent(percent, A, aggresive=aggressive)

            # convert our action into the portfolio percentages
            portfolio_breakdown = np.zeros(28)
            portfolio_breakdown[stock] = percent
            # print(portfolio_breakdown)
            # print(softmax_normalization(portfolio_breakdown))
            # TODO THE ENVIRONMENT AUTO NORMALIZES ACTION. CHANGE TO UPDATE REFLECT PORTFOLIO BREAKDOWN???
            # JUST HAVE 28 SARSA's and have softmax handle it in the environment
            next_state, reward, done, _, _ = env.step(portfolio_breakdown)

            # update ticker prices
            previous_data = current_data
            current_data = env.data

            # create discrete next state
            next_state = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)

            # record information
            episode.append((S, A, reward, percent))

            # get next action
            A_star = policy(next_state)
            # update
            A_star = A_star.value

            Q[S][A] = Q[S][A] + (step_size * (reward + (gamma * Q[next_state][A_star]) - Q[S][A]))
            S = next_state
            A = A_star

            if done:
                break

        episodes.append(episode)
    return env, episodes, Q



def generate_episode(env: StockPortfolioEnv, policy: Callable, stock: int = 0, aggressive:bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
    """
    episode = []
    state = env.reset()
    # get stock data to check movement
    env.step(np.zeros(28))
    previous_data = env.data
    env.step(np.zeros(28))
    current_data = env.data
    percent = 0
    S = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)

    while True:
        A = policy(S)
        A = A.value

        percent = update_single_stock_percent(percent, A, aggresive=aggressive)
        # convert our action into the portfolio percentages
        portfolio_breakdown = np.zeros(28)
        portfolio_breakdown[stock] = percent

        next_state, reward, done, _, _ = env.step(portfolio_breakdown)

        # update ticker prices
        previous_data = current_data
        current_data = env.data

        # create discrete next state
        next_state = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)

        # record information
        episode.append((S, A, reward, percent))
        S = next_state
        if done:
            break

    return episode


def on_policy_mc_control_single_stock(env: StockPortfolioEnv, num_episodes: int, gamma: float, epsilon: float, step_size: float,
                       stock: int = 0, q: Dict = None, aggressive: bool =False):
    if q is None:
        Q = defaultdict(lambda: np.zeros(len(PortfolioAction)))
    else:
        Q = q
    episodes = []
    returns = defaultdict(list)

    for _ in trange(num_episodes, desc="Episode", leave=False):
        policy = create_epsilon_policy(Q, epsilon)

        episode = generate_episode(env, policy, stock=stock, aggressive=aggressive)
        episodes.append(episode)
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            G = (gamma * G) + episode[t][2]

            current_state, current_action, current_reward, percent = episode[t - 1]

            # Check if it's a first visit to the (state, action) pair
            state_action = (current_state, current_action)
            if state_action not in [(step[0], step[1]) for step in episode[:t - 1]]:
                returns[state_action].append(G)
                Q[current_state][current_action] = np.mean(returns[state_action])

    return env, episodes, Q

def nstep_sarsa_single_stock(env: StockPortfolioEnv, num_episodes: int, gamma: float, epsilon: float, step_size: float, n: int, stock: int = 0, aggressive: bool = False, q: Dict = None):
    # Initialize Q
    if q is None:
        Q = defaultdict(lambda: np.zeros(len(PortfolioAction)))
    else:
        Q = q
    # Initialize pi
    policy = create_epsilon_policy(Q, epsilon)
    # Initialize trainsition data structures
    prev_states = []
    prev_actions = []
    prev_rewards = []
    episodes = []
    for _ in trange(num_episodes, desc="Episode", leave = False):
        env.reset()
        env.step(np.zeros(28))

        previous_data = env.data
        env.step(np.zeros(28))
        current_data = env.data

        # Initialize and store S_0 and A_0
        S = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)
        A = policy(S)
        A = A.value
        prev_states.append(S)
        prev_actions.append(A)

        episode = []
        percent = 0
        t_ep = 100
        step = 0
        tau = 0
        G = 0
        while True:
            if step < t_ep:
                percent = update_single_stock_percent(percent, A, aggresive=aggressive)
                portfolio_breakdown = np.zeros(28)
                portfolio_breakdown[stock] = percent
                next_state, reward, done, _, _ = env.step(portfolio_breakdown)

                previous_data = current_data
                current_data = env.data
                next_state = convert_prices_to_discrete_state(prev_data_df=previous_data, current_data_df=current_data)

                episode.append((S, A, reward, percent))
                prev_states.append(next_state)
                prev_actions.append(A)
                prev_rewards.append(reward)
                if done:
                    t_ep = step + 1
                else:
                    A_star = policy(next_state)
                    A_star = A_star.value
            tau = step - n + 1
            if tau >= 0:
                G = np.sum([np.power(gamma, (i - tau - 1)) * prev_rewards[i] for i in range(tau + 1, min(tau + n, step))])
                if tau + n < step:
                    G += np.power(gamma, n) * Q[prev_states[(tau+n)]][prev_actions[(tau+n)]]
                Q[prev_states[tau]][prev_actions[tau]] += step_size * (G - Q[prev_states[tau]][prev_actions[tau]])
                t_ep += 1
            S = next_state
            A = A_star
            step += 1
            if done:
                break
        episodes.append(episode)
    return env, episodes, Q

