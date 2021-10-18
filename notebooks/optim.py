import numpy as np
import policies
import utils
import dql
from functools import partial
from tqdm import tqdm

def noop(*args, **kwargs):
    pass

def q_learning(
    env,
    num_episodes=300,
    repetitions=5,
    discount_factor=1.0,
    alpha=0.1,
    epsilon=0.1,
    reduction="mean",
    callback=None,
):
    total_episode_lengths = []
    total_episode_returns = []

    for repetition in tqdm(
        range(repetitions),
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    ):
        nS = env.nS if isinstance(env.nS, (tuple, list)) else (env.nS,)
        Q = np.zeros((*nS, env.nA))
        policy = policies.EpsilonGreedyPolicy(Q, epsilon=epsilon, reduction=reduction)
        utils.seed(env, policy, seed=42 + repetition)

        _, (episode_length, episode_returns) = dql.q_learning(
            env, policy, Q,
            num_episodes=num_episodes,
            discount_factor=discount_factor,
            alpha=alpha,
            callback=partial(callback or noop, repetition=repetition),
        )
        total_episode_lengths.append(episode_length)
        total_episode_returns.append(episode_returns)

    total_episode_lengths = np.array(total_episode_lengths)
    total_episode_returns = np.array(total_episode_returns)
    return total_episode_lengths, total_episode_returns

def double_q_learning(
    env,
    num_episodes=300,
    repetitions=5,
    discount_factor=1.0,
    alpha=0.1,
    epsilon=0.1,
    reduction="mean",
    callback=None,
):
    total_episode_lengths = []
    total_episode_returns = []

    for repetition in tqdm(
        range(repetitions),
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    ):
        nS = env.nS if isinstance(env.nS, (tuple, list)) else (env.nS,)
        
        Q1 = np.zeros((*nS, env.nA))
        Q2 = np.zeros_like(Q1)
        policy = policies.EpsilonGreedyPolicy(Q1, Q2, epsilon=epsilon, reduction=reduction)
        utils.seed(env, policy, seed=42 + repetition)

        _, (episode_length, episode_returns) = dql.double_q_learning(
            env, policy, Q1, Q2,
            num_episodes=num_episodes,
            discount_factor=discount_factor,
            alpha=alpha,
            callback=partial(callback or noop, repetition=repetition),
        )
        total_episode_lengths.append(episode_length)
        total_episode_returns.append(episode_returns)

    total_episode_lengths = np.array(total_episode_lengths)
    total_episode_returns = np.array(total_episode_returns)
    return total_episode_lengths, total_episode_returns