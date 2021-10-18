import numpy as np

def numeric_tuple(state):
    state = state if isinstance(state, (tuple, list)) else (state,)
    return tuple(np.array(state, dtype=int))
    

def q_learning(
    env, policy, Q,
    num_episodes=300, discount_factor=1.0, alpha=0.5, callback=None,
):
    episode_lengths = np.zeros(num_episodes)
    episode_returns = np.zeros(num_episodes)
    
    for ep in range(num_episodes):
        dur, R = 0, 0
        # state = tuple(env.reset())
        state = numeric_tuple(env.reset())
        # print(state)
        while True:
            action = policy.sample_action(state, env.legal_actions(state))
            new_state, reward, done, _ = env.step(action)
            new_state = numeric_tuple(new_state)
            # print(new_state)
            R = discount_factor * R + reward
            
            if callback:
                callback(
                    env=env,
                    ep=ep,
                    state=state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    done=done,
                    dur=dur,
                    R=R
                )
            
            Q[state, action] += alpha * (
                reward + discount_factor * np.max(Q[new_state][env.legal_actions(new_state)]) - Q[state][action]
            )
            
            state = new_state
            dur += 1
            if done:
                break
        episode_lengths[ep] += dur
        episode_returns[ep] += R
            
    return Q, (
        episode_lengths,
        episode_returns,
    )

def double_q_learning(
    env, policy, Q1, Q2,
    num_episodes=300, discount_factor=1.0, alpha=0.5, reduction="sum", callback=None,
):
    episode_lengths = np.zeros(num_episodes)
    episode_returns = np.zeros(num_episodes)
    
    for ep in range(num_episodes):
        dur, R = 0, 0
        # state = env.reset()
        state = numeric_tuple(env.reset())
        while True:
            action = policy.sample_action(state, env.legal_actions(state))
            
            new_state, reward, done, _ = env.step(action)
            new_state = numeric_tuple(new_state)
            R = discount_factor * R + reward
            
            if callback:
                callback(
                    env=env,
                    ep=ep,
                    state=state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    done=done,
                    dur=dur,
                    R=R
                )
            
            if np.random.uniform() <= 0.5:
                Q1[state, action] += alpha * (
                    reward + discount_factor * Q2[new_state][np.argmax(Q1[new_state][env.legal_actions(new_state)])] - Q1[state][action]
                )
            else:
                Q2[state, action] += alpha * (
                    reward + discount_factor * Q1[new_state][np.argmax(Q2[new_state][env.legal_actions(new_state)])] - Q2[state][action]
                )
            
            state = new_state
            dur += 1
            if done:
                break
        
        episode_lengths[ep] += dur
        episode_returns[ep] += R
            
    if reduction == "sum":
        Q = np.array([Q1, Q2]).sum(axis=0)
    elif reduction == "mean":
        Q = np.array([Q1, Q2]).mean(axis=0)
    else:
        raise ValueError("unknown reduction function: %s" % reduction)
    return (Q, Q1, Q2), (
        episode_lengths,
        episode_returns,
    )