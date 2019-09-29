import numpy as np


# https://github.com/openai/gym/wiki/Pendulum-v0

def pendulum_is_done(agent_buffers, action, next_state):
    is_done_list = []
    for agent_buf in agent_buffers:
        is_done_list.append(agent_buf.get_episode_len() >= 100)
    return np.array(is_done_list).astype(np.int32)


def pendulum_calc_reward(agent_buffers, action, next_state):
    sin = next_state[0][:, -1, 1]
    cos = next_state[0][:, -1, 0]
    dangle = next_state[0][:, -1, 2]
    theta = np.arctan2(sin, cos)
    return -(theta ** 2 + 0.1 * dangle ** 2 + 0.001 * action[:, 0] ** 2)


# https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2

def lunar_lander_get_prev_obs(agent_buffers):
    return np.array([
        agent_buf.get_last_obs()[0]
        for agent_buf in agent_buffers
    ])


def lunar_lander_is_state_still(action, state):
    return \
        ((
            (np.abs(state[:, 2]) < 0.01).astype(np.int32) +
            (np.abs(state[:, 3]) < 0.01).astype(np.int32) +
            (action[:, 0] <= 0).astype(np.int32) +
            (action[:, 1] >= -0.5).astype(np.int32) +
            (action[:, 1] <= 0.5).astype(np.int32)
        ) > 0).astype(np.int32)


def lunar_lander_is_still(agent_buffers, action, next_state):
    prev_obs = lunar_lander_get_prev_obs(agent_buffers)
    still_prev = lunar_lander_is_state_still(action, prev_obs)
    still_next = lunar_lander_is_state_still(action, next_state)
    return ((still_prev + still_next) > 0).astype(np.int32)


def lunar_lander_is_done(agent_buffers, action, next_state):
    # done = False
    # if self.game_over or abs(next_state[0]) >= 1.0:
    #     done = True
    #     reward = -100
    # if not self.lander.awake:
    #     done = True
    #     reward = +100

    out_of_range = (np.abs(next_state[:, 0]) >= 1.0).astype(np.int32)
    is_still = lunar_lander_is_still(agent_buffers, action, next_state)

    return ((out_of_range + is_still) > 0).astype(np.int32)


def lunar_lander_calc_shaping(obs):
    return \
        - 100 * np.sqrt(obs[:, 0] * obs[:, 0] + obs[:, 1] * obs[:, 1]) \
        - 100 * np.sqrt(obs[:, 2] * obs[:, 2] + obs[:, 3] * obs[:, 3]) \
        - 100 * abs(obs[:, 4]) + 10 * obs[:, 6] + 10 * obs[:, 7]


def lunar_lander_reward(agent_buffers, action, next_state):

    # reward = 0
    prev_obs = lunar_lander_get_prev_obs(agent_buffers)
    prev_shaping = lunar_lander_calc_shaping(prev_obs)
    shaping = lunar_lander_calc_shaping(next_state[0])

    # And ten points for legs contact, the idea is if you
    # lose contact again after landing, you get negative reward
    reward = shaping - prev_shaping

    m_power = (np.clip(action[:, 0], 0.0, 1.0) + 1.0) * 0.5
    s_power = np.clip(np.abs(action[1]), 0.5, 1.0)

    reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
    reward -= s_power * 0.03

    out_of_range = (np.abs(next_state[:, 0]) >= 1.0).astype(np.int32)
    is_still = lunar_lander_is_still(agent_buffers, action, next_state)

    return reward
