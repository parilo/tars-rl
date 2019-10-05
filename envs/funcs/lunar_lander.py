import numpy as np


# warning those function doesn't work properly
# need to modify env to get it working
# https://github.com/openai/gym/wiki/Leaderboard#lunarlander-v2

def get_prev_obs(agent_buffers):
    return np.array([
        agent_buf.get_last_obs()[0]
        for agent_buf in agent_buffers
    ])


def is_state_still(action, state):
    print('--- state', np.abs(state[:, 2]), np.abs(state[:, 3]))
    return \
        ((
            (np.abs(state[:, 2]) < 0.0001).astype(np.int32) *
            (np.abs(state[:, 3]) < 0.0001).astype(np.int32) *
            (action[:, 0] <= 0).astype(np.int32) *
            (action[:, 1] >= -0.5).astype(np.int32) *
            (action[:, 1] <= 0.5).astype(np.int32)
        ) > 0).astype(np.int32)


def is_still(agent_buffers, action, next_state):
    prev_obs = get_prev_obs(agent_buffers)
    still_prev = is_state_still(action, prev_obs)
    still_next = is_state_still(action, next_state[0][:, -1])
    return ((still_prev + still_next) > 0).astype(np.int32)


def is_done(agent_buffers, action, next_state):
    # done = False
    # if self.game_over or abs(next_state[0]) >= 1.0:
    #     done = True
    #     reward = -100
    # if not self.lander.awake:
    #     done = True
    #     reward = +100

    # collision with the surface is ignored
    out_of_range = (np.abs(next_state[0][:, -1, 0]) >= 1.0).astype(np.int32)
    is_still_val = is_still(agent_buffers, action, next_state)


    return ((out_of_range + is_still_val) > 0).astype(np.int32)


def calc_shaping(obs):
    return \
        - 100 * np.sqrt(obs[:, 0] * obs[:, 0] + obs[:, 1] * obs[:, 1]) \
        - 100 * np.sqrt(obs[:, 2] * obs[:, 2] + obs[:, 3] * obs[:, 3]) \
        - 100 * abs(obs[:, 4]) + 10 * obs[:, 6] + 10 * obs[:, 7]


def calc_reward(agent_buffers, action, next_state):

    # reward = 0
    prev_obs = get_prev_obs(agent_buffers)
    prev_shaping = calc_shaping(prev_obs)
    shaping = calc_shaping(next_state[0][:, -1])

    # And ten points for legs contact, the idea is if you
    # lose contact again after landing, you get negative reward
    reward = shaping - prev_shaping

    m_power = (np.clip(action[:, 0], 0.0, 1.0) + 1.0) * 0.5
    s_power = np.clip(np.abs(action[:, 1]), 0.5, 1.0)

    reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
    reward -= s_power * 0.03

    out_of_range = (np.abs(next_state[0][:, -1, 0]) >= 1.0).astype(np.int32)
    reward[out_of_range == 1] -= 100
    is_still_val = is_still(agent_buffers, action, next_state)
    reward[is_still_val == 1] += 100

    return reward
