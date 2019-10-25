import numpy as np


# https://github.com/openai/gym/wiki/Pendulum-v0

def is_done(agent_buffers, action, next_state):
    is_done_list = []
    for agent_buf in agent_buffers:
        is_done_list.append(agent_buf.get_episode_len() >= 100)
    return np.array(is_done_list).astype(np.int32)


def calc_reward(agent_buffers, action, next_state):
    sin = next_state[0][:, -1, 1]
    cos = next_state[0][:, -1, 0]
    dangle = next_state[0][:, -1, 2]
    theta = np.arctan2(sin, cos)
    return -(theta ** 2 + 0.1 * dangle ** 2 + 0.001 * action[:, 0] ** 2)
