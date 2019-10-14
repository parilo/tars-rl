import numpy as np

from rl_server.server.agent_replay_buffer import AgentBuffer


# https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d.py


def is_done(agent_buffers, action, next_state):
    # height, ang = self.sim.data.qpos[1:3]
    # done = not (height > 0.8 and height < 2.0 and
    #             ang > -1.0 and ang < 1.0)

    height_ang = next_state[0][:, -1, :2]
    # height_out_of_range = np.logical_or(height_ang[:, 0] < 0.8, height_ang[:, 0] > 2.0)
    # ang_out_of_range = np.logical_or(height_ang[:, 1] < -1, height_ang[:, 1] > 1)
    height_out_of_range = np.logical_or(height_ang[:, 0] < 0.85, height_ang[:, 0] > 1.95)
    ang_out_of_range = np.logical_or(height_ang[:, 1] < -0.9, height_ang[:, 1] > 0.9)
    return np.logical_or(height_out_of_range, ang_out_of_range).astype(np.int32)


def calc_reward(agent_buffers: AgentBuffer, action, next_state):
    # posafter, height, ang = self.sim.data.qpos[0:3]
    # alive_bonus = 1.0
    # reward = ((posafter - posbefore) / self.dt)
    # reward += alive_bonus
    # reward -= 1e-3 * np.square(a).sum()

    # obs_before = np.stack([buf.get_last_obs()[0] for buf in agent_buffers], axis=0)
    # print('--- before', obs_before[:, 1:3].tolist())
    # print('--- after', next_state[0][:, -1, 1:3].tolist())

    # posbefore = obs_before[:, 0]
    # posafter = next_state[0][:, -1, 0]
    # reward = (posafter - posbefore) / 0.1
    reward = next_state[0][:, -1, 8]  # x vel
    # alive_bonus = 1.0
    # reward += alive_bonus
    # reward -= 1e-3 * np.square(action).sum(axis=1)
    # reward -= 1e-3 * np.abs(action).sum(axis=1)

    return reward
