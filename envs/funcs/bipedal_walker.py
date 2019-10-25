import numpy as np


# https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py

# state = [
#     self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
#     2.0 * self.hull.angularVelocity / FPS,
#     0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
#     0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
#     self.joints[0].angle,
#     # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
#     self.joints[0].speed / SPEED_HIP,
#     self.joints[1].angle + 1.0,
#     self.joints[1].speed / SPEED_KNEE,
#     1.0 if self.legs[1].ground_contact else 0.0,
#     self.joints[2].angle,
#     self.joints[2].speed / SPEED_HIP,
#     self.joints[3].angle + 1.0,
#     self.joints[3].speed / SPEED_KNEE,
#     1.0 if self.legs[3].ground_contact else 0.0
# ]
# state += [l.fraction for l in self.lidar]

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5


def is_done(agent_buffers, action, next_state):
    # done = False
    # if self.game_over or pos[0] < 0:
    #     reward = -100
    #     done = True
    # if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
    #     done = True

    too_long_ep = np.array([agent_buf.get_episode_len() >= 200 for agent_buf in agent_buffers])
    lidar_too_close = np.any(next_state[0][:, -1, 14:] < 0.2, axis=1)
    return np.logical_or(too_long_ep, lidar_too_close).astype(np.int32)


def calc_reward(agent_buffers, action, next_state):
    # shaping = 130 * pos[0] / SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
    # shaping -= 5.0 * abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished
    #
    # reward = 0
    # if self.prev_shaping is not None:
    #     reward = shaping - self.prev_shaping
    # self.prev_shaping = shaping
    #
    # for a in action:
    #     reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
    #     # normalized to about -50.0 using heuristic, more optimal agent should spend less
    #
    # done = False
    # if self.game_over or pos[0] < 0:
    #     reward = -100
    #     done = True
    # if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
    #     done = True

    next_obs = next_state[0][:, -1]
    reward = next_obs[:, 2] * 130 / 0.3 / VIEWPORT_W * FPS
    # reward += -1 * np.sign(next_obs[:, 1]) * next_obs[:, 1] * 5 / 2.0 * FPS
    # reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(action), 0, 1).sum(axis=-1)
    lidar_too_close = np.any(next_state[0][:, -1, 14:] < 0.2, axis=1)
    reward[np.where(lidar_too_close)] -= 100

    return reward
