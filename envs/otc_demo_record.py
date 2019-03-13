#!/usr/bin/env python

import os
import pickle

import numpy as np
import cv2
from obstacle_tower_env import ObstacleTowerEnv
import pygame

from rl_server.server.agent_replay_buffer import AgentBuffer
from misc.common import create_if_need

# action_remap_discrete: [
#     0,  # no
#     3,  # j
#     6,  # cam left
#     12,  # cam right
#     18,  # forward
#     21,  # jump + forward
#     24,  # forw + cam left
#     27,  # f + j + cam left
#     30,  # f + cam right
#     33  # f + j + cam right
#     36  # backward
# ]


action_map = {
    0: 0,  # no
    3: 1,  # j
    6: 2,  # cam left
    12: 3,  # cam right
    18: 4,  # forward
    21: 5,  # jump + forward
    24: 6,  # forw + cam left
    27: 7,  # f + j + cam left
    30: 8,  # f + cam right
    33: 9,  # f + j + cam right
    36: 10  # backward
}


SEED = 22
DEMO_EPS_DIR = '/home/anton/devel/otc/otc-rl/logs/demo_eps'
START_EPS_INDEX = 20
create_if_need(DEMO_EPS_DIR)


def main():

    env = ObstacleTowerEnv(
        '/home/anton/devel/otc/ObstacleTower/obstacletower.x86_64',
        retro=True,
        worker_id=SEED,
        realtime_mode=True
    )

    pygame.init()

    size = 320, 240
    black = 0, 0, 0
    screen = pygame.display.set_mode(size)

    act = 0

    def create_demo_buffer():
        return AgentBuffer(
            100000,
            [[3, 84, 84], [8]],
            ["uint8", "float32"],
            10,  # action_size,
            True  # discrete_actions
        )

    def process_obs(obs, info=None, reward=None):
        if info:
            vec_obs = info['brain_info'].vector_observations[0]
            vec_obs[6] /= 3000.
            vec_obs = np.array(vec_obs.tolist() + [reward], dtype=np.float32)
        else:
            vec_obs = np.zeros((8,), dtype=np.float32)
            vec_obs[0] = 1
            vec_obs[6] = 1
        return [np.transpose(obs, (2, 0, 1)), vec_obs]

    def print_eps_stats(episode):
        print('episode', len(episode[2]), np.sum(episode[2]))

    demo_buffer = create_demo_buffer()
    demo_buffer.push_init_observation(process_obs(env.reset()))

    stored_episode_index = START_EPS_INDEX

    while 1:
        for event in pygame.event.get():
            # if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    act += 18
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    act += 36
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    act += 6
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    act += 12
                elif event.key == pygame.K_SPACE:
                    act += 3
                elif event.key == pygame.K_b:
                    print(demo_buffer.pointer)

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    act -= 18
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    act -= 36
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    act -= 6
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    act -= 12
                elif event.key == pygame.K_SPACE:
                    act -= 3

        env_action = action_map[act] if act in action_map else 0

        observation, reward, done, info = env.step(act)
        demo_buffer.push_transition([process_obs(observation, info, reward), env_action, reward, done])
        cv2.imshow(
            'demo record',
            cv2.resize(observation, (800, 800))
        )
        cv2.waitKey(3)

        screen.fill(black)
        pygame.display.flip()

        if done:
            episode = demo_buffer.get_complete_episode()
            print_eps_stats(episode)
            ep_path = os.path.join(DEMO_EPS_DIR, 'episode_' + str(stored_episode_index) + '.pkl')
            with open(ep_path, 'wb') as f:
                pickle.dump(episode, f, pickle.HIGHEST_PROTOCOL)
            stored_episode_index += 1

            demo_buffer = create_demo_buffer()
            demo_buffer.push_init_observation(process_obs(env.reset()))

if __name__ == "__main__":

    main()
