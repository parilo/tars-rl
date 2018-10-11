import os
import numpy as np


def preprocess_obs(obs):

    res = []
    # joints
    for joint_val in ["joint_pos", "joint_vel"]:
        for joint in ["ground_pelvis", "hip_r", "hip_l", "back",
                      "knee_r", "knee_l", "ankle_r", "ankle_l"]:
            res += obs[joint_val][joint]
            
    res[3] = 0.
    res[5] = 0.

    # muscles
    muscles = ["abd_r", "add_r", "hamstrings_r", "bifemsh_r",
               "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
               "abd_l", "add_l", "hamstrings_l", "bifemsh_l",
               "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
               "gastroc_l", "soleus_l", "tib_ant_l"]
    for muscle in muscles:
        res += [obs["muscles"][muscle]["activation"]]
        res += [obs["muscles"][muscle]["fiber_length"]]
    return res
