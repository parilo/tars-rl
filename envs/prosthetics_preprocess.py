import os
import math
import numpy as np


norm_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'prosthetics_norm_v21.npz')
with np.load(norm_file) as data:
    obs_means = data['means']
    obs_stds = data['stds']

norm_file_mini = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'prosthetics_norm_mini.npz')
with np.load(norm_file_mini) as data:
    obs_mini_means = data['means']
    obs_mini_stds = data['stds']


# Calculates Rotation Matrix given euler angles.
def euler_angles_to_rotation_matrix(theta):
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])]
    ])
    R_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])]
    ])
    R_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def preprocess_obs(obs):

    res = []

    force_mult = 5e-4
    acc_mult = 1e-2

    # body parts
    body_parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l',
                  'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
    # calculate linear coordinates of pelvis to switch
    # to the reference frame attached to the pelvis
    x, y, z = obs['body_pos']['pelvis']
    rx, ry, rz = obs['body_pos_rot']['pelvis']
    res += [y, z]
    res += [rx, ry, rz]
    # 30 components -- relative linear coordinates of body parts (except pelvis)
    for body_part in body_parts:
        if body_part != 'pelvis':
            x_, y_, z_ = obs['body_pos'][body_part]
            res += [x_-x, y_-y, z_-z]
    # 2 components -- relative linear coordinates of center mass
    x_, y_, z_ = obs['misc']['mass_center_pos']
    res += [x_-x, y_-y, z_-z]
    # 35 components -- linear velocities of body parts (and center mass)
    for body_part in body_parts:
        res += obs['body_vel'][body_part]
    res += obs['misc']['mass_center_vel']
    # 35 components -- linear accelerations of body parts (and center mass)
    for body_part in body_parts:
        res += obs['body_acc'][body_part]
    res += obs['misc']['mass_center_acc']
    # calculate angular coordinates of pelvis to switch
    # to the reference frame attached to the pelvis
    # 30 components -- relative angular coordinates of body parts (except pelvis)
    for body_part in body_parts:
        if body_part != 'pelvis':
            rx_, ry_, rz_ = obs['body_pos_rot'][body_part]
            res += [rx_-rx, ry_-ry, rz_-rz]
    # 33 components -- linear velocities of body parts
    for body_part in body_parts:
        res += obs['body_vel_rot'][body_part]
    # 33 components -- linear accelerations of body parts
    for body_part in body_parts:
        res += obs['body_acc_rot'][body_part]

    # joints
    for joint_val in ['joint_pos', 'joint_vel', 'joint_acc']:
        for joint in ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r',
                      'hip_l', 'knee_l', 'ankle_l']:
            res += obs[joint_val][joint][:3]

    # muscles
    muscles = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r',
       'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r',
       'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l',
       'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l',
       'gastroc_l', 'soleus_l', 'tib_ant_l']
    for muscle in muscles:
        res += [obs['muscles'][muscle]['activation']]
        res += [obs['muscles'][muscle]['fiber_length']]
        res += [obs['muscles'][muscle]['fiber_velocity']]
    for muscle in muscles:
        res += [obs['muscles'][muscle]['fiber_force']*force_mult]
    forces = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r',
      'glut_max_r', 'iliopsoas_r', 'rect_fem_r', 'vasti_r',
      'abd_l', 'add_l', 'hamstrings_l', 'bifemsh_l',
      'glut_max_l', 'iliopsoas_l', 'rect_fem_l', 'vasti_l',
      'gastroc_l', 'soleus_l', 'tib_ant_l', 'ankleSpring',
      'pros_foot_r_0', 'foot_l', 'HipLimit_r', 'HipLimit_l',
      'KneeLimit_r', 'KneeLimit_l', 'AnkleLimit_r', 'AnkleLimit_l',
      'HipAddLimit_r', 'HipAddLimit_l',]

    # forces
    for force in forces:
        f = obs['forces'][force]
        if len(f) == 1: res+= [f[0]*force_mult]

    res = (np.array(res) - obs_means) / obs_stds

    return res.tolist()


def get_simbody_state(state_desc):
    res = []
    # joints
    for joint_val in ["joint_pos", "joint_vel"]:
        for joint in ["ground_pelvis", "hip_r", "hip_l", "back",
                      "knee_r", "knee_l", "ankle_r", "ankle_l"]:
            res += state_desc[joint_val][joint]
    # muscles
    muscles = ["abd_r", "add_r", "hamstrings_r", "bifemsh_r",
               "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r",
               "abd_l", "add_l", "hamstrings_l", "bifemsh_l",
               "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l",
               "gastroc_l", "soleus_l", "tib_ant_l"]
    for muscle in muscles:
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
    return res


def preprocess_obs_mini(state_desc):
    if isinstance(state_desc, dict):
        # if input is state_desc dictionary
        state = get_simbody_state(state_desc)
    else:
        # if input is simbody state of 72 values
        state = state_desc
    prep_obs = [state[4]] + state[:3] + state[6:]
    return prep_obs


def preprocess_obs_mini_cosine(state_desc):
    prep_obs = preprocess_obs_mini(state_desc)
    cosines = np.cos(prep_obs[1:15]).tolist()
    sines = np.sin(prep_obs[1:15]).tolist()
    prep_obs = [prep_obs[0]] + cosines + sines + prep_obs[15:]
    res = (np.array(prep_obs) - obs_mini_means) / obs_mini_stds
    return res.tolist()
