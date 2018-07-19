# IMPORTANT
Start carefully recording all results and conclusions.

# nips2018_prosthetics_challenge
Repository for NIPS 2018 prosthetics challenge ([CrowdAI](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge), [GitHub](https://github.com/stanfordnmbl/osim-rl)).

# Work progress

### Small but important things
1. Handy tool for storing and visualizing agents' performance (not good and handy yet)
2. ~~Simple baselines to evaluate changes made in the code (pendulum and lunar lander)~~
3. ~~Normalize input observations based on magnitude statistics~~
4. Make SAC work in Prosthetics environment (maximum reward is 600 now)

### Global TODO list
1. Efficient distributed prioritized experience replay buffer
2. ~~Learning with n-step returns~~
3. ~~Implement other algorithms (SoftAC, Distributional Critic)~~
4. Implement and test different exploration techniques (Ornstein-Uhlenbeck, ~~gradient~~, curiosity, SIL)
5. ~~Test whether Layer normalization or Batch normalization improve training~~ (Layer normalization is pretty ok, at least it does not do any harm)
6. Save some episodes to disk and pretrain new policies (and critics) on them

### Ideas to try
1. ~~Shift positions with respect to pelvis or center mass~~
2. ~~Penalize for dying (probably not a good idea)~~ (not a good idea)
3. ~~Smart reward shaping (e.g. give some reward for bending a knee or doing a step forward)~~ (good idea, without reward bending bonus, agent does not bend its knees)
4. Take previous action into consideration through residual connection (lowest priority, weird idea)
5. Train gaussian policy with reward equal to the absolute speed (in any direction) and use it as a prior for training policy for moving in particular direction (or along designated speed vector)
6. Augment training data with rotations in (x,z) plane, especially important for round 2 of the competition
7. Ensembles of actors and critics
8. Implement and test Normalized Advantage Functions (NAF) algorithm
9. Implement and test Latent Space Hierarchial Policies

### Performance of different approaches (START RECORDING PERFORMANCE!!!)
```python
model='3D', prosthetic=True, difficulty=0, seed=25
```
| Approach | Experiment info | 5K episodes | 10K episodes | 15K episodes | 20K episodes |
|-|-|-|-|-|-|
| DDPG + PrioReplay | fs2, hl2, relu, [400,300] | 22.13 | 175.81 | 246.22 |
| DDPG + PrioReplay | fs2, hl2, ns4, relu, [400,300] | 86.71 | 107.08 | 175.61 | **260.77** |

# Hacks and hyperparameters from the literature

### DDPG
1. L2 weight decay (L2 regularization) of **0.01** for critic network.
2. Actions were not included until the **second hidden layer** of critic network.
3. The final layer weights and biases of both the actor and critic were initialized from **Uniform[-3e-3, 3e-3]** to ensure the initial outputs for the policy and value etimates were near zero.
4. Ornstein-Uhlenbeck process with **theta=0.15** and **sigma=0.2** for exploration.
5. Soft target updates with smoothing coefficient **tau=0.001**.
### NAF
1. Angles were often converted to sine and cosine encoding.
2. Actor and critic parameters are updated **5** times per each step of experience.
3. Batch normalization is employed as in the original DDPG paper.
### Ape-X DPG
1. The gradient used to update the **actor network** is clipped to [-1,1] element-wise.
2. **Hard** target updates every 100 training batches.
3. Prioritization parameters: priority exponent (alpha) -- 0.6, importance sampling exponent (beta) -- 0.4.
### Soft Actor-Critic
1. Soft target updates with smoothing coefficient **tau=0.01**, also works with tau=1.
2. **4** gradients steps per time step is optimal for DDPG, SAC allows for higher values (up to 16).
3. For deterministic policy choose the action that maximizes the Q-function among the mixture component means.

# Resources
### Learning to Run challenge
1. Learning to Run challenge: Synthesizing physiologically accurate motion using deep reinforcement learning ([pdf](https://arxiv.org/pdf/1804.00198.pdf)).
2. Learning to Run challenge solutions: Adapting RL methods for neuromusculoskeletal environments ([pdf](https://arxiv.org/pdf/1804.00361.pdf)).
3. Optimizing Locomotion Controllers Using Biologically-Based Actuators and Objectives (energy expenditure) ([pdf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4523558/pdf/nihms641752.pdf)).
### Base Algorithms
1. Continuous Control with Deep Reinforcement Learning (DDPG) ([pdf](https://arxiv.org/pdf/1509.02971.pdf)).
2. A Distributional Perspective on Reinforcement Learning (C51) ([pdf](https://arxiv.org/pdf/1707.06887.pdf)).
3. Distributional Reinforcement Learning with Quantile Regression (QR-DQN) ([pdf](https://arxiv.org/pdf/1710.10044.pdf)).
4. Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor (SAC-GMM) ([pdf](https://arxiv.org/pdf/1801.01290.pdf)).
5. Continuous Deep Q-Learning with Model-based Acceleration (NAF) ([pdf](http://proceedings.mlr.press/v48/gu16.pdf)).
6. Latent Space Policies for Hierarchial Reinforcement Learning (SAC-LSP) ([pdf](https://arxiv.org/pdf/1804.02808.pdf)).
7. Addressing Function Approximation Error in Actor-Critic Methods (TD3) ([pdf](https://arxiv.org/pdf/1802.09477.pdf)).
8. Smoothed Action Value Functions for Learning Gaussian Policies (Smoothie) ([pdf](https://arxiv.org/pdf/1803.02348.pdf)).
### Distributed RL systems
1. Distributed Prioritized Experience Replay (Ape-X) ([pdf](https://arxiv.org/pdf/1803.00933.pdf)).
2. Distributed Distributional Deterministic Policy Gradients (D4PG) ([pdf](https://arxiv.org/pdf/1804.08617.pdf)).
3. IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures (IMPALA) ([pdf](https://arxiv.org/pdf/1802.01561.pdf)).
### Exploration
1. Self-Imitation Learning (SIL) ([pdf](https://arxiv.org/pdf/1806.05635.pdf)).
### Hyper parameters and convergence
1. Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control ([pdf](https://arxiv.org/pdf/1708.04133.pdf)).
2. Deep Reinforcement Learning that Matters ([pdf](https://arxiv.org/pdf/1709.06560.pdf)).
### Third party code
1. Ray RLlib: Scalable Reinforcement Learning. Ray RLlib is an RL execution toolkit built on the Ray distributed execution framework. RLlib implements a collection of distributed policy optimizers that make it easy to use a variety of training strategies with existing RL algorithms written in frameworks such as PyTorch and TensorFlow ([docs](http://ray.readthedocs.io/en/latest/rllib.html), [github](https://github.com/ray-project/ray/tree/master/python/ray/rllib), [paper](https://arxiv.org/pdf/1712.09381.pdf)).
