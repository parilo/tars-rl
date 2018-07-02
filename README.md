# nips2018_prosthetics_challenge
Repository for NIPS 2018 prosthetics challenge ([CrowdAI](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge), [GitHub](https://github.com/stanfordnmbl/osim-rl)).

# Work progress

### Small but important things
1. Handy tool for storing and visualizing agents' performance (not good and handy yet)
2. ~~Simple baselines to evaluate changes made in the code (pendulum and lunar lander)~~
3. ~~Beat score of **162.245** in Prosthetics environment~~
4. Normalize input observations based on magnitude statistics (done stupidly, rework)

### Global TODO list
1. ~~Efficient distributed prioritized experience replay buffer~~
2. ~~Learning with n-step returns~~
3. ~~Implement other algorithms (SoftAC, Distributional Critic)~~
4. Ensembles of actors and critics
5. Implement and test different exploration techniques

### Ideas to try
1. ~~Shift positions with respect to pelvis or center mass~~
2. Penalize for dying
3. Smart reward shaping (e.g. give some reward for bending a knee or doing a step forward)
4. Take previous action into consideration through residual connection
5. Train gaussian policy with reward equal to the absolute speed (in any direction) and use it as a prior for training policy for moving in particular direction (or along designated speed vector)


### Performance of different approaches
```python
model='3D', prosthetic=True, difficulty=0, seed=25
```
| Approach | Experiment info | 5K episodes | 10K episodes | 15K episodes | 20K episodes |
|-|-|-|-|-|-|
| DDPG + PrioReplay | fs2, hl2, relu, [400,300] | 22.13 | 175.81 | 246.22 |
| DDPG + PrioReplay | fs2, hl2, ns4, relu, [400,300] | 86.71 | 107.08 | 175.61 | **260.77** |

# Hacks and hyperparameters from the literature

### Ape-X DPG
1. The gradient used to update the **actor network** is clipped to [-1,1] element-wise.
2. **Hard** target updates every 100 training batches.
3. Prioritization parameters: priority exponent (alpha) -- 0.6, importance sampling exponent (beta) -- 0.4

# Resources
### Learning to Run challenge
1. Learning to Run challenge: Synthesizing physiologically accurate motion using deep reinforcement learning ([pdf](https://arxiv.org/pdf/1804.00198.pdf)).
2. Learning to Run challenge solutions: Adapting RL methods for neuromusculoskeletal environments ([pdf](https://arxiv.org/pdf/1804.00361.pdf)).
### Main papers
1. Distributed Prioritized Experience Replay (Ape-X) ([pdf](https://arxiv.org/pdf/1803.00933.pdf)).
2. Distributed Distributional Deterministic Policy Gradients (D4PG) ([pdf](https://arxiv.org/pdf/1804.08617.pdf)).
3. A Distributional Perspective on Reinforcement Learning (C51) ([pdf](https://arxiv.org/pdf/1707.06887.pdf)).
4. Distributional Reinforcement Learning with Quantile Regression (QR-DQN) ([pdf](https://arxiv.org/pdf/1710.10044.pdf)).
5. Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor (SAC) ([pdf](https://arxiv.org/pdf/1801.01290.pdf))
### Third party code
1. Ray RLlib: Scalable Reinforcement Learning. Ray RLlib is an RL execution toolkit built on the Ray distributed execution framework. RLlib implements a collection of distributed policy optimizers that make it easy to use a variety of training strategies with existing RL algorithms written in frameworks such as PyTorch and TensorFlow ([docs](http://ray.readthedocs.io/en/latest/rllib.html), [github](https://github.com/ray-project/ray/tree/master/python/ray/rllib), [paper](https://arxiv.org/pdf/1712.09381.pdf)).
