# nips2018_prosthetics_challenge
Repository for NIPS 2018 prosthetics challenge ([CrowdAI](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge), [GitHub](https://github.com/stanfordnmbl/osim-rl)).

# Work progress

### Small but important things
1. Handy tool for storing and visualizing agents' performance (not good and handy yet)
2. ~~Simple baselines to evaluate changes made in the code (pendulum and lunar lander)~~
3. Beat score of **162.245** in Prosthetics environment
4. Normalize input observations based on magnitude statistics

### Global TODO list
1. Efficient distributed prioritized experience replay buffer
2. Learning with n-step returns
3. Implement other algorithms (SoftAC, Distributional Critic)

### Ideas to try
1. Shift positions with respect to pelvis or center mass


# Resources
### Learning to Run challenge
1. Learning to Run challenge: Synthesizing physiologically accurate motion using deep reinforcement learning ([pdf](https://arxiv.org/pdf/1804.00198.pdf)).
2. Learning to Run challenge solutions: Adapting RL methods for neuromusculoskeletal environments ([pdf](https://arxiv.org/pdf/1804.00361.pdf)).
### Main papers
1. Distributed Prioritized Experience Replay (Ape-X) ([pdf](https://arxiv.org/pdf/1803.00933.pdf)).
2. Distributed Distributional Deterministic Policy Gradients (D4PG) ([pdf](https://arxiv.org/pdf/1804.08617.pdf)).
3. A Distributional Perspective on Reinforcement Learning (C51) ([pdf](https://arxiv.org/pdf/1707.06887.pdf)).
4. Distributional Reinforcement Learning with Quantile Regression (QR-DQN) ([pdf](https://arxiv.org/pdf/1710.10044.pdf)).
### Third party code
1. Ray RLlib: Scalable Reinforcement Learning. Ray RLlib is an RL execution toolkit built on the Ray distributed execution framework. RLlib implements a collection of distributed policy optimizers that make it easy to use a variety of training strategies with existing RL algorithms written in frameworks such as PyTorch and TensorFlow ([docs](http://ray.readthedocs.io/en/latest/rllib.html), [github](https://github.com/ray-project/ray/tree/master/python/ray/rllib), [paper](https://arxiv.org/pdf/1712.09381.pdf)).

# Description
Now client-server approach is used to be able to utilize many simulators during training
# Files description
Client code
- agent.py - agent code
- run_many_agents.py - edit file to specify number of agents and its types

Server code
- osim-rl-server.py - run this file to run the server, edit file to set parameters
- rl_server/osim_rl_mode_dense.py - NN description
- rl_server/algo/ddpg.py - DDPG algorithm main file
- rl_server/osim_rl_ddpg.py - adapter for DDPG
- server - folder with client-server api
