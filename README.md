# TARS-RL
Distributed Reinforcement Learning Framework.

## Algorithms

- DDPG [pdf](https://arxiv.org/pdf/1509.02971.pdf)
- C51 (Categorical DDPG) [pdf](https://arxiv.org/pdf/1707.06887.pdf)
- QR-DQN (Quantile DDPG) [pdf](https://arxiv.org/pdf/1710.10044.pdf)
- Soft Actor-Critic (SAC) [pdf](https://arxiv.org/pdf/1801.01290.pdf)
- TD3 [pdf](https://arxiv.org/pdf/1802.09477.pdf)
- Quantile TD3
- Ensemble of algorithms (use same batch for training)

## Features

- Client-Server architecture (you don't need incorporate RL framework into your environment, just use client)
- Server collects experience and to training
- Arbitrary number of parallel agents (clients) can send gathered experience to server over network
- All hyperparameters in one file
- Different exploration parameters for every agent
- Easy to implement new algorithms
- Support any gym compatible environment out of the box
- Python 3.6

## Example envs

OpenAI Gym:
- Bipedal Walker both simple and hardcore [see](experiments/bipedal_walker)
- Lunar Lander [see](experiments/lunar_lander)
- Pendulum [see](experiments/pendulum)

Challanges:
- NeurIPS 2017: Learning To Run [see](experiments/l2run)
- NeurIPS 2018: AI for Prosthetics Challenge [see](experiments/prosthetics)

## Documentation

See [config file description](CONFIG_FILE.md)

## Installation

Step 0. Install anaconda with python 3.6 from [download page](https://www.anaconda.com/download/#linux) or see [archived versions](https://repo.anaconda.com/archive/) (Optional, but highly recommended)
```buildoutcfg
1. Clone repo
$ git clone

2. Add to PATH your anaconda
$ export PATH=/path/to/your/anaconda/bin/:$PATH

3. Install requirements
$ pip install tensorflow
or
$ pip install tensorflow-gpu
if you have supported by tensorflow GPU 

$ pip install tensorboardX

4. For OpenAI gym examples
$ pip install gym['box2d']

or see how to install all Gym envs
https://github.com/openai/gym

```

## How to run

```buildoutcfg
$ cd root/of/tars-rl

run server (Lunar Lander config as an example)
$ python -m rl_server.server.run_server --config experiments/lunar_lander/config_ddpg.yml

run agents
(7 parallel agents on your computer,
 supposed you have CPU with 8 threads)
$ CUDA_VISIBLE_DEVICES="" python -m rl_server.server.run_agents --config experiments/lunar_lander/config_ddpg.yml
 
CUDA_VISIBLE_DEVICES=""
is needed if you don't want agents
to not interrupt server train operations

run trained policy from a checkpoint without server
python -m rl_server.server.play --config path/to/config.yml --checkpoint path/to/model-10000.ckpt --seed 1234
```

## Credits

- Oleksii Hrinchuk, [e-mail](oleksii.hrinchuk@gmail.com), [github](https://github.com/AlexGrinch) 
- Anton Pechenko, [github](https://github.com/parilo), [linkedin](https://www.linkedin.com/in/antonpechenko), [youtube](https://www.youtube.com/c/AntonPechenko) 
- Sergey Kolesnikov, [github](https://github.com/Scitator)

## References
1. Continuous Control with Deep Reinforcement Learning (DDPG) ([pdf](https://arxiv.org/pdf/1509.02971.pdf)).
2. A Distributional Perspective on Reinforcement Learning (C51) ([pdf](https://arxiv.org/pdf/1707.06887.pdf)).
3. Distributional Reinforcement Learning with Quantile Regression (QR-DQN) ([pdf](https://arxiv.org/pdf/1710.10044.pdf)).
4. Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor (SAC-GMM) ([pdf](https://arxiv.org/pdf/1801.01290.pdf)).
7. Addressing Function Approximation Error in Actor-Critic Methods (TD3) ([pdf](https://arxiv.org/pdf/1802.09477.pdf)).

## Roadmap

3. Train envs, make videos, write docs
4. Release TARS-RL
5. Add HER
6. Support pytorch
7. Add self-play
