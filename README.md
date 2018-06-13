# nips2018_prosthetics_challenge
Repository for NIPS 2018 prosthetics challenge ([CrowdAI](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge), [GitHub](https://github.com/stanfordnmbl/osim-rl)).

# Resources
1. Learning to Run challenge: Synthesizing physiologically accurate motion using deep reinforcement learning ([pdf](https://arxiv.org/pdf/1804.00198.pdf))
2. Learning to Run challenge solutions: Adapting RL methods for neuromusculoskeletal environments ([pdf](https://arxiv.org/pdf/1804.00361.pdf))

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
