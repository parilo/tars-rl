# Config file

"Keep all hyperparameters in one file" is main principle for config files.
So TARS-RL use one yaml config file for server and clients.
This is not a working config file but rather full options description.

```yaml
# for now only tensroflow is available
framework: tensorflow

# environment parameters
env:
  # you may specify gym env by name
  name: Pendulum-v0
  is_gym: True
  
  # or if you have custom env
  # see envs/gym_env.py or envs/l2run.py
  # as example of custom envs
  env_module: envs.l2run
  env_class: L2RunEnvWrapper
  is_gym: False
  
  # observation size
  obs_size: 3
  
  action_size: 1
  
  # how much steps should action be repeated for
  frame_skip: 2
  
  # maximum number of steps per episode
  # 0 - unlimited
  step_limit: 0
  
  # agent stores transitions and sends whole episode
  # when it ends. So keep agent buffer size
  # bigger than episode length 
  agent_buffer_size: 110
  
  # how much last observations
  # to use as input to algorithm
  history_length: 2
  
  # multiply env reward by this factor
  reward_scale: 0.1
  
  # keep random action the same during
  # specified number of steps
  reinit_random_action_every: 5
  
  # log episode mean reward once every
  # specified number env steps
  log_every_n_steps: 1000

  # example of action remapping
  remap_action:
    # lower limit
    low:
      before: -1.  # limits of your activation function
      after: 0.  # desired limit
    # upper limit
    high:
      before: 1.
      after: 1.

# server side parameters
server:
  # seed that will be used for model initialization
  seed: 42
  
  # maximum number of clients may connect in parallel
  num_clients: 16
  experience_replay_buffer_size: 5000000
  use_prioritized_buffer: false
  
  # train steps count will not exceed replay_buffer / train_every_nth   
  train_every_nth: 1.
  
  # start learning when replay buffer will contain more than
  start_learning_after: 5000
  
  # update target networks every specified number of train ops
  target_critic_update_period: 1
  target_actor_update_period: 1
  
  # show stats in terminal every specified number of train ops
  show_stats_period: 100
  
  # save model to logs dir every specified number of train ops
  save_model_period: 5000
  
  # every client will must comunicate through
  # its own port, so server will allocate
  # num_clients ports starting from specified
  # client port number will be
  # client_start_port + client_index
  client_start_port: 10977
  
  # dir to save logs, config file and model to
  logdir: "experiments/pendulum/logs/pendulum_qtd3_seed_42"
  
  # load model from file before start learning
  load_checkpoint: "experiments/pendulum/logs/pendulum_qtd3_seed_43/model-5000.ckpt"

# algorithm you want to use
# possible values:
# ddpg
# quantile_ddpg
# categorical_ddpg
# td3
# categorical_td3
# quantile_td3
# sac
algo_name: "td3"

# algorithm parameters
algorithm:
  # see what is multi-step q learning
  # 1 is good choice at start
  n_step: 1
  
  # discount factor
  gamma: 0.99
  
  # clip actor gradients byspecified  value
  actor_grad_val_clip: 1.0
  
  # update target network is
  # target_network = target_network * (1 - target_update_rate) +
  #                  target_update_rate * main_network 
  target_actor_update_rate: 0.0025
  target_critic_update_rate: 0.005

# actor network parameters
actor:
  # network have no lstm layers inside
  lstm_network: False
  # number of fully connected layers
  # and number of neurons in each layer
  hiddens: [128, 128, 128]
  # activation functions of layers
  activations: ["relu", "relu", "relu"]
  
  # network have lstm layers inside
  lstm_network: True
  # fully connected layers before lstm layers
  # will process each time step with same parameters
  # so you will have embeddings of each time steps
  embedding_layers: [128]
  embedding_activations: ["relu"]
  # lstm layers
  lstm_layers: [128]
  lstm_activations: ["relu"]
  # fully connected layers after lstm layers
  output_layers: [128]
  output_layers_activations: ["relu"]
  
  # layers normalization for all fully connected layers
  layer_norm: False
  
  # make all fully connected layers noisy
  noisy_layer: False
  
  # actor output activation function
  output_activation: "tanh"

# critic network description
critic:
  # the same as for actor
  lstm_network: False
  hiddens: [128, 128]
  activations: ["relu", "relu"]
  layer_norm: False
  noisy_layer: False
  output_activation: "linear"
  
  # insert action after specified number of layers
  # in case you want to mix in action
  # into later layer and process state
  # separately in some starting layers
  action_insert_block: 0
  
  # for ddpg and td3 is must be 1
  # for quantile and categorical versions
  # it is number of atoms used in distribution
  # 128 is good choice for them
  num_atoms: 1

# schedule of learning rate for actor 
actor_optim:
  schedule:
    - limit: 0  # of train ops
      lr: 0.001
    # if number of train ops > 500000 lr will be 0.0005
    - limit: 500000
      lr: 0.0005
    - limit: 1000000
      lr: 0.0005
    - limit: 1500000
      lr: 0.00025

# schedule of learning rate for critic 
critic_optim:
  schedule:
    - limit: 0  # of train ops
      lr: 0.001
    - limit: 500000
      lr: 0.0005
    - limit: 1000000
      lr: 0.0005
    - limit: 1500000
      lr: 0.00025

# scheduling of batch size
training:
  schedule:
    - limit: 0  # of train ops
      batch_size: 2560
    - limit: 500000
      batch_size: 2560
    - limit: 1000000
      batch_size: 5120
    - limit: 1500000
      batch_size: 5120

# parameters of agents that will be used during training
agents:
  # if you are using ensemble of algorithms
  # each algorithm has its own agents
  - algorithm_id: 0  # agents belonging to algo 0
    agents:
      - agents_count: 1  # how many agents to start with this settings
        visualize: True  # visualize policy

      - agents_count: 1
        visualize: False

      - agents_count: 1
        visualize: True
        # exploration parameters
        exploration:
          # normal noise added to action
          normal_noise: 0.5
          # probability to perform completely random action
          # (sampled from uniform distribution)
          random_action_prob: 0.2

      - agents_count: 4
        visualize: False
        exploration:
          normal_noise: 0.5
          random_action_prob: 0.2
```