import tensorflow as tf


def create_placeholders(state_shapes, action_size, scope="placeholders"):
    with tf.name_scope(scope):
        states_ph, next_states_ph = [], []
        for i, shape in enumerate(state_shapes):
            states_batch_shape = [None] + list(shape)
            states_ph.append(tf.placeholder(
                tf.float32, states_batch_shape, "states" + str(i) + "_ph"))
            next_states_ph.append(tf.placeholder(
                tf.float32, states_batch_shape, "next_states" + str(i) + "_ph"))
        actions_ph = tf.placeholder(
            tf.float32, [None, action_size], "actions_ph")
        rewards_ph = tf.placeholder(
            tf.float32, [None, ], "rewards_ph")
        dones_ph = tf.placeholder(
            tf.float32, [None, ], "dones_ph")

        actor_lr = tf.placeholder(tf.float32, (), "actor_lr")
        critic_lr = tf.placeholder(tf.float32, (), "critic_lr")

    return (actor_lr, critic_lr, states_ph, actions_ph, rewards_ph, next_states_ph, dones_ph)


# def create_placeholders_n_algos_with_split(state_shapes, action_size, num_algos, batch_size, scope="placeholders"):
#     # construct one big batch for optimal loading
#     big_batch_size = num_algos * batch_size
#     with tf.name_scope(scope):
#         states_ph, next_states_ph = [], []
#         for i, shape in enumerate(state_shapes):
#             states_batch_shape = [big_batch_size] + list(shape)
#             states_ph.append(tf.placeholder(
#                 tf.float32, states_batch_shape, "states"+str(i)+"_ph"))
#             next_states_ph.append(tf.placeholder(
#                 tf.float32, states_batch_shape, "next_states"+str(i)+"_ph"))
#         actions_ph = tf.placeholder(
#             tf.float32, [big_batch_size, action_size], "actions_ph")
#         rewards_ph = tf.placeholder(
#             tf.float32, [big_batch_size, ], "rewards_ph")
#         dones_ph = tf.placeholder(
#             tf.float32, [big_batch_size, ], "dones_ph")
#
#         # split big batch into individual batches
#         splited_states_ph = [tf.split(st_ph, num_algos, axis=0) for st_ph in states_ph]
#         splited_next_states_ph = [tf.split(st_ph, num_algos, axis=0) for st_ph in next_states_ph]
#         splited_actions_ph = tf.split(actions_ph, num_algos, axis=0)
#         splited_rewards_ph = tf.split(rewards_ph, num_algos, axis=0)
#         splited_dones_ph = tf.split(dones_ph, num_algos, axis=0)
#         splited_batches = list(zip(
#             *(
#                 splited_states_ph +
#                 splited_next_states_ph +
#                 [splited_actions_ph, splited_rewards_ph, splited_dones_ph]
#             )
#         ))
#         # group observations
#         num_obs = len(state_shapes)
#         for i, splited_batch in enumerate(splited_batches):
#             splited_batches[i] = [
#                 splited_batch[:num_obs],
#                 splited_batch[-3],
#                 splited_batch[-2],
#                 splited_batch[num_obs:2 * num_obs],
#                 splited_batch[-1]
#             ]
#
#     # big batch, then splitted batches
#     return (states_ph, actions_ph, rewards_ph, next_states_ph, dones_ph), splited_batches


# def create_placeholders_n_algos_random_sample(state_shapes, action_size, num_algos, big_batch_size, algo_batch_size, scope="placeholders"):
#     # construct one big batch for optimal loading
#     with tf.name_scope(scope):
#         states_ph, next_states_ph = [], []
#         for i, shape in enumerate(state_shapes):
#             states_batch_shape = [big_batch_size] + list(shape)
#             states_ph.append(tf.placeholder(
#                 tf.float32, states_batch_shape, "states"+str(i)+"_ph"))
#             next_states_ph.append(tf.placeholder(
#                 tf.float32, states_batch_shape, "next_states"+str(i)+"_ph"))
#         actions_ph = tf.placeholder(
#             tf.float32, [big_batch_size, action_size], "actions_ph")
#         rewards_ph = tf.placeholder(
#             tf.float32, [big_batch_size, ], "rewards_ph")
#         dones_ph = tf.placeholder(
#             tf.float32, [big_batch_size, ], "dones_ph")
#
#         # split big batch into individual batches
#         step = int(float(big_batch_size - algo_batch_size) / num_algos)
#         splited_states_ph = [[st_ph[i * step:algo_batch_size + i * step] for i in range(num_algos)] for st_ph in states_ph]
#         splited_next_states_ph = [[st_ph[i * step:algo_batch_size + i * step] for i in range(num_algos)] for st_ph in next_states_ph]
#         splited_actions_ph = [actions_ph[i * step:algo_batch_size + i * step] for i in range(num_algos)]
#         splited_rewards_ph = [rewards_ph[i * step:algo_batch_size + i * step] for i in range(num_algos)]
#         splited_dones_ph = [dones_ph[i * step:algo_batch_size + i * step] for i in range(num_algos)]
#         splited_batches = list(zip(
#             *(
#                 splited_states_ph +
#                 splited_next_states_ph +
#                 [splited_actions_ph, splited_rewards_ph, splited_dones_ph]
#             )
#         ))
#         # group observations
#         num_obs = len(state_shapes)
#         for i, splited_batch in enumerate(splited_batches):
#             splited_batches[i] = [
#                 splited_batch[:num_obs],
#                 splited_batch[-3],
#                 splited_batch[-2],
#                 splited_batch[num_obs:2 * num_obs],
#                 splited_batch[-1]
#             ]
#
#     # big batch, then splitted batches
#     return (states_ph, actions_ph, rewards_ph, next_states_ph, dones_ph), splited_batches


def target_network_update(target, source, tau):
    update_ops = tf.group(
        *(
            w_target.assign_sub(
                tau * (w_target - w_source)
            ) for w_source, w_target in zip(
            source.variables(),
            target.variables()
        )
        )
    )
    return update_ops


def network_update(
        loss,
        network,
        optimizer,
        grad_val_clip=None,
        grad_norm_clip=None
):
    gradients = optimizer.compute_gradients(
        loss,
        var_list=network.variables()
    )
    if grad_val_clip is not None:
        gradients = [
            (
                tf.clip_by_value(
                    grad,
                    -grad_val_clip,
                    grad_val_clip
                ),
                var
            ) for grad, var in gradients
        ]

    if grad_norm_clip is not None:
        gradients = [
            (
                tf.clip_by_norm(
                    grad,
                    grad_norm_clip
                ),
                var
            ) for grad, var in gradients
        ]
    update_op = optimizer.apply_gradients(gradients)
    return update_op


class BaseAlgo:

    def __init__(
        self,
        state_shapes,
        action_size,
        placeholders,
        actor_optim_schedule,
        critic_optim_schedule,
        training_schedule
    ):
        # public properties
        self.state_shapes = state_shapes
        self.action_size = action_size
        self.placeholders = placeholders
        self.actor_optim_schedule = actor_optim_schedule
        self.critic_optim_schedule = critic_optim_schedule
        self.training_schedule = training_schedule

    def create_placeholders(self):
        if self.placeholders is None:
            self.placeholders = create_placeholders(self.state_shapes, self.action_size)

        self.actor_lr_ph = self.placeholders[0]
        self.critic_lr_ph = self.placeholders[1]
        self.states_ph = self.placeholders[2]
        self.actions_ph = self.placeholders[3]
        self.rewards_ph = self.placeholders[4]
        self.next_states_ph = self.placeholders[5]
        self.dones_ph = self.placeholders[6]

    def init(self, sess):
        sess.run(tf.global_variables_initializer())

    def get_schedule_params(self, schedule, step_index):
        for training_params in schedule['schedule']:
            if step_index >= training_params['limit']:
                return training_params
        return schedule['schedule'][0]

    def get_batch_size(self, step_index):
        return self.get_schedule_params(self.training_schedule, step_index)['batch_size']

    def get_actor_lr(self, step_index):
        return self.get_schedule_params(self.actor_optim_schedule, step_index)['lr']

    def get_critic_lr(self, step_index):
        return self.get_schedule_params(self.critic_optim_schedule, step_index)['lr']
