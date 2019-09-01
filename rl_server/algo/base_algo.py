
class BaseAlgo:

    def __init__(
        self,
        state_shapes=None,
        action_size=None,
        actor_optim_schedule=None,
        critic_optim_schedule=None,
        training_schedule=None
    ):
        # public properties
        self.state_shapes = state_shapes
        self.action_size = action_size
        self.actor_optim_schedule = actor_optim_schedule
        self.critic_optim_schedule = critic_optim_schedule
        self.training_schedule = training_schedule

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
