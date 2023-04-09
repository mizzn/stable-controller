from policy.policy import Policy
import numpy as np

class Agent():
    """どんなポリシーでもこのエージェントとして機能するようにしたい"""
    def __init__(self, param_dic):
        self.param_dic = param_dic
        self.n_arms = param_dic['n_arms'] #冗長？

        self.policy = Policy(param_dic, self.n_arms)

    def initialize(self):
        self.policy = Policy(self.param_dic, self.n_arms)

    def choose_arm(self, x):
        return self.policy.policy(x)

    def update(self, x, chosen_arm, reward):
        self.policy.update(x, chosen_arm, reward)

    def get_theta(self):
        pass

    def get_theta_x(self):
        return self.policy.theta_hat_x

    def get_entropy_arm(self):
        pass

    def get_agent_name(self):
        return self.policy.name

    def get_reward_sum(self):
        return self.policy.reward_sum

    def get_count(self):
        return self.policy.count

    def get_arm_rewards(self):
        return self.policy.arm_rewards

    def change_param(self, chosen_prm):
        pass