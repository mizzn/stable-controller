from stable_controller.policy import Policy
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
        # print("stable_controller/agent.py choose_arm")
        return self.policy.policy(x)

    def update(self, x, chosen_arm, reward):
        # print("stable_controller/agent.py update")
        self.policy.update(x, chosen_arm, reward)

    def get_theta(self):
        pass

    def get_theta_x(self):
        # print("stable_controller/agent.py/get_theta_x theta_hat_x = ", self.policy.theta_hat_x)
        return self.policy.theta_hat_x

    def get_entropy_arm(self):
        pass

    def get_agent_name(self):
        return self.policy.name

    def get_reward_sum(self):
        return self.policy.reward_sum

    # def get_count(self):
    #     return self.policy.count

    def get_arm_rewards(self):
        return self.policy.arm_rewards

    def change_prm(self, chosen_prm_idx):
        self.policy.change_prm(chosen_prm_idx)

    # debug
    def print_prm(self):
        self.policy.print_prm()

    def get_entropy_arm(self):
        return self.policy.get_entropy_arm()