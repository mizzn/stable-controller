from stable_controller.higher_policy import HigherPolicy
import stable_param
import numpy as np

class HigherAgent():
    """どんなポリシーでもこのエージェントとして機能するようにしたい"""
    def __init__(self, param_dic):
        # print("stable_controller/HigherAgent/__init__")
        self.param_dic = param_dic
        # print("self.param_dic = ", self.param_dic)
        # exit()

        '''if ((self.param_dic['policy_dic']['policy'] == "RS") | (self.param_dic['policy_dic']['policy'] == "SRS")):
            # 上位がRSのとき、下位はregional linRS
            self.n_arms = len(stable_param.STABLE_W_LIST)
        elif self.param_dic['policy_dic']['policy'] == "UCB":
            # 上位がUCBのとき、下位はLinUCB
            self.n_arms = len(stable_param.ALPHA_LIST)
        elif self.param_dic['policy_dic']['policy'] == "TS":
            # 上位がTSのとき、下位はLinTS
            self.n_arms = len(stable_param.AB_LIST)
        else:
            print("stable_controller/higher_agent.py/__init__ param_dic ERROR!")
            exit(0)'''

        # if self.param_dic['policy_name'] == "RS":
        #     # 上位がRSのとき、下位はregional linRS
        #     self.n_arms = len(stable_param.STABLE_W_LIST)
        # elif self.param_dic['policy_name'] == "UCB":
        #     # 上位がUCBのとき、下位はLinUCB
        #     self.n_arms = len(stable_param.ALPHA_LIST)
        # elif self.param_dic['policy_name'] == "TS":
        #     # 上位がTSのとき、下位はLinTS
        #     self.n_arms = len(stable_param.AB_LIST)
        # else:
        #     print("stable_controller/higher_agent.py/__init__ param_dic ERROR!")
        #     exit(0)

        # print("上位エージェントの行動数 self.n_arms = ", self.n_arms) #確認OK
        # exit(0)

        self.policy = HigherPolicy(self.param_dic)

    def initialize(self):
        self.policy = HigherPolicy(self.param_dic)

    def choose_arm(self):
        # print("stable_controller/higher_agent.py choose_arm")
        return self.policy.policy()

    def update(self, chosen_arm, reward):
        # print("stable_controller/higher_agent.py update")
        self.policy.update(chosen_arm, reward)

    # def get_agent_name(self):
    #     return self.policy.name

    def get_reward_sum(self):
        return self.policy.reward_sum

    # def get_count(self):
    #     return self.policy.count

    def get_arm_rewards(self):
        return self.policy.arm_rewards
    
    def print_prm(self):
        # print("stable controller/higher_agent.py print_prm")
        return self.policy.print_prm()