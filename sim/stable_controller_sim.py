import numpy as np
import copy as cp
from policy_dic import AGENT
from policy_dic import HIGHER_AGENT
from stable_controller.agent import Agent
from stable_controller.higher_agent import HigherAgent

class StableControllerSimulator(object):
    """下位エージェントの重みをコントロールするアルゴリズムとなるクラス

    Attributes:
        p : 各腕の確率
        n_arms(int) : アーム数(行動数)
        agent(str) : エージェント
        higher_agent(str) : 上位エージェント
        n_features (int) : 特徴量の次元
        L(int) : 変化を検知してからどっちを使うか決めるまで
        delta : 
        lmd : 閾値
    """

    # 引数のpとか要る？？？？？？ 必要なさそうなので削除
    def __init__(self, n_arms, agent, higher_agent, n_features, L=30, delta=0, lmd=30): # 2つの方策が必要
        """ クラスの初期化 """
        self.n_arms = n_arms
        self.n_features = n_features
        self.counts = np.zeros(self.n_arms) #　各腕が選択された回数

        # self.alert = 0
        self.step = 0

        # ここまでは文字列
        self.agent_str = agent
        self.higher_agent_str = higher_agent

        # 文字列から上位エージェントをインスタンス化
        self.higher_agt_dic = HIGHER_AGENT[self.higher_agent_str]
        self.higher_agent = HigherAgent(self.higher_agt_dic)

        # 文字列から下位エージェントをインスタンス化
        self.agt_dic = AGENT[self.agent_str]
        self.agent = Agent(self.agt_dic) # ここだとパラメータはまだ初期値

        # new or old
        self.choose_agent = 0


    def choose_arm(self, x):
        """
            特徴量を受け取って、選ばれた腕とtheta_hatを返す関数
            Arg:
                x : 特徴量
            return:
                chosen_arm (int) : 選ばれた腕
                theta_hat : 近似の結果
        """
        # print("\n\n\n")
        # print("stable controller/choose_arm")

        self.chosen_prm_idx = self.higher_agent.choose_arm() # 三択なら変える必要がある
        # print("上位エージェントの選択は", self.chosen_prm_idx)

        # print("下位エージェントの重みを変える")
        self.agent.change_prm(self.chosen_prm_idx)

        chosen_arm = self.agent.choose_arm(x)
        # print("下位エージェントが選択した腕 = ", chosen_arm)
        theta_x = self.agent.get_theta_x()
        # print("返されるtheta_xの値 = ", theta_x)
        

        # print("下位エージェントのパラメータが変わっているか確認")
        # self.agent.print_prm()
        # print("上位エージェントのパラメータの確認")
        # self.higher_agent.print_prm()

        return  chosen_arm, theta_x


    def update(self, x, chosen_arm, reward): # 特徴量追加
        """パラメータ更新、target生成
        Args:
            x : 特徴量
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        # print("\n\n\n")
        # print("stable controller/update")

        # print("stable_controller/updateで受け取ったreward = ", reward)
        # print("stable_controller/updateで受け取ったchosen_arm = ", chosen_arm)
        # print("self.chosen_prm_idx = ", self.chosen_prm_idx)
        # print("上位エージェントと下位エージェントのアップデート")
        self.higher_agent.update(self.chosen_prm_idx, reward)
        self.agent.update(x, chosen_arm, reward)

        # print("下位エージェントのパラメータのupdate確認")
        # self.agent.print_prm()
        # print("上位エージェントのパラメータのupdate確認")
        # self.higher_agent.print_prm()
        

    # meta-bandit/environment.pyより
    # 必要そうだったので持ってきた
    def greedy(self, values):
        # print("greedy")
        # print("values = ", values)
        # print("values.max() = ", values.max)
        max_values = np.where(values ==values.max())[0]
        return np.random.choice(max_values)

    # 名前を返す関数
    def get_agent_name(self):
        return self.agent_str
        
    def get_chosen_prm_idx(self):
        return self.chosen_prm_idx

    def get_entropy_arm(self):
        # 下位エージェントのエントロピーを返す
        return self.agent.get_entropy_arm()
    