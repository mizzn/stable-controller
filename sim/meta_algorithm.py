import numpy as np
import copy as cp
from policy_dic import AGENT
from policy_dic import HIGHER_AGENT
from policy.agent import Agent
from policy.higher_agent import HigherAgent

class MetaAlgorithm(object):
    """メタアルゴリズムとなるクラス

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
    def __init__(self, n_arms, agent, higher_agent, n_features, stable_w_list, L=30, delta=0, lmd=30): # 2つの方策が必要
        """ クラスの初期化 """
        self.n_arms = n_arms
        self.n_features = n_features
        self.counts = np.zeros(self.n_arms) #　各腕が選択された回数

        self.alert = 0
        self.step = 0

        self.stable_w_list = stable_w_list

        # ここまでは文字列
        self.agent = agent
        self.higher_agent = higher_agent
        # print("self.agent = ", self.agent)
        # print("type of self.agent = ", type(self.agent))


        # 下位エージェントをインスタンス化
        self.agt_dic_0 = cp.deepcopy(AGENT[self.agent])
        self.agt_dic_1 = cp.deepcopy(AGENT[self.agent])
        if self.agent == 'Regional_LinRS':
            self.agt_dic_0['policy_dic']['w'] = stable_w_list[0]
            self.agt_dic_1['policy_dic']['w'] = stable_w_list[1]

        # tmpみたいな 初期状態のままとっておく
        # self.copy_agent0 = Agent(self.agt_dic_0) #ex) greedyのところの辞書が入る
        # self.copy_agent1 = Agent(self.agt_dic_1)

        self.agent0 = Agent(self.agt_dic_0)
        self.agent1 = Agent(self.agt_dic_1)
        #print("下位エージェントに指定したクラスでcopy_agent, old_agent, new_agentをインスタンス化")


        # 上位エージェントをインスタンス化
        self.higher_agt_dic = HIGHER_AGENT[self.higher_agent]
        self.higher_agent = HigherAgent(self.higher_agt_dic)

        # tmp
        # self.copy_higher_agent = HigherAgent(self.higher_agt_dic)
        # print("上位エージェントに指定したクラスでcopy_higher_agentとhigher_agentを作成")

        # new or old
        self.choose_agent = 0
        # param
        # self.delta = 0
        # detection
        # self.lmd = lmd
        # self.L = L
        # self.l_count = 0
        # self.mt_sum = 0
        # self.MT = 0
        # print("各パラメータを初期化")

        # self.current_agent_flag = 0 # 0ならagent0, 1ならagent1

    def reset_params(self):
        self.alert = 0
        self.l_count = 0
        self.mt_sum = 0
        self.MT = 0

    def choose_arm(self, x):
        # print("MetaAlgorithm/choose_arm 開始")

        self.chosen_agent = self.higher_agent.choose_arm() # 三択なら変える必要がある
        # print("上位エージェントの選択は", self.chosen_agent)
        return  self.choose_current_agent(x, self.chosen_agent)

        '''環境変化検知あり
        if self.alert:
        # 1--> new,    0--> old
            self.chosen_agent = self.higher_agent.choose_arm() # 三択なら変える必要がある
            print("上位エージェントの選択は", self.chosen_agent)
            if self.chosen_agent ==0:
                # print("MetaAlgorithm/choose_arm 終了")
                print("agent0が選択")
                return  self.agent0.choose_arm(x), self.agent0.get_theta_x()
            else:
                # print("MetaAlgorithm/choose_arm 終了")
                print("agent1が選択")
                return  self.agent1.choose_arm(x), self.agent1.get_theta_x()
        else:
            print("アラートなし")
            print("現在のエージェントが選択")
            # print("MetaAlgorithm/choose_arm 終了")
            return  self.choose_current_agent(x, self.current_agent_flag)
        '''
    
    def choose_current_agent(self, x, chosen_agent):
        """今の下位エージェントのchoose_arm, get_thetaを呼び出して、選ばれた腕と特徴量を返す関数
        Args:
            x : 特徴量
            chosen_agent (int) : 現在のエージェント
        """
        if (chosen_agent==0):
            # print("agent0が選択")
            return self.agent0.choose_arm(x), self.agent0.get_theta_x()
        elif (chosen_agent==1):
            # print("agent1が選択")
            return  self.agent1.choose_arm(x), self.agent1.get_theta_x()
        else:
            print("Choose current agent ERROR!")
            exit(0)


    def update(self, x, chosen_arm, reward): # 特徴量追加
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        # self.step += 1　継承してるからたぶんいらない
        # print("meta_algorithm/update reward = ", reward)
        # print("update chosen agent = ", self.chosen_agent)
        self.higher_agent.update(self.chosen_agent, reward) # 引数あとで調整 今はgreedyはいってるけど上位には使わない　xはあとで抜くべきかも　削除済み
        self.update_current_agent(x, chosen_arm, reward, self.chosen_agent)

        # reward_sum, count = self.get_result_current_agent(self.chosen_agent)

        '''
        環境変化検知あり
        if self.alert==1: # True
            # アラートがなっているとき
            self.l_count+=1 # Lstepの間に旧/新規エージェントを使うか決める
            self.higher_agent.update(self.chosen_agent, reward) # 引数あとで調整 今はgreedyはいってるけど上位には使わない　xはあとで抜くべきかも　削除済み

            # 両方ともupdate
            self.agent0.update(x, chosen_arm, reward)
            self.agent1.update(x, chosen_arm, reward)
            # print("higher_agent, old_agent, new_agentをアップデート")

            if self.l_count == self.L:
                # Lstep目
                self.reset_params()
                # print("パラメータをリセット")

                arm_rewards = self.higher_agent.get_arm_rewards()
                if self.greedy(arm_rewards) == 1:
                    print("上位エージェントが選んだ新しい下位エージェントはagent1")
                    print("現在のエージェントは", self.current_agent_flag, "だから")
                    if (self.current_agent_flag!=1):
                        self.kill_current_agent(self.current_agent_flag)
                    self.current_agent_flag = 1
                    print("flagを", self.current_agent_flag, "に変更")
                    # self.old_agent = self.new_agent
                else:
                    print("上位エージェントの選択が選んだ新しい下位エージェントはagent0")
                    print("現在のエージェントは", self.current_agent_flag, "だから")
                    if (self.current_agent_flag!=0):
                        self.kill_current_agent(self.current_agent_flag)
                    self.current_agent_flag = 0
                    print("flagを", self.current_agent_flag, "に変更")
                    

        else: # False
            # アラートがなっていないとき
            self.update_current_agent(x, chosen_arm, reward, self.current_agent_flag)
            # print("現在の下位エージェントをアップデート")

            # self.count reward_sumはpolicyに追加必要 ←しました
            #　agentに受け取り用関数必要　
            # reward_sum = self.old_agent.get_reward_sum()
            # count = self.old_agent.get_count()
            reward_sum, count = self.get_result_current_agent(self.current_agent_flag)


            # average = self.old_agent.reward_sum / self.old_agent.count # 論文中(6) /r_tを計算 
            average = reward_sum / count
            mt = reward - average + self.delta # m_T計算用
            self.mt_sum += mt # 論文中(7) m_T

            if self.mt_sum > self.MT:
                self.MT = self.mt_sum # 論文中(8) 最大をMTに
            PHt = self.MT - self.mt_sum # 論文中(9) Page-Hinkley統計量

            # 超えたらアラーム: True
            if PHt > self.lmd: #lmdは基準値
                print("アラート！！！！！！！！！！！！！！！！！！！！！！！！！")
                self.alert = 1
                print("現在の下位エージェントは", self.current_agent_flag)
                # self.new_agent = cp.deepcopy(self.agent) 初期化済みだからいらないはず
                # print(self.agent0.get_arm_rewards()) # 初期化の確認
                # print(self.agent1.get_arm_rewards())
                print("上位エージェントを初期化")
                self.higher_agent = cp.deepcopy(self.copy_higher_agent)
                # print(self.higher_agent.get_reward_sum())'''

    def update_current_agent(self, x, chosen_arm, reward, chosen_agent):
        """今の下位エージェントのupdateをする関数
            Args:
                x : 特徴量
                chosen_arm : 選択された腕
                reward : 報酬
                chosen_agent : 現在のエージェント
        """
        if (chosen_agent==0):
            # print("agent0をアップデート")
            self.agent0.update(x, chosen_arm, reward)
        elif (chosen_agent==1):
            # print("agent1をアップデート")
            self.agent1.update(x, chosen_arm, reward)
        else:
            # print("update current agent ERROR!")
            exit(0)

    def get_result_current_agent(self, flag):
        """現在の下位エージェントからreward_sumとcountを受け取る関数
            Args:
            flag : 現在の下位エージェントのフラグ
        """
        if (flag==0):
            return self.agent0.get_reward_sum(), self.agent0.get_count()
        elif (flag==1):
            return self.agent1.get_reward_sum(), self.agent1.get_count()
        else:
            print("get result current agent ERROR!")
            exit(0)

    def kill_current_agent(self, flag):
        """現在の下位エージェントを初期化する関数
            Args:
                flag : 現在の下位エージェントのフラグ
        """
        if (flag==0):
            # print("エージェント", flag, "を初期化")
            self.agent0 = cp.deepcopy(self.copy_agent0)
        elif (flag==1):
            # print("エージェント", flag, "を初期化")
            self.agent1 = cp.deepcopy(self.copy_agent1)
        else:
            print("Kill agent ERROR!")
            exit(0)

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
        return self.agent
        
    def get_chosen_agent(self):
        return self.chosen_agent
    