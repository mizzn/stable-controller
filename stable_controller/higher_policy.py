import numpy as np

class HigherPolicy():
    def __init__(self, param_dic):
        """文字列からその方策のエージェントを作るためのクラス

        param_dic : 各方策に必要なパラメータを持った辞書
        n_arms : 腕の数（たぶん）
        """
        # print(param_dic)
        # print("stable_controller/Higherpolicy/__init__") # 確認OK
        self.param_dic = param_dic
        # print("self.param_dic = ", self.param_dic)
        # exit()
        self.set_policy()

    def set_policy(self):
        """方策ごとに初期値をセット"""
        # superの部分 共通ってこと
        self.n_arms = self.param_dic['n_arms']
        self.steps = 0 #総試行回数
        # superの部分ここまで
        self.reward_sum = 0
        self.arm_counts = np.zeros(self.n_arms) #腕ごと
        self.arm_rewards = np.zeros(self.n_arms)

        print("上位エージェントの方策 = ", self.param_dic['policy_name'])
        if self.param_dic['policy_name']=='RS':
            self.policy = self.rs
            self.update = self.rs_update

            self.E = np.zeros(self.n_arms)
            self.RS = np.zeros(self.n_arms)
            self.aleph = self.param_dic['aleph']
            print("self.aleph = ", self.aleph)
            print("print RS prm")
            self.print_prm()
            # exit()

        elif self.param_dic['policy_name']=='UCB':
            self.policy = self.ucb
            self.update = self.ucb_update
            self.E = np.zeros(self.n_arms)
            print("print UCB prm")
            self.print_prm()

        elif self.param_dic['policy_name']=='TS':
            self.policy = self.ts
            self.update = self.ts_update
            self.E = np.zeros(self.n_arms)
            self.alpha = np.array([self.param_dic['alpha']] * self.n_arms)
            self.beta = np.array([self.param_dic['beta']] * self.n_arms)
            self.mu0 = np.array([self.param_dic['mu0']] * self.n_arms)
            self.v0 = self.beta / (1 + self.alpha)
            print("print TS prm")
            self.print_prm()

        else:
            print("stable_controller/higher_policy.py/__init__ Policy Name ERROR!!!")
            exit(0)
    
    def rs(self):
        return np.random.choice(np.where(self.RS == self.RS.max())[0])

    def rs_update(self, chosen_arm, reward):
        # print("RS UPDATE")
        # print("rs_update/reward = ", reward)
        # print("chosen_arm = ", chosen_arm)
        self.reward_sum += reward
        self.steps += 1
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward
        # print("reward_sum = ", self.reward_sum)
        # print("steps = ", self.steps)
        # print("arm_counts = ", self.arm_counts)
        # print("arm_rewards = ", self.arm_rewards)

        self.E[chosen_arm] += (reward - self.E[chosen_arm]) / self.arm_counts[chosen_arm] # 忘却率なし
        # print("average")
        # print("self.E = ", self.E)
        self.RS = (self.arm_counts/ self.steps)*(self.E - self.aleph)
        # for i in range(2):
            # print("(", self.arm_counts[i] , "/", self.count, ") * (", self.E[i], "-", self.aleph, ")")
        # print("self.RS = ", self.RS)

    def ucb(self):
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0.0: 
                return arm
        bonus = np.sqrt((np.log(self.steps)) / (2*self.arm_counts))
        ucb = self.E + bonus
        # print("bonus = ", bonus)
        # print("ucb = ", ucb)
        return np.random.choice(np.where(ucb == ucb.max())[0])

    def ucb_update(self, chosen_arm, reward):
        self.reward_sum += reward
        self.steps += 1 #総試行回数
        self.arm_counts[chosen_arm] += 1 # 腕ごとの回数
        self.arm_rewards[chosen_arm] += reward

        self.E[chosen_arm] += (reward - self.E[chosen_arm]) / self.arm_counts[chosen_arm]
        # print("ucb_update self.E = ", self.E)

    def ts(self):
        precision = [np.random.gamma(self.alpha[i], 1/self.beta[i]) for i in range(self.n_arms)]
        # print("precision = ", precision)
        for i in range(self.n_arms):
            if precision[i] == 0 or self.arm_counts[i] == 0:
                precision[i] = 0.001 
        
        estimated_variance = [1/precision[i] for i in range(self.n_arms)]
        # print("estimated variance = ", estimated_variance)
        sample = np.random.normal(self.mu0, np.sqrt(estimated_variance))
        # print("sample = ", sample)
        return np.random.choice(np.where(sample == sample.max())[0])

    def ts_update(self, chosen_arm, reward):
        self.steps += 1
        self.arm_rewards[chosen_arm] += reward
        self.reward_sum += reward

        n = 1 # 観測されたサンプル数
        v = self.arm_counts # 選択回数　更新前のを使う

        self.alpha[chosen_arm] = self.alpha[chosen_arm] + n/2     
        self.beta[chosen_arm] = self.beta[chosen_arm] + ((n*self.arm_counts[chosen_arm]/(self.arm_counts[chosen_arm] + n)) * (((reward - self.mu0[chosen_arm])**2)/2))

        self.v0 = self.beta / (self.alpha + 1)
        self.arm_counts[chosen_arm] += 1

        self.mu0[chosen_arm] = np.mean(self.arm_rewards[chosen_arm])

    def print_prm(self):
        if self.param_dic['policy_name']=='RS':
            print("E = ", self.E)
            print("RS = ", self.RS)
            print("aleph = ", self.aleph)
        elif self.param_dic['policy_name']=='UCB':
            print("E = ", self.E)
        elif self.param_dic['policy_name']=='TS':
            # print("E = ", self.E)
            print("alpha = ", self.alpha)
            print("beta = ", self.beta)
            print("mu0 = ", self.mu0)
            print("v0 = ", self.v0)
        else:
            print("stable_controller/higher_policy.py print_prm ERROR!")
            exit(0)
