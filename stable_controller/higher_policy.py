import numpy as np

class HigherPolicy():
    def __init__(self, param_dic, n_arms):
        """文字列からその方策のエージェントを作るためのクラス

        param_dic : 各方策に必要なパラメータを持った辞書
        n_arms : 腕の数（たぶん）
        """
        # print(param_dic)
        # print("stable_controller/Higherpolicy/__init__") # 確認OK
        self.param_dic = param_dic
        self.set_policy(n_arms)

    def set_policy(self,n_arms):
        """方策ごとに初期値をセット"""
        # superの部分 共通ってこと
        self.n_arms = n_arms
        self.steps = 0 #総試行回数
        # superの部分ここまで
        self.reward_sum = 0
        self.arm_counts = np.zeros(self.n_arms) #腕ごと
        self.arm_rewards = np.zeros(self.n_arms)

        print("上位エージェントの方策 = ", self.param_dic['policy_dic']['policy'])
        if self.param_dic['policy_dic']['policy']=='RS':
            self.policy = self.rs
            self.update = self.rs_update

            self.E = np.zeros(self.n_arms)
            self.RS = np.zeros(self.n_arms)
            self.aleph = self.param_dic['policy_dic']['aleph']
            self.weight_flag = self.param_dic['policy_dic']['weight_flag']
            self.w = self.param_dic['policy_dic']['w']
            print("self.aleph = ", self.aleph)
            print("print RS prm")
            self.print_prm()

        elif self.param_dic['policy_dic']['policy']=='UCB':
            self.policy = self.ucb
            self.update = self.ucb_update
            self.E = np.zeros(self.n_arms)
            print("print UCB prm")
            self.print_prm()

        elif self.param_dic['policy_dic']['policy']=='TS':
            self.policy = self.ts
            self.update = self.ts_update
            self.E = np.zeros(self.n_arms)
            self.alpha = np.array([self.param_dic['policy_dic']['alpha']] * self.n_arms)
            self.beta = np.array([self.param_dic['policy_dic']['beta']] * self.n_arms)
            self.mu0 = np.array([self.param_dic['policy_dic']['mu0']] * self.n_arms)
            self.v0 = self.beta / (1 + self.alpha)
            print("print TS prm")
            self.print_prm()

        elif self.param_dic['policy_dic']['policy']=='SRS':
            self.policy = self.srs
            self.update = self.srs_update

            self.epsilon = self.param_dic['policy_dic']['epsilon']
            self.aleph = self.param_dic['policy_dic']['aleph']
            self.E = np.zeros(self.n_arms)
            self.rho = np.zeros(self.n_arms)
            self.b = np.zeros(self.n_arms)
            self.SRS = np.zeros(self.n_arms)
            self.pi = np.array([1/self.n_arms] * self.n_arms)

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

        if self.weight_flag == True:
            self.E[chosen_arm] += self.w * (reward - self.E[chosen_arm])
            # print("weighted average")
        else:
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




        

    def srs(self):
        if np.amax(self.E) >= self.aleph:
            fix_aleph = np.amax(self.E) + self.epsilon
            diff = fix_aleph - self.E
            if np.amin(diff) < 0:
                diff -= np.amin(diff)
            self.Z = 1 / np.sum(1 / diff)
            self.rho = self.Z / diff
            self.b = self.arm_counts/self.rho - self.steps + self.epsilon
            self.SRS = (self.steps + np.amax(self.b)) * self.rho - self.steps
            if np.amin(self.SRS) < 0: self.SRS -= np.amin(self.SRS)
            self.pi = self.SRS / np.sum(self.SRS)
        else:
            self.Z = 1 / np.sum(1 / (self.aleph - self.E))
            self.rho = self.Z / (self.aleph - self.E)
            self.b = self.arm_counts/self.rho - self.steps + self.epsilon
            self.SRS = (self.arm_counts + np.amax(self.b)) * self.rho - self.steps
            if np.amin(self.SRS) < 0: self.SRS -= np.amin(self.SRS)
            self.pi = self.SRS / np.sum(self.SRS)
        
        current_prob = np.random.rand()
        top = self.n_arms
        bottom = -1
        while (top - bottom > 1):
            mid = int(bottom + (top - bottom)/2)
            if current_prob < np.sum(self.pi[0:mid]): top = mid
            else: bottom = mid
        if mid == bottom: arm = mid
        else: arm = mid-1
        return arm

    def srs_update(self, chosen_arm, reward):
        self.arm_counts[chosen_arm] += 1
        self.steps += 1
        self.E[chosen_arm] += (reward - self.E[chosen_arm]) / self.arm_counts[chosen_arm]

    def print_prm(self):
        if self.param_dic['policy_dic']['policy']=='RS':
            print("E = ", self.E)
            print("RS = ", self.RS)
            print("aleph = ", self.aleph)
        elif self.param_dic['policy_dic']['policy']=='UCB':
            print("E = ", self.E)
        elif self.param_dic['policy_dic']['policy']=='TS':
            # print("E = ", self.E)
            print("alpha = ", self.alpha)
            print("beta = ", self.beta)
            print("mu0 = ", self.mu0)
            print("v0 = ", self.v0)
        else:
            print("stable_controller/higher_policy.py print_prm ERROR!")
            exit(0)
