import numpy as np
from sklearn.neighbors import NearestNeighbors

class Policy():
    def __init__(self, param_dic, n_arms):
        """文字列からその方策のエージェントを作るためのクラス

        param_dic : 各方策に必要なパラメータを持った辞書
        n_arms : 腕の数（たぶん）
        """
        # print(param_dic)
        self.set_policy(param_dic, n_arms)

    def set_policy(self, param_dic, n_arms):
        """方策ごとに初期値をセット"""
        # superの部分 共通ってこと
        self.n_arms = n_arms
        self.n_features = param_dic['n_features']
        self.warmup = param_dic['warmup']
        self.batch_size = param_dic['batch_size']
        self.steps = 0
        self.counts = np.zeros(self.n_arms) #　各腕が選択された回数
        # superの部分ここまで
        self.reward_sum = 0
        self.count = 0
        self.arm_counts = np.zeros(self.n_arms)
        self.arm_rewards = np.zeros(self.n_arms)

        print(param_dic['policy_dic']['policy'])
        if param_dic['policy_dic']['policy']=='Greedy':
            self.policy = self.greedy
            self.update = self.greedy_update

            self.name = 'Greedy'

            self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
            self.b = np.zeros((self.n_arms, self.n_features))  # a*f
            self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
            self._b = np.zeros((self.n_arms, self.n_features))

            self.theta_hat = np.zeros((self.n_arms, self.n_features))
            self.theta_hat_x = np.zeros(self.n_arms)

            # greedy特有のパラメータ
            self.n_steps = param_dic['policy_dic']['n_steps']
        
        elif param_dic['policy_dic']['policy']=='Regional_LinRS':
            self.policy = self.regional_lin_rs
            self.update = self.regional_lin_rs_update

            self.aleph = param_dic['policy_dic']['aleph']
            self.k = param_dic['policy_dic']['k']
            self.episodic_memory = []
            self.memory_capacity = param_dic['policy_dic']['memory_capacity']
            self.zeta = param_dic['policy_dic']['zeta']
            self.epsilon = param_dic['policy_dic']['epsilon']
            self.stable_flag = param_dic['policy_dic']['stable_flag']
            self.w = param_dic['policy_dic']['w']
            print(self.w)
            
            #self.name = 'Regional LinRS(stable) ℵ={}'.format(self.aleph)
            self.name = 'Regional LinRS'
            
            self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
            self.b = np.zeros((self.n_arms, self.n_features))  # a*f
            self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
            self._b = np.zeros((self.n_arms, self.n_features))

            self.theta_hat = np.zeros((self.n_arms, self.n_features))
            self.theta_hat_x = np.zeros(self.n_arms)
            self.rs = np.zeros(self.n_arms)
            self.n = np.zeros(self.n_arms)
        else:
            print("Policy Name ERROR!!!")
            exit(0)

    def greedy(self, x):
        """腕の中から1つ選択肢し、インデックスを返す.

        Args:
            x(int, float):特徴量
            step(int):現在のstep数
        Retuens:
            result(int):選んだ行動
        """

        if True in (self.counts < self.warmup):
            # print("warmup = ", self.warmup)
            # warmup中の処理
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f,[2,117]
            # print("self.theta_hat = ", self.theta_hat)
            self.theta_hat_x = self.theta_hat @ x
            # print("self.theta_hat_x = ", self.theta_hat_x)

            result = np.argmax(self.theta_hat_x)

        return result

    def greedy_update(self, x, chosen_arm, reward):
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        # superの分
        self.steps += 1
        self.counts[chosen_arm] += 1
        # superの分終わり
        self.reward_sum += reward
        self.count += 1
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward

        # print(x)
        x = np.expand_dims(x, axis=1) # xを縦ベクトルに変換
        # print(x)
        """パラメータの更新"""
        # print(self.A_inv[chosen_arm])
        # print(x)
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)

        # print("self.A_inv[chosen_arm]")
        # print(self.A_inv[chosen_arm])

        self.b[chosen_arm] += np.ravel(x) * reward
        # print("self.b[chosen_arm]")
        # print(self.b[chosen_arm])

        #更新
        if self.steps % self.batch_size == 0:
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)

    def regional_lin_rs(self, x):
        """腕の中から1つ選択肢し、インデックスを返す.

        Args:
            x(int, float):特徴量
            step(int):現在のstep数
        Retuens:
            result(int):選んだ行動
        """
        # print("self.counts = ", self.counts)
        if True in (self.counts < self.warmup):
            # print("TRUE")
            result = np.argmax(np.array(self.counts < self.warmup))
        else:
            # print("FLASE")
            """報酬期待値の不偏推定量を計算"""
            # 全ての腕に対してtheta_hatを計算
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f,[2,117]
            self.theta_hat_x = self.theta_hat @ x
            # print("self.theta_hat = ", self.theta_hat)
            # print("self.theta_hat_x = ", self.theta_hat_x)

            """疑似試行割合の計算"""
            # 入力に対するk近傍法(ユークリッド距離)を計算しリストN_kに格納 (episodicメモリーに対してk-近傍法を行う)

            #episodic memory から中身を抽出
            context = np.array([r for r in self.episodic_memory])
            # print("context")
            # print(context)
            #print(context.dtype)
            
            #print(len(x))

            # 中身を特徴とaction vectorに分離
            context_x = context[:,:len(x)]
            action_vec = context[:,len(x):]

            # print("len(x)")
            # print(len(x))
            # print("context_x")
            # print(context_x)
            # print("action_vec")
            # print(action_vec)

            # 次元を増やしてepisodic memory の特徴群に合わせる
            x = np.expand_dims(x, axis = 0)
            # print(x)

            #episodic_memoryの特徴群に対してk近傍法のモデルを適用
            if len(context_x) <= self.k:
                k_num = len(context_x)
            elif len(context_x) > self.k:
                k_num = self.k

            nbrs = NearestNeighbors(n_neighbors = k_num, algorithm='brute', metric='euclidean').fit(context_x)
            distance, indices = nbrs.kneighbors(x) #現状態xと特徴群の距離を算出

            # print("indices = ", indices)#ラベル
            # print("distance = ", distance)#距離

            # 次元が多いので削除して計算しやすいようにする
            distance = np.squeeze(distance)
            action_vec = action_vec[indices]
            action_vec = np.squeeze(action_vec)

            # print("indices = ", indices)#ラベル
            # print("distance = ", distance)#距離

            # カーネルの計算の準備
            ## 平方ユークリッド距離(ユークリッドの2乗)を計算しd_kに格納
            d_k = np.asarray(distance) ** 2
            # print("d_k = ")
            # print(d_k)
            ## d_kを使ってユークリッド距離の移動平均d^2_mを計算
            d_m_ave = np.average(d_k)
            # print("d_m_ave = ", d_m_ave)
            ## カーネル値の分母の分数を計算(d_kの正則化)
            #d_n = d_k / d_m_ave #ここでときどきinvalid value encountered in true_divideが起きる→0除算によりNanが生まれるため発生、代わりに0を置き換えるようにする
            d_n = np.divide(d_k, d_m_ave, out=np.zeros_like(d_k), where = d_m_ave!=0)
            # print("d_n")
            # print(d_n)
            ## d_nをクラスタリング(具体的にはあまりに小さい場合0に更新)
            d_n -= self.zeta #d_k-zetaをして、
            d_n = [i if i > 0 else 0 for i in d_n] # マイナスは0に置き換え
            # print("d_n")
            # print(d_n)
            ## 入力と近傍値のカーネル値(類似度)K_vを計算
            K_v = [self.epsilon / (i + self.epsilon) for i in d_n]
            # print("K_v")
            # print(K_v)
            # 類似度K_vから総和が1となる重み生成。疑似試行回数 n の総和を1にしたいため
            sum_K = np.sum(K_v)
            # print("sum_k = ", sum_K)
            weight = [K_i/sum_K for K_i in K_v]
            # print("weight")
            # print(weight)
            #類似度から算出した重みと action vector で加重平均を行い疑似試行割合を計算
            self.n = np.average(action_vec, weights = weight, axis = 0) #疑似試行割合ファイのこと
            
            if self.stable_flag:
                base = self.counts / self.steps # これが論文のrho
                # print("self.counts = ", self.counts)
                # print("self.steps = ", self.steps)
                # print("base = ", base)
                self.n = base * (1-self.w) + self.n * self.w # phi

            self.rs = self.n *(self.theta_hat_x - self.aleph)  # a*1,[2]　#価値観数の計算

            result = np.argmax(self.rs)

        return result

    def regional_lin_rs_update(self, x, chosen_arm, reward):
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        # print("選んだ腕　chosen arm = ", chosen_arm)
        # superの分
        self.steps += 1
        self.counts[chosen_arm] += 1
        # superの分終わり
        self.reward_sum += reward
        self.count += 1
        self.arm_counts[chosen_arm] += 1
        self.arm_rewards[chosen_arm] += reward

        x = np.expand_dims(x, axis=1)
        # print("x = ") # 117行1列
        # print(x)
        # ウッドベリーの公式を使いたいがためにxを一列にしている

        # print("A[chosen_arm]更新前")
        # print(self.A_inv[chosen_arm])
        """パラメータの更新"""
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
        # ウッドベリーの公式 bはd次元ベクトル
        # (A + bb.T)^(-1) = (A^(-1) - A^(-1)bb.TA^(-1)/(1+b.TA(-1)b))
        # print("A[chosen_arm]更新後")
        # print(self.A_inv[chosen_arm])

        # print("b[chosen_arm]更新前")
        # print(self.b[chosen_arm])
        self.b[chosen_arm] += np.ravel(x) * reward
        # print("b[chosen_arm]更新後")
        # print(self.b[chosen_arm])
        # B = 特徴量*報酬

        # print("ravel(x)")
        # print(np.ravel(x))

        # エピソードメモリに現特徴量を格納 (FIFO形式)
        self.T = np.zeros((self.n_arms))
        self.T[chosen_arm] = 1 # 選択した腕に対してのみ1を加える
        # print(chosen_arm, "を選んだから、")
        # print(self.T)
        x = np.ravel(x)
        memory = np.append(x,self.T,axis=0)
        # print("memoryに合う形にすると")
        # print(memory)
        # print("これをエピソードメモリに追加すると")
        self.episodic_memory.append(memory) # appendできていない！謎のまま
        # print(self.episodic_memory)
        # print(len(self.episodic_memory))
        # エピソードメモリの容量を超えてるかチェック
        if len(self.episodic_memory) > self.memory_capacity:
            # print("メモリより大きいので、一番古いメモリを一個消す")
            self.episodic_memory.pop(0)
        
        #更新
        if self.steps % self.batch_size == 0:
            
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)

        # print("self._A_inv = ", self._A_inv)
        # print("self._b = ", self._b)
