import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import stable_param
from scipy.stats import entropy
from scipy.stats import invgamma

class Policy():
    def __init__(self, param_dic):
        """文字列からその方策のエージェントを作るためのクラス

        param_dic : 各方策に必要なパラメータを持った辞書
        n_arms : 腕の数（たぶん）
        """
        self.param_dic = param_dic
        # print("self.param_dic = ", self.param_dic)
        self.set_policy()

    def set_policy(self):
        """方策ごとに初期値をセット"""
        # superの部分 共通ってこと
        self.n_arms = self.param_dic['n_arms']
        self.n_features = self.param_dic['n_features']
        self.warmup = self.param_dic['warmup']
        self.batch_size = self.param_dic['batch_size']
        self.steps = 0
        # self.counts = np.zeros(self.n_arms) #　各腕が選択された回数
        # superの部分ここまで
        self.reward_sum = 0
        self.arm_counts = np.zeros(self.n_arms) #腕ごとの回数
        self.arm_rewards = np.zeros(self.n_arms)

        self.A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # a*f*f
        self.b = np.zeros((self.n_arms, self.n_features))  # a*f
        self._A_inv = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self._b = np.zeros((self.n_arms, self.n_features))

        self.theta_hat = np.zeros((self.n_arms, self.n_features))
        self.theta_hat_x = np.zeros(self.n_arms)

        print("下位エージェントの方策 = ", self.param_dic['policy_name'])
        if self.param_dic['policy_name']=='Greedy':
            self.policy = self.greedy
            self.update = self.greedy_update

            self.name = 'Greedy'

            # greedy特有のパラメータ
            self.n_steps = self.param_dic['n_steps']
        
        elif self.param_dic['policy_name']=='Regional_LinRS':
            self.policy = self.regional_lin_rs
            self.update = self.regional_lin_rs_update

            self.aleph = self.param_dic['aleph']
            self.k = self.param_dic['k']
            self.episodic_memory = []
            self.memory_capacity = self.param_dic['memory_capacity']
            self.zeta = self.param_dic['zeta']
            self.epsilon = self.param_dic['epsilon']
            self.stable_flag = self.param_dic['stable_flag']
            self.w = self.param_dic['w']
            # print("インスタンス化時のself.w = ", self.w)
            
            #self.name = 'Regional LinRS(stable) ℵ={}'.format(self.aleph)
            self.name = 'Regional LinRS'

            self.rs = np.zeros(self.n_arms)
            self.n = np.zeros(self.n_arms)

            print("print regional linrs prm")
            self.print_prm()


        elif self.param_dic['policy_name']=='LinUCB':
            self.policy = self.lin_ucb
            self.update = self.lin_ucb_update

            self.alpha = self.param_dic['alpha']
            self.name = 'LinUCB'

            print("print LinUCB prm")
            self.print_prm()

        elif self.param_dic['policy_name'] == 'LinTS':
            self.policy = self.lin_ts
            self.update = self.lin_ts_update

            self._lambda_prior = self.param_dic['lambda_prior']
            self._a0 = self.param_dic['alpha']
            self._b0 = self.param_dic['beta']
            self.name = 'LinTS'

            self.mu = np.array([np.zeros(self.n_features + 1) for _ in range(self.n_arms)])
            self.cov = np.array([(1.0 / self._lambda_prior) * np.identity(self.n_features + 1) for _ in range(self.n_arms)])
            self.precision = np.array([self._lambda_prior * np.identity(self.n_features + 1) for _ in range(self.n_arms)])

            self.ig_a = np.array([self._a0 for _ in range(self.n_arms)])
            self.ig_b = np.array([self._b0 for _ in range(self.n_arms)])

            self.A = np.array([self._lambda_prior * np.identity(self.n_features + 1) for _ in range(self.n_arms)])
            self.b = np.zeros((self.n_arms, self.n_features + 1))
            self.c = np.zeros(self.n_arms)

            self._mu = np.array([np.zeros(self.n_features + 1) for _ in range(self.n_arms)])
            self._cov = np.array([(1.0 / self._lambda_prior) * np.identity(self.n_features + 1) for _ in range(self.n_arms)])
            self._precision = np.array([self._lambda_prior * np.identity(self.n_features + 1) for _ in range(self.n_arms)])

            self._ig_a = np.array([self._a0 for _ in range(self.n_arms)])
            self._ig_b = np.array([self._b0 for _ in range(self.n_arms)])

            self._A = np.array([self._lambda_prior * np.identity(self.n_features + 1) for _ in range(self.n_arms)])
            self._b = np.zeros((self.n_arms, self.n_features + 1))
            self._c = np.zeros(self.n_arms)
            self.flag = True

            print("print LinTS prm")
            self.print_prm()

        else:
            print("stable_controller/policy.py/__init__ Policy Name ERROR!!!")
            exit(0)

    def greedy(self, x):
        """腕の中から1つ選択肢し、インデックスを返す.

        Args:
            x(int, float):特徴量
            step(int):現在のstep数
        Retuens:
            result(int):選んだ行動
        """

        if True in (self.arm_counts < self.warmup):
            # print("warmup = ", self.warmup)
            # warmup中の処理
            result = np.argmax(np.array(self.arm_counts < self.warmup))
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
        self.arm_counts[chosen_arm] += 1
        self.reward_sum += reward
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
        # print("REGIONAL LIN RS CHOOSE ARM")
        # print("self.arm_counts = ", self.arm_counts)
        if True in (self.arm_counts < self.warmup):
            # print("TRUE")
            result = np.argmax(np.array(self.arm_counts < self.warmup))
        else:
            # print("FLASE")
            """報酬期待値の不偏推定量を計算"""
            # 全ての腕に対してtheta_hatを計算
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f,[2,117]
            self.theta_hat_x = self.theta_hat @ x
            # print("stable_controller/policy.py/regional LinRS self.theta_hat_x = ", self.theta_hat_x)

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
            self.n = np.average(action_vec, weights = weight, axis = 0) #疑似試行割合
            
            if self.stable_flag:
                base = self.arm_counts / self.steps # これが論文のrho
                # print("self.arm_counts = ", self.arm_counts)
                # print("self.steps = ", self.steps)
                # print("base = ", base)
                # print("ベースラインいりphi")
                self.phi = base * (1-self.w) + self.n * self.w # phi
            else:
                # print("ただのphi")
                self.phi = self.n

            self.rs = self.phi *(self.theta_hat_x - self.aleph)  # a*1,[2]　#価値観数の計算

            result = np.argmax(self.rs)

        return result

    def regional_lin_rs_update(self, x, chosen_arm, reward):
        """パラメータ更新、target生成
        Args:
            chosen_arm(int):引いた腕
            reward(int, float):chosen_armを引いた結果得られた報酬
        """
        # print("REGIONAL LIN RS UPDATE")
        # print("regional lin rs update self.w = ", self.w)
        # print("選んだ腕　chosen arm = ", chosen_arm)
        # superの分
        self.steps += 1
        self.arm_counts[chosen_arm] += 1
        # superの分終わり
        self.reward_sum += reward
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

    def lin_ucb(self, x):
        # print("x.T = ", x.T)
        """腕の中から1つ選択肢し、インデックスを返す"""
        if True in (self.arm_counts < self.warmup):
            result = np.argmax(np.array(self.arm_counts < self.warmup))
        else:
            self.theta_hat = np.array([self._A_inv[i] @ self._b[i] for i in range(self.n_arms)])  # a * (f*f @ f*1) -> a*f
            self.theta_hat_x = self.theta_hat @ x
            #sigma_hat : rootの中の計算
            sigma_hat = np.array([np.sqrt(x.T @ self._A_inv[i] @ x) for i in range(self.n_arms)])  # a * (1*f @ f*f @ f*1) -> a*1
            mu_hat = self.theta_hat_x + self.alpha*math.sqrt(math.log(self.steps+1)) * sigma_hat  # a*f @ f*1 + a*1
            # print("mu_hat = ", mu_hat)
            result = np.argmax(mu_hat)

        return result

    def lin_ucb_update(self, x, chosen_arm, reward):
        # superの分
        self.steps += 1
        self.arm_counts[chosen_arm] += 1
        # superの分終わり
        self.reward_sum += reward
        self.arm_rewards[chosen_arm] += reward

        x = np.expand_dims(x, axis=1)
        self.A_inv[chosen_arm] -= \
            self.A_inv[chosen_arm] @ x @ x.T @ self.A_inv[chosen_arm] / (1 + x.T @ self.A_inv[chosen_arm] @ x)
        self.b[chosen_arm] += np.ravel(x) * reward

        if self.steps % self.batch_size == 0:
            self._A_inv, self._b = np.copy(self.A_inv), np.copy(self.b)

    def lin_ts(self, x):
        """腕の中から1つ選択肢し、インデックスを返す"""
        if True in (self.arm_counts < self.warmup):
            result = np.argmax(np.array(self.arm_counts < self.warmup))
        else:
            sigma2 = [self._ig_b[i] * invgamma.rvs(self._ig_a[i])
                      for i in range(self.n_arms)]
            try:
                beta = [np.random.multivariate_normal(self._mu[i], sigma2[i] * self._cov[i]) for i in range(self.n_arms)]
                #L = [np.linalg.cholesky(sigma2[i] * self._cov[i]) for i in range(self.n_arms)] #コレスキー分解
                #z = [np.random.standard_normal(len(sigma2[i] * self._cov[i])) for i in range(self.n_arms)] #標準正規乱数ベクトル
                #beta = [L[i] @ z[i] + self._mu[i] for i in range(self.n_arms)]
                '''L = [np.linalg.cholesky(self._cov[i]) for i in range(self.n_arms)] # コレスキー分解 この時点でsigma2でスケーリングするとエラーが増える印象?なので_covだけに
                z = [np.random.standard_normal(len(self._cov[i])) for i in range(self.n_arms)] # 標準正規乱数ベクトル 上に合わせて_covだけに
                beta = [sigma2[i] * L[i] @ z[i] + self._mu[i] for i in range(self.n_arms)]  # 最後にスケーリング'''
            except np.linalg.LinAlgError as e:
                """
                コレスキー分解等出来ない場合は従来通り多重正規分布をそのまま生成
                その分時間がかかる
                ※RuntimeError で multivariate_normal の共分散に関するエラーがでたりするが実行には一応問題なさそう
                """
                '''if self.gen_f():'''
                print('except')
                #d = self.n_features + 1
                #beta = [np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(self.n_arms)]
                '''try:
                    beta = [np.random.multivariate_normal(self._mu[i], sigma2[i] * self._cov[i]) for i in range(self.n_arms)] #多次元正規分布
                except np.linalg.LinAlgError as e:'''
                d = self.n_features + 1
                beta = [np.random.multivariate_normal(np.zeros(d), np.identity(d)) for _ in range(self.n_arms)]
            vals = [beta[i][: -1] @ x.T + beta[i][-1]
                    for i in range(self.n_arms)]
            self.theta_hat_x=[beta[i][: -1] @ x.T for i in range(self.n_arms)]

            result = np.argmax(vals)

        return result

    def lin_ts_update(self, x, chosen_arm, reward):
        # superの分
        self.steps += 1
        self.arm_counts[chosen_arm] += 1
        # superの分終わり
        self.reward_sum += reward
        self.arm_rewards[chosen_arm] += reward

        x = np.append(x, 1.0).reshape((1, self.n_features + 1))
        #A,bの更新
        precision_a = self.A[chosen_arm] + x.T @ x  # f*f
        cov_a = np.linalg.inv(precision_a)
        precision_b = self.b[chosen_arm] + x * reward  # 1*f
        mu_a = precision_b @ cov_a  # 1*f
        #alpha,betaの更新
        a_post = self._a0 + self.arm_counts[chosen_arm] / 2.0
        precision_c = self.c[chosen_arm] + reward * reward
        b_upd = 0.5 * (precision_c - mu_a @ precision_a @ mu_a.T)
        b_post = self._b0 + b_upd

        self.mu[chosen_arm] = mu_a[0]
        self.cov[chosen_arm] = cov_a
        self.A[chosen_arm] = precision_a
        self.b[chosen_arm] = precision_b
        self.c[chosen_arm] = precision_c
        self.ig_a[chosen_arm] = a_post
        self.ig_b[chosen_arm] = b_post[0]

        if self.steps % self.batch_size == 0:
            self._mu, self._cov = np.copy(self.mu), np.copy(self.cov)
            self._A, self._b, self._c = np.copy(self.A), np.copy(self.b), np.copy(self.c)
            self._ig_a, self._ig_b = np.copy(self.ig_a), np.copy(self.ig_b)


    def gen_f(self):
        """行列分解に関してのエラーが1度でも起きたかどうかのフラグを返す"""
        # for LinTS
        if self.flag:
            self.flag = False
            return True
        else:
            return False
    
    def change_prm(self, chosen_prm_idx):
        """
            パラメータを変える関数
            Arg:
                chosen_prm_idx : 上位エージェントが選んだ下位エージェントのパラメータインデックス

            他の方策を使いたいときはstable_param.pyにリストを追加すればOK
            パラメータの候補(上位エージェントの選択肢の数)は増やせるようにできてるはず
            ※pngに出力する用の選択比率は2本腕前提のまま
        """
        # print("self.name = ", self.name)
        if self.name == 'Regional LinRS':
            self.w = stable_param.STABLE_W_LIST[chosen_prm_idx]
        elif self.name == 'LinUCB':
            self.alpha = stable_param.ALPHA_LIST[chosen_prm_idx]
        elif self.name == 'LinTS':
            self._a0 = stable_param.AB_LIST[chosen_prm_idx]
            self._b0 = stable_param.AB_LIST[chosen_prm_idx]
        else:
            print("stable_controller/policy.py change_prm name ERROR!")
            exit(0)

    # debug
    def print_prm(self):
        # print("n_arms = ", self.n_arms)
        # print("n_features = ", self.n_features)
        # print("warmup = ", self.warmup)
        # print("batch_size = ", self.batch_size)
        print("steps = ", self.steps)
        print("reward_sum = ", self.reward_sum)
        print("arm_counts = ", self.arm_counts)
        print("arm_rewards = ", self.arm_rewards)

        if self.param_dic['policy_name']=='Regional_LinRS':
            print("aleph = ", self.aleph)
            # print("k = ", self.k)
            # print("episodic memory = ", self.episodic_memory)
            # print("memory capacity = ", self.memory_capacity)
            # print("zeta = ", self.zeta)
            # print("epsilon = ", self.epsilon)
            print("stable flag = ", self.stable_flag)
            print("w = ", self.w)
            print("theta hat x = ", self.theta_hat_x)
            print("rs = ", self.rs)
            print("n = ", self.n)
        elif self.param_dic['policy_name']=='LinUCB':
            print("alpha = ", self.alpha)
        elif self.param_dic['policy_name']=='LinTS':
            print("_lambda_prior = ", self._lambda_prior)
            print("_a0 = ", self._a0)
            print("_b0 = ", self._b0)
        else:
            print("stable_controller/policy.py print_prm ERROR!")

    def get_entropy_arm(self) -> np.ndarray:
        if np.sum(self.n)==0:
            return 1
        return entropy(self.n, base=self.n_arms)
