import datetime
import os
import time
import numpy as np
import pandas as pd

from typing import List

from bandit.base_bandit import BaseBandit
from realworld.setup_context import ContextData
from sim.stable_controller_sim import StableControllerSimulator

from sklearn.metrics import mean_squared_error

class ContextualBanditSimulator(object):
    """Stable Controllerを使用した文脈付きバンディットを実行する

    Args:
        context_dim(int): 特徴ベクトルの次元数
        num_actions(int): 選択肢の数
        dataset(int, float): 特徴量 + 各行動の報酬が入ったデータセット
        algos(list,): 使用するアルゴリズムのリスト
    """

    def __init__(self, policy_list:List[str], bandit:BaseBandit, n_sims:int, n_steps:int, n_arms:int, n_features: int, data_type) -> None:
        """クラスの初期化
        Args:
        policy_list(list of str) : 下位エージェントの方策リスト
        meta_policy : 上位エージェントの方策
        bandit : bandit環境
        n_sims(int) : sim数
        n_steps(int) : step数
        n_arms(int) : アーム数(行動数)
        n_features(int) : 特徴量の次元数
        data_type(str) : dataset名
        """
        # print("real_sim_sc.py")
        
        self.bandit = bandit
        self.policy_list = policy_list
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.n_arms = n_arms
        self.n_features = n_features #追加

        self.data_type = data_type

        self.policy_name = [] 
        self.result_list = []
        self.elapsed_time_tmp = np.zeros(len(policy_list))

    def generate_feature_data(self):
        """加工したdataset・最適行動を取った時の報酬・最適行動・報酬期待値を返す"""
        #realworld/setup_context.py
        dataset, opt_rewards, opt_actions, exp_rewards = ContextData.sample_data(self.data_type, self.n_steps) #データの種類とステップ数が引数
        return dataset, opt_rewards, opt_actions, exp_rewards

    def processing_data(self, rewards,regrets,accuracy,successes,errors,entropy_of_reliability,times):
        """データを平均、最小値、最大値に処理して返す"""
        result_data = np.concatenate(([rewards], [regrets], [accuracy],[successes],[errors],[entropy_of_reliability],[times]),axis=0) # 結合して一気に
        min_data = result_data.min(axis=1)
        max_data = result_data.max(axis=1)
        ave_data = np.sum(result_data, axis=1) / self.n_sims

        return ave_data[0], ave_data[1], ave_data[2],ave_data[3],ave_data[4],ave_data[5],ave_data[6], min_data, max_data

    def run_sim(self):
        """方策ごとにシミュレーションを回す"""
        cmab = self.bandit
        i = 0 

        # ディレクトリを作る
        time_now = datetime.datetime.now() #現在時刻取得
        result_dir = 'csv/{0:%Y%m%d%H%M}'.format(time_now) #Yは西暦4桁　ただの文字列
        os.makedirs(result_dir, exist_ok=True) #ディレクトリを作る、exit_okがTrueのため既存のディレクトリがあってもエラーにならない
        print(result_dir, "を作成")

        result_dir = 'csv_reward_count/{0:%Y%m%d%H%M}'.format(time_now)
        os.makedirs(result_dir, exist_ok=True)
        print(result_dir, "を作成")

        result_dir = 'csv_mse/{0:%Y%m%d%H%M}'.format(time_now)
        os.makedirs(result_dir, exist_ok=True)
        print(result_dir, "を作成")

        result_dir = 'csv_chosen_agent/{0:%Y%m%d%H%M}'.format(time_now)
        os.makedirs(result_dir, exist_ok=True)
        print(result_dir, "を作成")


        print("")
        print("")
        print("")


        """方策ごとに実行"""
        for policy in self.policy_list:
            # print(policy.name, "で実行")
            # self.policy_name.append(policy.name) # 名前を記憶しておく

            """結果表示に用いる変数の初期化"""
            rewards = np.zeros((self.n_sims, self.n_steps), dtype=float) #sims行steps列の0行列を作っておく
            reward_csv = np.zeros((self.n_sims, self.n_steps), dtype=float)
            regrets = np.zeros((self.n_sims, self.n_steps), dtype=float)
            entropy_of_reliability = np.zeros((self.n_sims, self.n_steps), dtype=float)
            errors = np.zeros((self.n_sims, self.n_steps), dtype=float)
            mse_arm = np.zeros((self.n_sims, self.n_steps,self.n_arms), dtype=float)
            successes = np.zeros((self.n_sims, self.n_steps), dtype=float)
            accuracy = np.zeros((self.n_sims, self.n_steps), dtype=float)
            times = np.zeros((self.n_sims, self.n_steps), dtype=float)
            elapsed_time = 0.0 #何用？
            chosen_agent_history = np.zeros((self.n_sims, self.n_steps), dtype=int)

            start_tmp = time.time() # たぶん計算時間表示用

            

            """シミュレーション開始"""
            for sim in np.arange(self.n_sims):
                # print('{}'.format(sim), end='')
                print("")
                print(sim, "回目のシミュレーション")
                # print("加工したデータセット、理想の報酬、するべき選択、報酬期待値を受け取る")
                dataset, opt_rewards, opt_actions,exp_rewards = self.generate_feature_data() #加工したデータセットを持ってくる
                # print("加工したデータセット dataset = ", dataset)
                # print("理想の報酬 opt_rewars = ", opt_rewards)
                # print("するべき選択 opt_action = ", opt_actions)
                # print("報酬期待値 exp_rewards = ", exp_rewards)

                # print("1シミュレーションごとにデータの中から必要なぶんだけ取り出し(初期化も兼ねてる)")
                cmab.feed_data(dataset)
                # print("stableControllerSimulatorを実行するクラスをインスタンス化")
                
                higher_policy = self.set_higher_policy(policy)
                print("higher policy = ", higher_policy)
                
                scs = StableControllerSimulator(n_arms=self.n_arms, agent=policy, higher_agent=higher_policy, n_features=self.n_features) # stable_w_list削除版
                if (sim==0):
                    # 0回目のシミュレーションのときだけ名前を追加
                    name = scs.get_agent_name()
                    self.policy_name.append(name)

                # print("self.policy_name = ", self.policy_name) # 要確認

                """初期化"""
                elapsed_time =0.0
                sum_reward, sum_regret = 0.0, 0.0
                mse_tmp = np.zeros((self.n_steps, self.n_arms), dtype=float)

                """step開始"""
                for step in np.arange(self.n_steps):
                    start=time.time()
                    # print("\n\n\n")
                    # print(step, "step目")
                    # print("バンディットから1step分の特徴量を受け取る")
                    x = cmab.context(step)#特徴量のみ持ってくる
                    # print("x = ", x)
                    # print("方策から選んだ腕を受け取る")
                    chosen_arm, theta_hat = scs.choose_arm(x)
                    # print("simが受け取った下位エージェントが選んだ腕 chosen_arm = ", chosen_arm)
                    # print("simが受け取ったtheta_hat = ", theta_hat) # ちゃんと帰ってきてる
                    
                    chosen_prm_idx = scs.get_chosen_prm_idx()
                    # print("simが受け取った選ばれたパラメータのインデックス chosen_prm_idx = ", chosen_prm_idx) # 上位エージェントの選択と一致、いいね
                    
                    reward = cmab.reward(step, chosen_arm)
                    # print("選んだ腕をバンディットに渡して報酬をもらう")
                    # print("real_sim_sc/reward = ", reward)

                    # print("regretを計算")
                    regret = opt_rewards[step] - self.get_exp_reward(reward, step, opt_actions)
                    # print("opt_reward[step] = ", opt_rewards[step])
                    # print("計算されたregret =", regret)

                    # print("name = ", name)
                    if "LinRS" in name:
                        reliability = scs.get_entropy_arm() #底が腕の本数のエントロピーの計算
                        entropy_of_reliability[sim,step] += reliability

                    success_acc = 1 if chosen_arm == opt_actions[step] else 0#真のgreedy(Accuracy)　選んだ腕がoptなら1、違うなら0
                    success_greedy = 1 if chosen_arm == np.argmax(theta_hat) else 0#主観greedy 選んだ腕がgreedyなら1、違うなら0

                    if self.data_type == 'mushroom':
                        # 平均２乗誤差を計算
                        # print("平均２乗誤差を計算")
                        theta_error = mean_squared_error([row[opt_actions[step]] for row in exp_rewards], theta_hat, squared=False) #True:MSE, False:RMSE
                        #theta_error = mean_absolute_error([row[opt_actions[step]] for row in exp_rewards], theta_hat)
                    elif self.data_type.startswith('mixed_artificial'):
                        #theta_error = mean_squared_error(exp_rewards[step], theta_hat, squared=False)
                        #theta_error = mean_absolute_error(exp_rewards[step], theta_hat)
                        theta_error = np.average((exp_rewards[step]-theta_hat)/exp_rewards[step]) #MPE
                        old_sort_mse= mean_squared_error([exp_rewards[step]], [theta_hat], multioutput='raw_values') #各腕ごとの MSE
                        l = np.argsort(exp_rewards[step])[::-1]
                        m = 0
                        for k in l:
                            mse_tmp[step,k] += old_sort_mse[m]
                            m += 1
                    else:
                        print("The errror for data_type is not calculated!!")

                    # print("meta algorithmのアップデートを呼び出す")
                    scs.update(x, chosen_arm, reward)#方策アップデート

                    sum_reward += reward
                    sum_regret += regret
                    
                    rewards[sim, step] += sum_reward
                    reward_csv[sim,step] += reward
                    regrets[sim, step] += sum_regret
                    errors[sim,step] += theta_error
                    accuracy[sim, step] += success_acc
                    successes[sim, step] += success_greedy
                    elapsed_time += time.time()-start
                    times[sim,step] +=elapsed_time
                    chosen_agent_history[sim, step] = chosen_prm_idx # ここは可変じゃない　上位エージェントの行動数を増やしたときは変える必要あり
                    
                # print("entropy_of_reliability = ", entropy_of_reliability)

                print('{}'.format(regrets[sim, -1]))
                mse_arm[sim,:,:]+= mse_tmp
            # シミュレーション終わり

            # print("chosen_agent_history = ")
            # print(chosen_agent_history)

            self.elapsed_time_tmp[i] = time.time() - start_tmp
            i += 1
            #print("経過時間 : {}".format(elapsed_time_tmp))
            mse = np.mean(mse_arm,axis=0)
            print("mse:",mse)

            ave_rewards, ave_regrets, accuracy,greedy_rate,errors,entropy_of_reliability,ave_times,min_data,max_data = \
            self.processing_data(rewards, regrets, accuracy,successes,errors,entropy_of_reliability,times)
            data = [ave_rewards, ave_regrets, accuracy,greedy_rate,errors,entropy_of_reliability,ave_times,min_data, max_data]

            data_dic = \
                {'rewards': data[0], 'regrets': data[1], 'accuracy': data[2],'greedy_rate': data[3],'errors':data[4],'entropy_of_reliability':data[5],'times':data[6],
                'min_rewards': data[7][0], 'min_regrets': data[7][1],'min_accuracy': data[7][2],'min_greedy_rate': data[7][3],'min_errors':data[7][4],'min_entropy_of_reliability':data[7][5],'min_times':data[7][6],
                'max_rewards': data[8][0],'max_regrets': data[8][1], 'max_accuracy': data[8][2],'max_greedy_rate': data[8][3],'max_errors':data[8][4],'max_entropy_of_reliability':data[8][5],'max_times':data[8][6]}
            data_dic_pd = pd.DataFrame(data_dic) # 辞書をpandasのdata_frameにする
            # data_dic_pd.to_csv('csv/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, policy.name.replace(' ', '_')))
            data_dic_pd.to_csv('csv/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, name.replace(' ', '_'))) # dataframeをcsvに変換

            # chosen_agent_pd = pd.DataFrame(chosen_agent_history)
            # print('pd = ')
            # print(chosen_agent_pd)
            # chosen_agent_pd.to_csv('csv_chosen_agent/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, name.replace(' ', '_'))) # dataframeをcsvに変換
            np.savetxt('csv_chosen_agent/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, name.replace(' ', '_')), chosen_agent_history, fmt='%d')

            reward_csv_pd = pd.DataFrame(reward_csv) # 配列をdataframeに変換
            # reward_csv_pd.to_csv('csv_reward_count/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, policy.name.replace(' ', '_')))
            reward_csv_pd.to_csv('csv_reward_count/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, name.replace(' ', '_'))) # dataframeをcsvに変換
            # print('rewards: {0}\nregrets: {1}\naccuracy: {2}\ngreedy_rate: {3}\nerrors:{4}\ntimes: {5}\n'
                # 'min_rewards: {6}\nmin_regrets: {7}\nmin_accuracy: {8}\nmin_greedy_rate: {9}\nmin_errors: {10}\nmin_times:{11}\n'
                # 'max_rewards: {12}\nmax_regrets: {13}\nmax_accuracy: {14}\nmax_greedy_rate: {15}\nmax_errors:{16}\nmax_times:{17}'
                # .format(data_dic['rewards'], data_dic['regrets'],data_dic['accuracy'],data_dic['greedy_rate'],data_dic['errors'] ,data_dic['times'],
                #         data_dic['min_rewards'],data_dic['min_regrets'], data_dic['min_accuracy'],data_dic['min_greedy_rate'],data_dic['min_errors'],data_dic['min_times'],
                #         data_dic['max_rewards'], data_dic['max_regrets'],data_dic['max_accuracy'],data_dic['max_greedy_rate'],data_dic['max_errors'],data_dic['max_times']))
            mse_pd = pd.DataFrame(mse)
            # mse_pd.to_csv('csv_mse/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, policy.name.replace(' ', '_')))
            mse_pd.to_csv('csv_mse/{0:%Y%m%d%H%M}/{1}.csv'.format(time_now, name.replace(' ', '_')))

            # chosen_agent_rate_pd = data_dic_pd
            # print("data = ")
            # print(data_dic['chosen_agent_rate'])

            self.result_list.append(data_dic)
            self.dy = np.gradient(data[1])
            # print('傾き: ', self.dy)
        sim_time = self.elapsed_time_tmp / self.n_sims
        print("1 sim time:", sim_time)
        self.result_list = pd.DataFrame(self.result_list)


    # わかりやすくするだけらしい
    def run(self):
        """一連のシミュレーションを実行"""
        self.run_sim()

    def plot(self):
        """結果データのプロット"""
        pass

    def get_exp_reward(self, reward, step, opt_actions):
        if self.data_type == "mushroom":
            # print("sim/real_sim_sc.py/get_exp_reward mushroom")
            if reward == -35:
                #絶対に毒キノコ
                exp = -15
            elif reward == 0:
                #食用だけど食べなかったとき
                exp = 0
            else:
                if opt_actions[step]==0:
                    exp = 0
                else:
                    exp = 5
            return exp             
        else:
            # print("sim/real_sim_sc.py/get_exp_reward else")
            return reward

    def set_higher_policy(self, current_policy):
        print("current_policy = ", current_policy)
        if current_policy == "Regional_LinRS":
            higher_policy = "RS"
        elif current_policy == "LinUCB":
            higher_policy = "UCB"
        elif current_policy == "LinTS":
            higher_policy = "TS"
        else:
            print("real_sim_sc.py set_higher_policy ERROR!")
            exit(0)
        return higher_policy



