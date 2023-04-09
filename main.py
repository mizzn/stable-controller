
from realworld.setup_context import ContextData
from bandit.contextual_bandit import ContextualBandit
from sim.real_sim import ContextualBanditSimulator
from policy.greedy import Greedy
from policy.regional_linrs import RegionalLinRS
from policy.linucb import LinUCB


if __name__ == "__main__":
    """main

    Args:
        N_SIMS(int) : sim数
        N_STEPS(int) : step数
        n_contexts(int) : データの個数(step数)
        N_ARMS(int) : 行動の選択肢
        N_FEATURES(int) : 特徴量の次元数

        aleph(float) : 満足化基準値

        policy_list(list[str]) : 検証する方策クラスをまとめたリスト
        bandit : バンディット環境
        bs : バンディットシュミレーター
        data_type(str) : データの種類[mushroom, financial, jester, 
                                        artificial_0.5, artificial_0.7, artificial_0.9, 
                                        mixed_artificial_0.5, mixed_artificial_0.7, mixed_artificial_0.9]
    """

    # パラメータ設定
    # data_type = "mushroom"
    data_type = 'mixed_artificial_0.7'
    n_contexts = 2500 # データの個数
    # realworld/setup_context.py/ContextData/get_data_infoから行動数と特徴量の次元をもらってくる
    num_actions, context_dim = ContextData.get_data_info(data_type)
    N_ARMS = num_actions #行動の選択肢=腕の本数
    N_FEATURES = context_dim #特徴量の次元数

    N_SIM = 50
    N_STEPS = n_contexts

    # debug
    print("パラメータを設定")
    print("データの種類 data_type = ", data_type)
    print("データの個数 n_contexts = ", n_contexts)
    print("選択肢の数 N_ARMS = ", N_ARMS)
    print("特徴量の次元数 N_FEATURES = ", N_FEATURES)
    print("シミュレーション数 N_SIM = ", N_SIM)
    print("ステップ数 N_STEPS = ", N_STEPS)

    print("方策をインスタンス化")
    policy_list = [RegionalLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=0.6, k = 50, memory_capacity = 10000, zeta = 0.008, epsilon = 0.0001, stable_flag=True, w=0.0),
                   RegionalLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=0.6, k = 50, memory_capacity = 10000, zeta = 0.008, epsilon = 0.0001, stable_flag=True, w=0.5),
                   RegionalLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=0.6, k = 50, memory_capacity = 10000, zeta = 0.008, epsilon = 0.0001, stable_flag=True, w=1.0)]
    # policy_list = [LinUCB(n_arms= N_ARMS, n_features=N_FEATURES, warmup=1, batch_size=20, alpha=0.0)]
    print("文脈付きbanditをインスタンス化")
    bandit = ContextualBandit(n_features=N_FEATURES, n_arms=N_ARMS, n_contexts=N_STEPS, data_type=data_type)
    print("シミュレーターをインスタンス化")
    bs = ContextualBanditSimulator(policy_list=policy_list, bandit=bandit, n_sims=N_SIM, n_steps=N_STEPS, n_arms=N_ARMS, data_type=data_type, n_features=N_FEATURES)

    print("シミュレーターを実行")
    bs.run()


