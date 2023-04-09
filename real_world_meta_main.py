# -*- coding: utf-8 -*-
"""実行モジュール"""

from sim.real_meta_sim import ContextualBanditSimulator
from bandit.contextual_bandit import ContextualBandit
# from policy.linucb import LinUCB
# from policy.lints import LinTS
# from policy.linear_full_posterior_sampling import LinearTS
# from policy.linear_full_posterior_sampling_fixed import LinearTSfixed
# from policy.linrs import LinRS
# from policy.linrs_stable import StableLinRS
from policy.regional_linrs import RegionalLinRS
# from policy.uniform import Uniform
from policy.greedy import Greedy
from realworld.setup_context import ContextData
from policy_dic import AGENT
from policy_dic import HIGHER_AGENT

def main():
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

    # n_contexts = 100000
    # n_contexts = 8000 #データの個数
    n_contexts = 10
    data_type = 'mushroom'
    #data_type = 'financial'
    #data_type = 'jester'
    #data_type = 'artificial_0.7'
    # data_type = 'mixed_artificial_0.7'
    # print("n_contexts = "+str(n_contexts))
    # print("data_type = "+str(data_type))
    
    #setup_context.pyより
    #data_typeによってnum_actions(int):行動数とcontext_dim(int):特徴量の次元を返す
    num_actions, context_dim = ContextData.get_data_info(data_type)
    # print("num_actions = "+str(num_actions))
    # print("context_dim = "+str(context_dim))

    # N_SIMS = 100 #シュミレーション数
    N_SIMS = 2
    N_STEPS = n_contexts #step数 データの個数と同じにするのはどうして？

    N_ARMS = num_actions #行動の選択肢=腕の本数
    N_FEATURES = context_dim #特徴量の次元数

    STABLE_W_LIST = [1.0, 0.5]

    print("N_ARMS = ", N_ARMS)
    print("N_FEATURES = ", N_FEATURES)

    # print(AGENT)

    #Mushroom
    policy_list = ['Regional_LinRS']
    #何入れたらいいのかわからないのでとりあえずgreedy
    meta_policy = 'RS'

    for i in policy_list:
        AGENT[i]['n_arms'] = N_ARMS
        AGENT[i]['n_features'] = N_FEATURES

    HIGHER_AGENT[meta_policy]['n_arms'] = N_ARMS

    # n_arms, n_featuresは更新できてる！
    # print(AGENT)

    #Jester
    # policy_list = [LinUCB(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, alpha=0.1)
    #             , LinearTSfixed(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, lambda_prior=0.25, a0=6, b0=6)
    #             , Stable(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20,n_steps=N_STEPS, aleph=0.2,eta=0.01)]

    #artificial
    #policyの中身は後で見る
    """policy_list = [RegionalLinRS(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, aleph=0.6, k = 50, memory_capacity = 10000, zeta = 0.008, epsilon = 0.0001)
                , LinearTSfixed(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, lambda_prior=0.25, a0=6, b0=6)
                , LinUCB(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20, alpha=0.1)
                , Greedy(n_arms=N_ARMS, n_features=N_FEATURES, warmup=10, batch_size=20,n_steps=N_STEPS)]"""
  
    #bandit/contextual_bandit.pyより文脈付きバンディットをインスタンス化
    bandit = ContextualBandit(n_arms=N_ARMS, n_features=N_FEATURES, n_contexts=N_STEPS, data_type=data_type)

    # これmetaでも変わらんかも←役割は変わらないけどコードは書き換えるので増やしました
    # real_meta_simのContextualBanditSimulator
    bs = ContextualBanditSimulator(policy_list=policy_list, meta_policy=meta_policy, bandit=bandit, n_sims=N_SIMS,
                         n_steps=N_STEPS, n_arms=N_ARMS, n_features=N_FEATURES, data_type=data_type, stable_w_list=STABLE_W_LIST)
    #シミュレーターを実行
    bs.run()

main()
