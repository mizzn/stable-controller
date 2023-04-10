# -*- coding: utf-8 -*-
"""実行モジュール"""

# from sim.real_sim_sc import ContextualBanditSimulator
# from bandit.contextual_bandit import ContextualBandit
from realworld.setup_context import ContextData
from policy_dic import AGENT
# from policy_dic import HIGHER_AGENT

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
    n_contexts = 10 #データの個数
    # n_contexts = 100
    data_type = 'mushroom'
    #data_type = 'financial'
    #data_type = 'jester'
    # data_type = 'artificial_0.7'
    # data_type = 'mixed_artificial_0.7'
    # artificialは重すぎてgitあがらなかった
    
    #setup_context.pyより
    num_actions, context_dim = ContextData.get_data_info(data_type)

    # N_SIMS = 100 #シュミレーション数
    N_SIMS = 2
    N_STEPS = n_contexts #step数 データの個数と同じにするのはどうして？

    N_ARMS = num_actions #行動の選択肢=腕の本数
    N_FEATURES = context_dim #特徴量の次元数

    print("N_ARMS = ", N_ARMS)
    print("N_FEATURES = ", N_FEATURES)

    # print(AGENT)

    policy_list = ['Regional_LinRS','LinUCB','LinTS']

    for i in policy_list:
        AGENT[i]['n_arms'] = N_ARMS
        AGENT[i]['n_features'] = N_FEATURES

    # n_arms, n_featuresは更新できてる！
    # print(AGENT)
    exit()
  
    #bandit/contextual_bandit.pyより文脈付きバンディットをインスタンス化
    bandit = ContextualBandit(n_arms=N_ARMS, n_features=N_FEATURES, n_contexts=N_STEPS, data_type=data_type)

    # sim/real_sim_scのContextualBanditSimulator
    bs = ContextualBanditSimulator(policy_list=policy_list, bandit=bandit, n_sims=N_SIMS,
                         n_steps=N_STEPS, n_arms=N_ARMS, n_features=N_FEATURES, data_type=data_type)
    #シミュレーターを実行
    bs.run()

main()
