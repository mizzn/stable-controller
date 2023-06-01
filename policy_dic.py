#下位エージェントのパラメータの辞書
# 
AGENT = {
    # n_arms: int, n_features: int, warmup: int=1, batch_size: int=1,n_steps:int=1
    'Greedy': {
        'n_arms': 2,
        'n_features': 128,
        'warmup': 10,
        'batch_size': 20,
        'policy_name': 'Greedy', 
        'n_steps': 1,
    },
    # _arms:int, n_features:int, warmup:int=1, batch_size:int=1, aleph:float=1.0, k:int=10, memory_capacity:int=30000, zeta:float=0.0001, epsilon:float= 0.0001, stable_flag:bool=False, w:float=0.1
    'Regional_LinRS': {#stable_flagはいらないのでは？って思うかもだけど疑似試行割合の計算に必要！あとで直したい
        'n_arms': 2,
        'n_features': 128,
        'warmup': 10,
        'batch_size': 20,
        'policy_name': 'Regional_LinRS', 
        'aleph': 0.6, 
        'k': 50, 
        'memory_capacity': 10000, 
        'zeta': 0.008, 
        'epsilon': 0.0001, 
        'stable_flag':True, 
        'w':0.1,
    },
    'SRS_CH': {
        'n_arms': 2,
        'n_features': 128,
        'warmup': 10,
        'batch_size': 20, # たぶん使わないけどpolicy分岐の前に入れちゃったから、、、可能ならあとで直したい上位エージェント用の新しい辞書を作ってもいいかも
        'policy_name': 'SRS_CH',
    },
    'LinUCB': {
        'n_arms': 2,
        'n_features': 128,
        'warmup': 10,
        'batch_size': 20,
        'policy_name': 'LinUCB', 
        'alpha':0.1,
    },
    'LinTS': {
        'n_arms': 2,
        'n_features': 128,
        'warmup': 10,
        'batch_size': 20,
        'policy_name': 'LinTS', 
        'alpha':1, 
        'beta':1, 
        'lambda_prior':0.25,
    }
}

# 上位エージェントのパラメータの辞書
# HIGHER_AGENT = {
#     'RS' : {
#         'n_arms' : 2, 
#         'policy_dic' : {'policy' : 'RS', 'aleph':0.6, 'weight_flag':False, 'w':0.9} #このwは期待値の重み,Falseで使わない
#     },
#     'SRS_CH' : {
#         'n_arms':2,
#         'policy_dic' : {'policy' : 'SRS_CH'}
#     },
#     'UCB' : {
#         'n_arms':2,
#         'policy_dic' : {'policy' : 'UCB'}
#     },
#     'TS' : {
#         'n_arms' : 2, 
#         'policy_dic' : {'policy' : 'TS', 'alpha':1, 'beta':10, 'mu0': 1.0}},
#     'SRS' : {
#         'n_arms':2,
#         'policy_dic' : {'policy' : 'SRS', 'epsilon':10**(-4), 'aleph':4.0}
#     }
# }
HIGHER_AGENT = {
    'RS' : {
        'n_arms' : 11, 
        'policy_name' : 'RS', 
        'aleph':0.6, 
    },
    # 'SRS_CH' : {
    #     'n_arms':11,
    #     'policy_name' : 'SRS_CH'
    # },
    'UCB' : {
        'n_arms':11,
        'policy_name' : 'UCB'
    },
    'TS' : {
        'n_arms' : 11, 
        'policy_name' : 'TS', 
        'alpha':1, 
        'beta':10, 
        'mu0': 1.0},
    # 'SRS' : {
    #     'n_arms':2,
    #     'policy_name' : 'SRS', 
    #     'epsilon':10**(-4), 
    #     'aleph':4.0
    # }
}