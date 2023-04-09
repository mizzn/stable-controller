import os
import pandas as pd
import numpy as np


base_route = os.getcwd() #カレントディレクトリ
data_route = 'datasets'

def one_hot(df, cols):
    """特徴量をone_hotベクトルに直す
    Args:
      cols(pandas.core.indexes.base.Index):特徴(edible,cap-shape...)

    Returns:
      df(pandas.core.frame.DataFrame):特徴量をone-hotベクトルに変換したもの

    Example:
      >>> df = one_hot(df, df.columns)
      >>> print(df.iloc(0, ))

      edible_e               0
      edible_p               1
      cap-shape_b            0
      cap-shape_c            0
      cap-shape_f            0
      cap-shape_k            0
      cap-shape_s            0
      cap-shape_x            1
      .
      .
      .
    """
    # 行ごとにダミー変数(one-hot)にする
    for col in cols:
      # print("col = ", col) #colは列の名前
      # print("df[col] = ", df[col]) #その列の要素
      dummies = pd.get_dummies(df[col], prefix=col, drop_first=False) #prefixでダミー変数のラベルの先頭に名前をつける
      # print("dummies = ", dummies)
      df = pd.concat([df, dummies], axis=1) # dataframeを連結
      # print("concated df = ", df)
      df = df.drop(col, axis=1)
      # print(df)

    return df

def sample_mushroom_data(num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):
    """mushroomデータセットの加工
    Args:
      num_contexts(int):データの数
      r_noeat(int):食べなかった時の報酬
      r_eat_safe(int):食用キノコを食べた時の報酬
      r_eat_poison_bad(int):毒キノコを食べた時の負の報酬
      r_eat_poison_good(int):毒キノコを食べた時の正の報酬
      prob_poison_bad(float):毒キノコを食べた時に負の報酬になる確率

    Returns:
      np.hstack((contexts, no_eat_reward, eat_reward)):加工後のデータセット[特徴ベクトル + 各行動の報酬]
      opt_vals(float):各stepで最適な行動を選択した時の報酬
      exp_rewards(float):各行動の報酬確率
    """


    # カレントディレクトリ/データセットのあるディレクトリ/データのパスを作る
    path = os.path.join(base_route, data_route, 'mushroom.csv')

    # print("作ったパスからcsvを読む")
    df = pd.read_csv(path)
    # print(df)
    # print("one-hotに変換")
    # print("列の名前：", df.columns)
    df = one_hot(df, df.columns) #df.columnsはデータフレームの列の名前
    # print(df)
    
    # print("サンプル抽出")
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True) # num_contextsの数だけランダムでサンプルを抽出[num_contexts,119]
    # print(ind)

    # print("特徴のみ持ってくる")
    contexts = df.iloc[ind, 2:] # ind番目のデータの特徴がcontextsに入ってる
    # print(contexts)

    exp_rewards =[[r_noeat,r_noeat],[(r_eat_poison_bad + r_eat_poison_good)*prob_poison_bad,r_eat_safe]]

    # print("キノコの報酬を設定する")
    no_eat_reward = r_noeat * np.ones((num_contexts, 1)) # num_context行1列の0行列
    # print("no_eat_reward = ",no_eat_reward)
    random_poison = np.random.choice(
          [r_eat_poison_bad, r_eat_poison_good],
          p=[prob_poison_bad, 1 - prob_poison_bad], # 確率prob_poison_badでr_eat_poison_badの報酬、確率1-prob_poison_badでr_eat_poison_goodの報酬
          size=num_contexts) 
    # print("random_poison = ", random_poison)
    # print("df.iloc[ind, 0] = ", df.iloc[ind, 0]) #ind行0列取り出し
    eat_reward = r_eat_safe * df.iloc[ind, 0] # 0列だから食用キノコを食べたときの報酬
    # print("eat_reward = ", eat_reward)
    # print("np.multiply(random_poison, df.iloc[ind, 1]) = ", np.multiply(random_poison, df.iloc[ind, 1]))
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1]) # 毒のときの報酬を要素ごとに掛け算、食用じゃないなら確率でマイナスになる
    # print("eat_reward = ", eat_reward)
    eat_reward = eat_reward.values.reshape((num_contexts, 1))
    # print("eat_reward = ", eat_reward)

    # 最適な期待報酬と最適な行動を計算
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad # 毒を食べたときの期待値+
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad) # 食用を食べたときの期待値
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]
    #　最適報酬=食用の報酬+max(食べない報酬, 食べたときの期待値)*毒
    # print("最適報酬 = ", opt_exp_reward)

    if r_noeat > exp_eat_poison_reward:
        opt_actions = df.iloc[ind, 0] # 食べない方がいいとき
    else:
        opt_actions = np.ones((num_contexts, 1))
    opt_vals = (opt_exp_reward.values, opt_actions.values)
    # print("最適行動 = ", opt_actions) #食べるなら1

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals, exp_rewards
    
def sample_mixed_artificial_data(data_type, num_contexts,context_dim):
    path_1 = os.path.join(base_route, data_route, 'artificial_feature_data_mixed_' + data_type[17:] + '.csv')
    df_x = pd.read_csv(path_1)

    ind = np.random.choice(range(df_x.shape[0]), num_contexts, replace=True)# num_contextsの数だけランダムに抽出
    df = df_x.iloc[ind,:].values

    exp_rewards = df_x.iloc[ind, context_dim:].values#報酬箇所

    """最適期待報酬と最適行動を求める"""
    opt_actions = np.argmax(exp_rewards, axis=1)#ユーザの評価値の中で1番大きい行動をopt_actionとする
    opt_rewards = np.array([exp_rewards[i,a] for i, a in enumerate(opt_actions)])#iはindex,aは要素(行動)
    opt_values = (opt_rewards, opt_actions)

    return df, opt_values, exp_rewards
