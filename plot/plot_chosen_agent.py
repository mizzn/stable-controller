import sys
import os
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import datetime

"""定数系""" # ここを必ず変える
n_steps = 10000
n_sims = 100
n_arms = 2 # 上位エージェントの行動数
names = ['Regional_LinRS','LinUCB','LinTS']
n_size = 2 # 棒グラフの幅
#うまくimportできなかった！必ず変えること！
STABLE_W_LIST = [1.0, 0.5]
ALPHA_LIST = [1.0, 0.5]
AB_LIST = [6.0, 6.0]

"""引数確認"""
args = sys.argv
if len(args) <= 1:
    print('Usage: python plot.py [path of directory where csv is stored]')
    sys.exit()

time_now = datetime.datetime.now()
results_dir = 'png/{0:%Y%m%d%H%M}/'.format(time_now)
os.makedirs(results_dir, exist_ok=True)

for name in names:
    """csvデータの取得"""
    file = np.loadtxt(args[1] + '/' + name + '.csv')

    # 初期化
    count_nonzero_array = np.zeros((n_arms, n_steps))
    result_array = np.zeros((n_arms, int(n_steps/n_size)))

    if name in 'Regional_LinRS':
        w_list = STABLE_W_LIST
        prm_legend = 'η='
    elif name in 'LinUCB':
        w_list = ALPHA_LIST
        prm_legend = 'α='
    elif name in 'LinTS':
        w_list = AB_LIST
        prm_legend = 'α=β='
    else:
        print("file name ERROR!")
        exit(0)

    for i in range(len(w_list)):
        # print("i = ", i)
        count_nonzero_array[i] = np.count_nonzero(file == i, axis=0)
    print("count_nonzero_array = ")
    print(count_nonzero_array)
    for row_i in range(n_arms):
        for column_i in range(int(n_steps/n_size)):
            result_array[row_i][column_i] = sum(count_nonzero_array[row_i][column_i*n_size:(column_i+1)*n_size])/(n_sims*n_size)
    print("result array = ")
    print(result_array)
    # exit(0)

    # 設定とか
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    # cmap = plt.get_cmap("tab10")
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('step', fontsize=23)
    ax.xaxis.offsetText.set_fontsize(23)
    ax.set_ylabel('chosen parameter rate',fontsize=23)
    # ax.legend(loc='upper right', fontsize=23)
    plt.tick_params(labelsize=30, rotation=30)
    ax.grid(alpha=0.8,color = "gray", linestyle="--")
    # fig.subplots_adjust(right=0.2)

    p1 = plt.bar(np.arange(1, int(n_steps/n_size)+1), result_array[0])
    p2 = plt.bar(np.arange(1, int(n_steps/n_size)+1), result_array[1], bottom=result_array[0])
    plt.legend((p1[0], p2[0]), (prm_legend+str(w_list[0]), prm_legend+str(w_list[1])),loc='upper right', fontsize=23)


    fig.savefig(results_dir + name+ '.png', bbox_inches='tight',pad_inches=0)
    plt.show()



# plt.clf()