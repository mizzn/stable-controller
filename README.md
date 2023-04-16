# 実行方法

`python main.py`で実行できます．

4/10現在
使うデータセットはmain.pyを書き換えることで変更してください．
上位エージェントの選択肢として与えたいはstable_param.pyを書き換えることで変更してください．
コード上は3つ以上の選択肢を持てるようにしてありますが，未検証です．
エージェントのパラメータはpolicy_dic.pyで変更してください．

# stable controller
stable controllerは上位エージェントと下位エージェントの組み合わせから構成されています．

#　その他補足
## bandit
    base_bandit.py 未使用
    contextual_bandit.py 文脈付きバンディットを実装

## datasets
    mushroom.csv https://archive.ics.uci.edu/ml/datasets/mushroom

##　policy
    これ使ってるっけ？
