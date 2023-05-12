# 実行方法

`python main.py`で実行できます．

4/10現在
使うデータセットはmain.pyを書き換えることで変更してください．
上位エージェントの選択肢として与えたいはstable_param.pyを書き換えることで変更してください．
コード上は3つ以上の選択肢を持てるようにしてありますが，未検証です．
エージェントのパラメータはpolicy_dic.pyで変更してください．

# パラメータ設定
![パラメータ設定](/img/prm.png)

# stable controller
stable controllerは上位エージェントと下位エージェントの組み合わせから構成されています．

#　その他補足
## bandit
    base_bandit.py 未使用
    contextual_bandit.py 文脈付きバンディットを実装

## datasets
### mushroom.csv 
https://archive.ics.uci.edu/ml/datasets/mushroom
毒キノコを食用キノコを判別する
行動数：2個　食べるか食べないか
報酬：
食用キノコを食べたときは5
毒キノコを食べたときは0.5の確率で5, 0.5の確率で-35
食べなかったときは0
次元：117

### mnist


##　policy
    これ使ってるっけ？
