[広範なデータへ動的に対応する目的志向探索](https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_3R5GS204/_article/-char/ja/)のシミュレーションに使用したコードです．
詳しくは論文を参照してください．

# 実行方法

`python main.py`で実行できます．

4/10現在
使うデータセットはmain.pyを書き換えることで変更してください．
上位エージェントの選択肢として与えたいはstable_param.pyを書き換えることで変更してください．
コード上は3つ以上の選択肢を持てるようにしてありますが，未検証です．
エージェントのパラメータはpolicy_dic.pyで変更してください．


# datasets
    mushroom.csv https://archive.ics.uci.edu/ml/datasets/mushroom
