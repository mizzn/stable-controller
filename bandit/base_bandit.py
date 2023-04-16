# 使っていない

class BaseBandit(object):
    """ベースとなるバンディットクラス

    Attributes:
        n_arms (int): 選択肢となる腕の数
        n_features (int): ユーザーの特徴数
    """

    def __init__(self):
        """クラスの初期化"""
        pass

    def initialize(self):
        """パラメータの初期化"""
        pass

    def pull(self):
        """選ばれた腕を引く"""
        pass