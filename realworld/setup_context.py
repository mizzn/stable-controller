from realworld.data_sampler import sample_mushroom_data
from realworld.data_sampler import sample_mixed_artificial_data

"""データのセットアップをおこなうモジュール"""

class ContextData(object):
    """特徴のデータの基本情報をセットするクラス"""

    def sample_data(data_type, num_contexts=None):
        """データセットの最適行動、報酬期待値をセット
        Args:
            data_type(str):データセット名
            num_contexts(int):データ数
        Returns:
            dataset:加工済みデータ
            opt_rewards(int, float):最適報酬
            opt_actions(int):最適行動
            exp_rewards(int, float):報酬
        Raises:
            DATA NAME ERROR: data_typeがどれも当てはまらない場合
        """
        if data_type=='mushroom':
            dataset, opt_mushroom, exp_rewards = sample_mushroom_data(num_contexts) #realworld/data_sampler.py 
            # print("dataset = ", dataset)
            # print("opt_mushroom = ", opt_mushroom)
            # print("exp_rewards = ", exp_rewards)
            opt_rewards, opt_actions = opt_mushroom
            # print("opt_rewards = ", opt_rewards)
            # print("opt_actions = ", opt_actions)
        elif data_type.startswith('mixed_artificial'):
            num_actions = 8
            context_dim = 128
            dataset, opt_artificial, exp_rewards = sample_mixed_artificial_data(data_type, num_contexts, context_dim)
            opt_rewards, opt_actions = opt_artificial
        elif data_type=="mnist":
            pass
        else:
            print("DATA NAME ERROR.")

        return dataset, opt_rewards, opt_actions, exp_rewards



    def get_data_info(data_type):
        """dataの基本情報の取得
        Returns:
            num_actions(int):行動数
            context_dim(int):特徴量の次元
        """
        # print("realworld/setup_context.py/ContextData/get_data_info")

        if data_type == "mushroom":
            num_actions = 2 #行動数
            context_dim = 117 #特徴量の次元
        elif data_type.startswith('mixed_artificial'):
            num_actions = 8
            context_dim = 128
        else:
            print("data_type error")

        
        return num_actions, context_dim

