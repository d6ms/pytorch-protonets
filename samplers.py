import numpy as np
from torch.utils.data import Sampler, Dataset


class FewShotBatchSampler(Sampler):

    def __init__(self,
                 dataset: Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = None):
        """
        dataset: サンプリング元となるデータセット
        episodes_per_epoch: 1エポックに含めるepisodeの数
        n: kクラスのそれぞれに与えられるサンプル数
        k: 分類先のクラス数
        q: クラスごとに確保するquery setのサンプル数
        num_tasks: 1 episodeに含めるタスクの数? 1 episodeに含めるquery setの数？
        """
        super(FewShotBatchSampler, self).__init__(dataset)
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.n = n
        self.k = k
        self.q = q
        self.num_tasks = num_tasks

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = list()  # (n+q, 1) のデータセットID列．前半n個は support set，後半q個は query set．
            for _ in range(self.num_tasks):
                # このepisodeの対象となるkクラスをデータセットからランダムに選び取る
                df = self.dataset.df
                episode_classes = np.random.choice(df['class_id'].unique(), size=self.k, replace=False)
                df = df[df['class_id'].isin(episode_classes)]

                # kクラスのそれぞれに対してn個のサンプルを選択し，support setを作る
                support_set = dict()
                for class_ in episode_classes:
                    n_samples = df[df['class_id'] == class_].sample(self.n)
                    support_set[class_] = n_samples
                    for _, sample in n_samples.iterrows():
                        batch.append(sample['id'])
                
                # kクラスのそれぞれに対してq個のサンプルを選択し，query setを作る
                for class_ in episode_classes:
                    q_queries = df[(df['class_id'] == class_) & (~df['id'].isin(support_set[class_]['id']))].sample(self.q)
                    for _, query in q_queries.iterrows():
                        batch.append(query['id'])

            yield np.stack(batch)
