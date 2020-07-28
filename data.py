import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms, datasets
from skimage import io
from tqdm import tqdm

import config


class OmniglotDataset(Dataset):

    def __init__(self, subset: str):
        super(OmniglotDataset, self).__init__()
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset can be whether (background, evaluation)')
        self.subset = subset
        self.df = self.__load_data(self.subset)
        self.idx_to_filepath = self.df.to_dict()['filepath']
        self.idx_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        f = self.idx_to_filepath[item]
        img = io.imread(f)  # [0, 255]で表現される行列
        img = img[np.newaxis, :, :]  # 先頭にchannelのための次元を追加
        img = (img - img.min()) / (img.max() - img.min())  # [0, 1]にnormalize
        label = self.idx_to_class_id[item]
        return torch.from_numpy(img), label

    def __len__(self):
        return len(self.df)
    
    def show_img(self, item):
        # method for debugging
        f = self.idx_to_filepath[item]
        from matplotlib import pyplot as plt
        io.imshow(img)
        plt.show()

    @staticmethod
    def __load_data(subset) -> pd.DataFrame:
        # 必要があればデータをダウンロード
        datasets.Omniglot(root=config.DATA_PATH, download=True, background=(subset == 'background'))

        # プログレスバー表示のため，全量をカウント
        print(f'loading omniglot dataset ({subset})')
        total_images = 0
        for root, folders, files in os.walk(f'{config.DATA_PATH}/omniglot-py/images_{subset}/'):
            total_images += len(files)
        
        # ファイルシステムを参照し，画像データに属性を付与してDataFrameをつくる
        progress = tqdm(total=total_images)
        images = list()
        for root, folders, files in os.walk(f'{config.DATA_PATH}/omniglot-py/images_{subset}/'):
            alphabet = root.split('/')[-2]
            class_name = alphabet + '.' + root.split('/')[-1]
            for f in files:
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })
                progress.update(1)
        progress.close()

        # DataFrameに変換
        df = pd.DataFrame(images)
        df = df.assign(id=df.index.values)  # indexに応じた値をIDカラムとして追加
        unique_characters = sorted(df['class_name'].unique())
        num_classes = len(df['class_name'].unique())
        class_name_to_id = {unique_characters[i]: i for i in range(num_classes)}
        df = df.assign(class_id=df['class_name'].apply(lambda c: class_name_to_id[c]))  # クラスごとにユニークなIDを振り，class_nameカラムとして追加
        return df


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
