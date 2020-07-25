import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from tqdm import tqdm

import config


class OmniglotDataset(Dataset):

    def __init__(self, subset: str):
        super(OmniglotDataset, self).__init__()
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset can be whether (background, evaluation)')
        self.subset = subset
        self.df = self.__load_data(self.subset)
        print(self.df.head())

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.df)

    @staticmethod
    def __load_data(subset) -> pd.DataFrame:
        data = datasets.Omniglot(root=config.DATA_PATH, download=True, background=(subset == 'background'), transform=transforms.ToTensor())
        print(f'loading omniglot dataset ({subset})')

        total_images = 0
        for root, folders, files in os.walk(f'{config.DATA_PATH}/omniglot-py/images_{subset}/'):
            total_images += len(files)
        
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
        return pd.DataFrame(images)

