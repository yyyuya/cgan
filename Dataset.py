import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.image_names = df['ID'].astype(str).to_list()
        self.transform = transform
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = f'/home/ytakeda/fashion_images/keep_aspect_resized/{image_name[0:3]}/{image_name}'
        if os.path.exists(image_path):
            image = Image.open(image_path) # 画像ファイルの読込
            image = np.array(image).astype(np.float32).transpose(2, 1, 0) # Dataloader で使うために転置する

            cols = self.df.columns[2:]
            tags = self.df[cols].values[index]

            print(index)
            return image, tags

    def _open_file(self, fname):
        return open(os.path.join(self._path, fname), 'rb')



# df_test = pd.read_csv('/home/ytakeda/fashion_images/keep_aspect_resized/test_images.csv')
# df_train = pd.read_csv('/home/ytakeda/fashion_images/keep_aspect_resized/train_images.csv', encoding="shift-jis") # trainだけなぜかShift-JIS
# df = pd.concat([df_test, df_train], ignore_index=True)

# mydata = Dataset(df)