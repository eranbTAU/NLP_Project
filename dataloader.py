import pickle
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from sklearn.model_selection import train_test_split

def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR * len(dataset)), len(dataset)))
    return trainDataset, valDataset

class RegPointDataset(Dataset):
    def __init__(self, data_array, data_transforms):
        self.transform = data_transforms
        self.data_array = data_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, index):
        sample = self.data_array[index]
        image_path = os.path.join(self.dir_img, str(sample[2]), str(sample[1]), 'image', str(sample[0]))
        img = Image.open(image_path)
        img_transform = self.transform(img)
        label = self.classes[sample[2]]
        return img_transform, label


def data_loaders(args_config, root_data):

    data = pd.read_csv(root_data, delimiter=',')
    print('data_shape : {}'
          '\n10-start_column : {}'.format(data.shape,
                                        data.head(10)))
    print(data.describe())
    category = [i for i in data.keys() if i.endswith('category')]
    grouped = data.groupby(category[0])
    print(grouped.mean())

    data_array = np.array(data)

    train_data, test_data = train_test_split(data_array, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    data_transforms = {
            'train':
                transforms.Compose([
                    # transforms.ColorJitter(brightness=.5, hue=.3),
                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                    transforms.ToTensor(),
                ]),

            'test':
                transforms.Compose([
                    transforms.ToTensor()
                ])
        }

    train_dataset = RegPointDataset(train_data, data_transforms['train'])
    val_dataset = RegPointDataset(val_data, data_transforms['test'])
    test_dataset = RegPointDataset(test_data, data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args_config.batch_size,
                                              shuffle=True, num_workers=args_config.workers, drop_last=args_config.drop_last)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args_config.batch_size,
                                              shuffle=False, num_workers=args_config.workers, drop_last=args_config.drop_last)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args_config.batch_size,
                                              shuffle=False, num_workers=args_config.workers, drop_last=args_config.drop_last)

    return train_loader, val_loader, test_loader


