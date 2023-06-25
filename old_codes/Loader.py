import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from Models import PreTrmodel
from sklearn.preprocessing import LabelEncoder

class IrelandDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def loader(args_config, root_data):

    # Data Preparation
    data = pd.read_csv(root_data, delimiter=',').iloc[:10000, :]
    data.drop_duplicates(inplace=True)
    data = data.dropna()
    print('Number of class:', data['headline_category'].nunique())

    # Splitting the Data
    X_train, X_test, y_train, y_test = train_test_split(data['headline_text'], data['headline_category'], test_size=0.2,
                                                        random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model, tokenizer = PreTrmodel(args_config, size=len(X_train.unique()))

    # Encoding the Text
    max_length = 128  # Set the maximum sequence length
    X_train_encodings = tokenizer.batch_encode_plus(list(X_train), truncation=True, padding=True,
                                                    max_length=max_length, return_tensors='pt')
    X_val_encodings = tokenizer.batch_encode_plus(list(X_val), truncation=True, padding=True,
                                                    max_length=max_length, return_tensors='pt')
    X_test_encodings = tokenizer.batch_encode_plus(list(X_test), truncation=True, padding=True,
                                                   max_length=max_length, return_tensors='pt')

    # Encoding the Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.fit_transform(y_val)
    y_test = label_encoder.transform(y_test)

    train_dataset = IrelandDataset(X_train_encodings, y_train)
    val_dataset = IrelandDataset(X_val_encodings, y_val)
    test_dataset = IrelandDataset(X_test_encodings, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args_config.batch_size,
                                              shuffle=True, num_workers=args_config.workers)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args_config.batch_size,
                                              shuffle=False, num_workers=args_config.workers)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args_config.batch_size,
                                              shuffle=False, num_workers=args_config.workers)

    return train_loader, val_loader, test_loader, model, label_encoder


