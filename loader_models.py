import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

class IrishTimesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)  # Return the length of the texts

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])  # Access the element using .iloc[idx]
        label = self.labels[idx]

        encoded_inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }

class TextClassificationModel:
    def __init__(self, args_config, df, class_weights_tensor, device):
        self.args_config = args_config
        self.class_weights_tensor = class_weights_tensor
        self.label_encoder = LabelEncoder()
        self.device = device
        self.df = df

    def prepare_model(self):
        if self.args_config.model_name == "bert":
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained("bert-base-uncased")
        else:
            from transformers import DistilBertModel, DistilBertTokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.classifier = nn.Linear(self.model.config.hidden_size, self.df['category'].nunique())
        self.classifier.to(self.device)
        self.model.to(self.device)

        if self.args_config.unbalance == False:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)

    def prepare_data_loaders(self, train_df, test_df):
        encoded_train_labels = self.label_encoder.fit_transform(train_df['category'])
        encoded_test_labels = self.label_encoder.fit_transform(test_df['category'])
        train_dataset = IrishTimesDataset(train_df['text'], encoded_train_labels, self.tokenizer)
        test_dataset = IrishTimesDataset(test_df['text'], encoded_test_labels, self.tokenizer)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args_config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args_config.batch_size)

    def train_model(self):
        self.model.train()
        self.classifier.train()

        if self.args_config.finetune:
            optimizer = torch.optim.AdamW(
                [{'params': self.classifier.parameters()}],
                lr=self.args_config.lr,
                weight_decay=self.args_config.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                [{'params': self.model.parameters()}, {'params': self.classifier.parameters()}],
                lr=self.args_config.lr
            )

        for epoch in range(self.args_config.epochs):
            pbar = tqdm(self.train_loader, total=len(self.train_loader))
            train_batch = []
            for i, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0]
                logits = self.classifier(pooled_output)

                loss = self.criterion(logits, labels)

                train_batch.append(loss.cpu().detach().numpy())

                loss.backward()
                optimizer.step()

                pbar.set_postfix({'Epoch': epoch,
                                  'Training Loss': np.mean(train_batch)
                                  })

    def evaluate(self):
        self.model.eval()
        self.classifier.eval()
        predictions = []
        true_labels = []
        epoch_test_loss = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, total=len(self.test_loader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0]
                logits = self.classifier(pooled_output)
                predicted_labels = logits.argmax(dim=1).cpu().numpy()
                loss = self.criterion(logits, labels)
                epoch_test_loss.append(loss.cpu().detach().numpy())

                predictions.extend(predicted_labels)
                true_labels.extend(labels.cpu().numpy())

        test_accuracy = accuracy_score(true_labels, predictions)
        test_losses = np.mean(epoch_test_loss)
        print('Test Accuracy:', test_accuracy, 'Test Loss:', test_losses)
        return self.model, self.classifier
