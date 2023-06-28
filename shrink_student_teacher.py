import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel

# Step 1: Import the necessary libraries
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random
from tqdm import tqdm
import numpy as np
import os
import re
import time
import datetime

random.seed(42)


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


def save_net(path, state, name=''):
    tt = str(time.asctime())
    img_name_save = name + '_net' + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save = img_name_save.replace(' ', '_') + '.pt'
    _dir = os.path.abspath('../test/')
    path = os.path.join(_dir, path)
    t = datetime.datetime.now()
    datat = t.strftime('%m/%d/%Y').replace('/', '_')
    dir = os.path.join(path, 'net' + '_' + datat)
    if not os.path.exists(dir):
        try:
            os.makedirs(dir, exist_ok=True)
            print("Directory '%s' created successfully" % ('net' + '_' + datat))
        except OSError as error:
            print("Directory '%s' can not be created" % ('net' + '_' + datat))

    net_path = os.path.join(dir, img_name_save)
    print(net_path)
    torch.save(state, net_path)
    return net_path


# Load the dataset
df = pd.read_csv('/fortest/ireland-news-headlines-cleaned-6.csv')
df.rename(columns={'clean_headline_text': 'text', 'headline_category': 'category'}, inplace=True)

df = df[['text', 'category']]  # Keep only 'text' and 'category' columns
class_counts = df['category'].value_counts()

df = df.dropna(subset=['text', 'category'])  # Remove rows with missing values in 'text' or 'category'
df.drop_duplicates(inplace=True)

# Shuffle the data
shuffled_data = df.sample(frac=1).reset_index(drop=True)

# Define the train-test split ratio
train_ratio = 0.8  # 80% for training, 20% for testing

# Calculate the split index
split_index = int(train_ratio * len(shuffled_data))

# Split the data into train and test sets
train_df = shuffled_data[:split_index]
test_df = shuffled_data[split_index:]

# Load the pre-trained BERT model as the teacher
teacher_model = BertModel.from_pretrained('bert-base-uncased')
teacher_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
teacher_classifer = nn.Linear(teacher_model.config.hidden_size, df['category'].nunique())
unique_classes = df['category'].unique()
print(unique_classes)

# Load model dicts
teacher_path = '/bert_6_bset_model/BERT_6.pt'
teacher_model.load_state_dict(torch.load(teacher_path))
teacher_model.eval()
teacher_classifer_path = '/bert_6_bset_model/Classifier_6.pt'
teacher_classifer.load_state_dict(torch.load(teacher_classifer_path))
teacher_classifer.eval()


# Define the student DistilBERT model
class DistilBERTStudent(nn.Module):
    def __init__(self, teacher_model, teacher_classifer, unique_classes):
        super(DistilBERTStudent, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.teacher_model = teacher_model
        self.teacher_classifer = teacher_classifer
        self.classifer = nn.Linear(teacher_model.config.hidden_size, unique_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            pooled_outputs = self.teacher_model(input_ids, attention_mask)
            teacher_outputs = self.teacher_classifer(pooled_outputs.last_hidden_state[:, 0])

        student_outputs = self.distilbert(input_ids, attention_mask)
        student_hidden_state = student_outputs.last_hidden_state[:, 0]
        student_logits = self.classifer(student_hidden_state)
        return student_logits, teacher_outputs


#  Instantiate the student model
student_model = DistilBERTStudent(teacher_model, teacher_classifer, df['category'].nunique())

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.fit_transform(test_df['category'])

# Create data loaders
train_dataset = IrishTimesDataset(train_df['text'], y_train, teacher_tokenizer)
test_dataset = IrishTimesDataset(test_df['text'], y_test, teacher_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)

#  Train the student model
num_epochs = 5  # Adjust the number of epochs as per your requirement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

teacher_model.to(device)
student_model.to(device)
teacher_classifer.to(device)

for epoch in range(num_epochs):
    student_model.train()

    pbar = tqdm(train_loader, total=len(train_loader))
    train_batch = []
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_logits, teacher_logits = student_model(input_ids, attention_mask)

        optimizer.zero_grad()
        loss = criterion(target_logits, teacher_logits)
        train_batch.append(loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()

        pbar.set_postfix({'Epoch': epoch,
                          'Training Loss': np.mean(train_batch)
                          })

        # # Evaluation
    student_model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            target_logits, teacher_logits = student_model(input_ids, attention_mask)

            predicted_labels = target_logits.argmax(dim=1).cpu().numpy()

            predictions.extend(predicted_labels)
            true_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(true_labels, predictions)
    print('Test Accuracy:', test_accuracy)

PATH = 'teacher_student'
save_net(PATH, student_model.state_dict(), name='student')
