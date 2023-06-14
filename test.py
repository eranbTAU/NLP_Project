import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random
from tqdm import tqdm
import numpy as np

random.seed(42)

# Load the dataset
df = pd.read_csv('/home/roblab20/Documents/NLP_Project/fortest/ireland-news-headlines-cleaned-25.csv')
df.rename(columns={'clean_headline_text': 'text', 'headline_category': 'category'}, inplace=True)

df = df[['text', 'category']]  # Keep only 'text' and 'category' columns

# Clean the data (optional, modify as per your requirements)
df = df.dropna(subset=['text', 'category'])  # Remove rows with missing values in 'text' or 'category'
df.drop_duplicates(inplace=True)

# df = df.iloc[:2000]

# Shuffle the data
shuffled_data = df.sample(frac=1).reset_index(drop=True)

# Define the train-test split ratio
train_ratio = 0.9  # 80% for training, 20% for testing

# Calculate the split index
split_index = int(train_ratio * len(shuffled_data))

# Split the data into train and test sets
train_df = shuffled_data[:split_index]
test_df = shuffled_data[split_index:]


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

# Instantiate the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
classifier = nn.Linear(model.config.hidden_size, df['category'].nunique())
unique_classes = df['category'].unique()
print(unique_classes)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['category'])
y_test = label_encoder.fit_transform(test_df['category'])

# Create data loaders
train_dataset = IrishTimesDataset(train_df['text'], y_train, tokenizer)
test_dataset = IrishTimesDataset(test_df['text'], y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define training parameters
epochs = 5
learning_rate = 2e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and classifier to the device
model.to(device)
classifier.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(
    [{'params': model.parameters()}, {'params': classifier.parameters()}],
    lr=learning_rate
)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
classifier.train()

for epoch in range(epochs):
    pbar = tqdm(train_loader, total=len(train_loader))
    train_batch = []
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = classifier(pooled_output)

        loss = criterion(logits, labels)
        train_batch.append(loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()

        pbar.set_postfix({'Epoch': epoch,
                          'Training Loss': np.mean(train_batch)
                          })

# # Evaluation
    model.eval()
    classifier.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            logits = classifier(pooled_output)
            predicted_labels = logits.argmax(dim=1).cpu().numpy()

            predictions.extend(predicted_labels)
            true_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(true_labels, predictions)
    print('Test Accuracy:', test_accuracy)



