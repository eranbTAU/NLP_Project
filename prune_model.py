import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random
from tqdm import tqdm
import os
import re
import time
import datetime
from transformers import BertModel, BertTokenizer
import torch.nn.utils.prune as prune

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


def save_net(path, state):
    tt = str(time.asctime())
    img_name_save = 'net' + " " + str(re.sub('[:!@#$]', '_', tt))
    img_name_save = img_name_save.replace(' ', '_') + '.pt'
    _dir = os.path.abspath('../')
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


def prune_model(model, pruning_method, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            pruning_method.apply(module, 'weight', amount)


# Load the dataset
df = pd.read_csv('/fortest/ireland-news-headlines-cleaned-6.csv')
df.rename(columns={'clean_headline_text': 'text', 'headline_category': 'category'}, inplace=True)

df = df[['text', 'category']]  # Keep only 'text' and 'category' columns
class_counts = df['category'].value_counts()

class_counts = class_counts.values
total_samples = sum(class_counts)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]

# Clean the data (optional, modify as per your requirements)
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
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classifer = nn.Linear(model.config.hidden_size, df['category'].nunique())
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

unique_classes = df['category'].unique()
print(unique_classes)

# load pre-trained model dicts
path = '/bert_6_bset_model/BERT_6.pt'
model.load_state_dict(torch.load(path))
model.eval()
classifer_path = '/bert_6_bset_model/Classifier_6.pt'
classifer.load_state_dict(torch.load(classifer_path))
classifer.eval()

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
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Move model and classifer to the device
model.to(device)
classifer.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(
    [{'params': model.parameters()}, {'params': classifer.parameters()}],
    lr=learning_rate
)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Apply pruning
pruning_method = prune.L1Unstructured
amount = 0.5  # Prune 50% of the weights
prune_model(model, pruning_method, amount)

# Training loop
model.train()
classifer.train()

for epoch in range(epochs):
    # # Evaluation
    model.eval()
    classifer.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]
            logits = classifer(pooled_output)
            predicted_labels = logits.argmax(dim=1).cpu().numpy()

            predictions.extend(predicted_labels)
            true_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(true_labels, predictions)
    print('Test Accuracy:', test_accuracy)

PATH = '../weights_prune_model'
save_net(PATH, model.state_dict())
