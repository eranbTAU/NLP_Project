import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random
from tqdm import tqdm
import os
from transformers import BertModel, BertTokenizer
import torch.quantization

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
train_ratio = 0.9  # 80% for training, 20% for testing

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

# Load model dicts
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
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(
    [{'params': model.parameters()}, {'params': classifer.parameters()}],
    lr=learning_rate
)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

model = torch.quantization.quantize_dynamic(
    model.to('cpu'), {nn.Linear}, dtype=torch.qint8
)

# Save the quantized model
quantized_model_path = 'quantized_model.pt'
torch.save(model.state_dict(), quantized_model_path)

# Print the size of the quantized model
print("Size of quantized model: %.2f MB" % (os.path.getsize(quantized_model_path) / (1024 * 1024)))


# Save the entire quantized model
quantized_full_model_path = 'quantized_full_model.pt'
torch.save(model, quantized_full_model_path)


# Evaluate the quantized model
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = classifer(pooled_output)
        predicted_labels = logits.argmax(dim=1).cpu().numpy()

        predictions.extend(predicted_labels)
        true_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(true_labels, predictions)
print('Test Accuracy (Quantized Model):', test_accuracy)

