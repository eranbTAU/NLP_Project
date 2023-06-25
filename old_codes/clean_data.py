import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import re

# Read the dataset
data = pd.read_csv('../data_eran/ireland-news-headlines.csv')  # Replace 'path_to_dataset.csv' with the actual path to your dataset file

# Clean the data_eran
data.dropna(subset=['headline_text'], inplace=True)  # Remove rows with missing text data_eran

# Remove special characters, numbers, and convert text to lowercase
data['clean_text'] = data['headline_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
data['clean_text'] = data['clean_text'].apply(lambda x: re.sub(r'\d+', '', x.lower()))

# Select the six classes with the most data_eran
class_counts = data['headline_category'].value_counts()
top_six_classes = class_counts.nlargest(6).index.tolist()
data = data[data['headline_category'].isin(top_six_classes)]

# Print the cleaned dataset summary
print("Cleaned Dataset Summary:")
print("Number of Rows:", len(data))
print("Number of Columns:", len(data.columns))
print("Classes:", data['headline_category'].unique())


# Tokenization and removing stopwords
stop_words = set(stopwords.words('english'))
data['tokenized_text'] = data['clean_text'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
data['stemmed_text'] = data['tokenized_text'].apply(lambda x: [stemmer.stem(word) for word in x])

# Create tokenizer
tokenizer = CountVectorizer().build_tokenizer()
data['tokenized_text_string'] = data['clean_text'].apply(tokenizer)

# Print a sample of cleaned and tokenized data
print("Sample of Cleaned and Tokenized Data:")
print(data[['clean_text', 'tokenized_text', 'stemmed_text']].head(5))

# Save the cleaned data
data.to_csv('cleaned_data.csv', index=False)  # Specify the desired file path and name for the cleaned data

print("Cleaned data saved successfully.")