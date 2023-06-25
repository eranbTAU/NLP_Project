import torch
import random
import os
import argparse

from utils import save_net, load_frompath
from loader_models import TextClassificationModel

# Config the parser
parser = argparse.ArgumentParser(description='Training Config', add_help=False)

# for Dataset & Path
parser.add_argument('--data_root', default=r'~/Documents/repose/nlp_project/NLP_Project/data/', metavar='DIR',
                    help='path to folder that include dataset')
parser.add_argument('--data_6class', default='ireland-news-headlines-cleaned-6.csv', metavar='DIR',
                    help='path to 6 class dataset')
parser.add_argument('--data_25class', default='ireland-news-headlines-cleaned-25.csv', metavar='DIR',
                    help='path to 25 class dataset')
parser.add_argument('--data_name', default='ireland_6c', metavar='DIR',
                    help='name to the require dataset. there is two options: [ireland_6c, ireland_25c]')
parser.add_argument('--unbalance', default='False', metavar='DIR',
                    help='working with unbalance data or balance')
parser.add_argument('--split_r', type=float, default=0.9, metavar='N',
                    help='Split the data, for example: 90% for training, 10% for testing')

# Hyperparameter
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=int, default=2e-5, metavar='N',
                    help='define the learning rate (default: 2e-5)')
parser.add_argument('--weight_decay', type=int, default=0.0067, metavar='N',
                    help='define the weight decay (default: 0.0067)')

# for Models
parser.add_argument('--model_name', type=str, default='distilbert', metavar='N',
                    help='The name of the chosen model')
parser.add_argument('--finetune', type=str, default='True', metavar='N',
                    help='type of pretrained model: finetune or end2end')

random.seed(42)

def main(args_config):
    # Load the dataset
    if args_config.data_name == 'ireland_6c':
        path2c = args_config.data_6class
    else:
        path2c = args_config.data_25class
    dir = os.path.join(args_config.data_root, path2c)
    df = load_frompath(dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Define device

    if args_config.unbalance == True:
        class_counts = df['category'].value_counts()
        class_counts = class_counts.values
        total_samples = sum(class_counts)
        class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    else:
        class_weights_tensor = None

    shuffled_data = df.sample(frac=1).reset_index(drop=True) # Shuffle the data
    # Define the train-test split ratio
    train_ratio = args_config.split_r
    split_index = int(train_ratio * len(shuffled_data)) # Calculate the split index
    # Split the data into train and test sets
    train_df = shuffled_data[:split_index]
    test_df = shuffled_data[split_index:]

    # Define the model & optimizer and loss function
    model = TextClassificationModel(args_config, df, class_weights_tensor, device)
    # Instantiate the tokenizer and model
    model.prepare_model()
    # prepare the data loaders
    model.prepare_data_loaders(train_df, test_df)
    # train the model
    model.trainModel()
    # evaluate the model
    model, classifier = model.evaluate()

    # save_net(./weights/model, model.state_dict())
    # save_net(./weights/classifier, classifier.state_dict())



if __name__ == '__main__':
    args_config = parser.parse_args()
    main(args_config)

