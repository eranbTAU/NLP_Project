from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import pipeline
import transformers
from transformers import BertTokenizer, BertForSequenceClassification


def PreTrmodel(args_config, size):

    if args_config.model_name == "distilbert":
        model_name = 'distilbert-base-uncased'
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
        model = transformers.DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=size)
        return model, tokenizer
    elif args_config.model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        return model, tokenizer


