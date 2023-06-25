
def PreTrmodel(args_config, size):

    if args_config.model_name == "distilbert":
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
        return model, tokenizer
    elif args_config.model_name == "bert":
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        return model, tokenizer
    elif args_config.model_name == "gpt2":
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('bert-base-uncased')
        return model, tokenizer
    elif args_config.model_name == "roberta":
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        return model, tokenizer
    elif args_config.model_name == "TransformerXL":
        from transformers import TransfoXLTokenizer, TransfoXLModel
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
        return model, tokenizer


