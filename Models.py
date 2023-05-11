from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


def PreTrmodel(args_config):
    config = AutoConfig.from_pretrained(args_config.model_name)
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained(args_config.model_name,
                                                                                  num_labels=2)
    return model_seq_classification
