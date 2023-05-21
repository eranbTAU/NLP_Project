import torch
from sklearn.metrics import accuracy_score

def train_val(args_config, optimizer, model, train_loader, val_loader, device, label_encoder):

    for epoch in range(args_config.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels.extend(batch['labels'])
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_labels = logits.argmax(dim=1).cpu().numpy()
                predictions.extend(predicted_labels)

        # predicted_classes = label_encoder.inverse_transform(predictions)
        # labels_classes = label_encoder.inverse_transform(labels)

        val_accuracy = accuracy_score(labels, predictions)
        print('Validation Accuracy:', val_accuracy)

    return model


def evaluation(model, test_loader, device):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels.extend(batch['labels'])
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = logits.argmax(dim=1).cpu().numpy()
            predictions.extend(predicted_labels)

    test_accuracy = accuracy_score(labels, predictions)
    print('Test Accuracy:', test_accuracy)
