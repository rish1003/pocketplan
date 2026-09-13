from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset

clf_df = final_df[["sms_text", "category"]]

label_list = clf_df["category"].unique().tolist()

label2id = {
    l: i
    for i, l in enumerate(label_list)
}

id2label = {
    i: l
    for l, i in label2id.items()
}

clf_df["label"] = clf_df["category"].map(label2id)

dataset = Dataset.from_pandas(
    clf_df[["sms_text", "label"]]
)

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(example):
    return tokenizer(
        example["sms_text"],
        truncation=True
    )

dataset = dataset.map(tokenize)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted"
    )

    acc = accuracy_score(
        labels,
        preds
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

from transformers import DataCollatorWithPadding, Trainer

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

final_path = "models/bert_classifier_final"

trainer.save_model(final_path)

tokenizer.save_pretrained(final_path)

import json

with open(
    f"{final_path}/label_map.json",
    "w"
) as f:
    json.dump({
        "label2id": label2id,
        "id2label": id2label
    }, f)

metrics = trainer.evaluate()

print(metrics)

import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

# Predictions
predictions = trainer.predict(test_dataset)

y_pred = np.argmax(
    predictions.predictions,
    axis=1
)

y_true = predictions.label_ids

# Accuracy
accuracy = accuracy_score(
    y_true,
    y_pred
)

print(
    f"Accuracy: {accuracy:.4f}"
)

# Full report
print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=label_list
    )
)

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    xticks_rotation=45
)

plt.show()