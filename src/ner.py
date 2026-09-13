def clean_tokens(tokens):
    result = []

    for t in tokens:
        if t.startswith("##") and result:
            result[-1] += t[2:]
        else:
            result.append(t)

    return " ".join(result)


import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)


def extract_entities(text):
    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    amount, merchant, txn_type = [], [], []

    for token, pred in zip(
        tokens,
        predictions[0]
    ):
        label = id2label[pred.item()]

        if token in ["[CLS]", "[SEP]"]:
            continue

        if label in ["B-AMOUNT", "I-AMOUNT"]:
            amount.append(token)

        elif label == "B-MERCHANT":
            merchant.append(token)

        elif label == "B-TYPE":
            txn_type.append(token)

    return {
        "amount": clean_tokens(amount).upper(),
        "merchant": clean_tokens(merchant).upper(),
        "type": clean_tokens(txn_type).lower()
    }


extract_entities(
    "Rs 1200 debited on card ending *1234 at AMAZON Ref 987654"
)

metrics = ner_trainer.evaluate()

print(metrics)

from seqeval.metrics import classification_report, f1_score
import numpy as np

predictions = ner_trainer.predict(test_dataset)

preds = np.argmax(
    predictions.predictions,
    axis=2
)

labels = predictions.label_ids

true_labels = []
true_preds = []

for pred, lab in zip(preds, labels):
    curr_true = []
    curr_pred = []

    for p, l in zip(pred, lab):
        if l != -100:
            curr_true.append(id2label[l])
            curr_pred.append(id2label[p])

    true_labels.append(curr_true)
    true_preds.append(curr_pred)

print(
    "F1 Score:",
    f1_score(true_labels, true_preds)
)

print("\nFull Report:\n")
print(
    classification_report(
        true_labels,
        true_preds
    )
)