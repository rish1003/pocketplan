import streamlit as st
import torch
import pdfplumber
import pandas as pd
import re

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)

from pdf2image import convert_from_bytes
import pytesseract
import io
import random

st.set_page_config(
    page_title="Financial Analyzer"
)

st.title(
    "Financial SMS + Statement Analyzer"
)

if "all_transactions" not in st.session_state:
    st.session_state.all_transactions = []


@st.cache_resource
def load_models():

    ner_tokenizer = AutoTokenizer.from_pretrained(
        "models/ner_model_final"
    )

    ner_model = AutoModelForTokenClassification.from_pretrained(
        "models/ner_model_final"
    )

    clf_tokenizer = AutoTokenizer.from_pretrained(
        "models/bert_classifier_final"
    )

    clf_model = AutoModelForSequenceClassification.from_pretrained(
        "models/bert_classifier_final"
    )

    return (
        ner_tokenizer,
        ner_model,
        clf_tokenizer,
        clf_model
    )


ner_tok, ner_model, clf_tok, clf_model = load_models()

id2label_ner = {
    int(k): v
    for k, v in ner_model.config.id2label.items()
}

id2label_clf = {
    int(k): v
    for k, v in clf_model.config.id2label.items()
}


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"(\d)([a-zA-Z])",
        r"\1 \2",
        text
    )

    text = re.sub(
        r"([a-zA-Z])(\d)",
        r"\1 \2",
        text
    )

    text = text.replace(
        "rs.",
        "rs "
    )

    text = re.sub(
        r"dear .*?,",
        "",
        text
    )

    text = re.sub(
        r"ref no \d+",
        "",
        text
    )

    text = re.sub(
        r"\b\d{2}-\d{2}-\d{2}\b",
        "",
        text
    )

    text = re.sub(
        r"x{2,}\d+",
        "",
        text
    )

    return text.strip()


def detect_type(text):

    text = text.lower()

    if any(
        w in text
        for w in [
            "credited",
            "credit",
            "cr",
            "received"
        ]
    ):
        return "credit"

    if any(
        w in text
        for w in [
            "debited",
            "debit",
            "dr",
            "spent",
            "paid"
        ]
    ):
        return "debit"

    return ""


def extract_merchant_rule(text):

    text = text.lower()

    match = re.search(
        r"(from|at)\s+([a-zA-Z]+)",
        text
    )

    if match:
        return match.group(2)

    for m in [
        "amazon",
        "flipkart",
        "meesho",
        "swiggy",
        "zomato",
        "uber"
    ]:
        if m in text:
            return m

    return ""


def clean_amount(amount):

    amount = amount.replace(
        "rs",
        ""
    ).strip()

    amount = re.sub(
        r"[^\d.]",
        "",
        amount
    )

    if "." in amount:
        amount = amount.split(".")[0]

    return amount


def classify(text):

    text = clean_text(text)

    inputs = clf_tok(
        text,
        return_tensors="pt",
        truncation=True
    )

    inputs.pop(
        "token_type_ids",
        None
    )

    with torch.no_grad():
        outputs = clf_model(**inputs)

    pred = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    return id2label_clf[pred]


def ner_predict(text):

    text = clean_text(text)

    inputs = ner_tok(
        text,
        return_tensors="pt",
        truncation=True
    )

    inputs.pop(
        "token_type_ids",
        None
    )

    with torch.no_grad():
        outputs = ner_model(**inputs)

    preds = torch.argmax(
        outputs.logits,
        dim=2
    )

    tokens = ner_tok.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    labels = preds[0].tolist()

    return [
        (tok, id2label_ner[label])
        for tok, label in zip(tokens, labels)
    ]


def extract_entities(decoded):

    amount_tokens = []
    merchant_tokens = []
    type_tokens = []

    current_label = None

    for tok, label in decoded:

        if tok in ["[CLS]", "[SEP]"]:
            continue

        if tok.startswith("##"):

            tok = tok[2:]

            if current_label == "AMOUNT" and amount_tokens:
                amount_tokens[-1] += tok

            elif current_label == "MERCHANT" and merchant_tokens:
                merchant_tokens[-1] += tok

            elif current_label == "TYPE" and type_tokens:
                type_tokens[-1] += tok

            continue

        if label.startswith("B-") or label.startswith("I-"):
            current_label = label.split("-")[1]
        else:
            current_label = None

        if current_label == "AMOUNT":
            amount_tokens.append(tok)

        elif current_label == "MERCHANT":
            merchant_tokens.append(tok)

        elif current_label == "TYPE":
            type_tokens.append(tok)

    return (
        " ".join(amount_tokens).strip(),
        " ".join(merchant_tokens).strip(),
        " ".join(type_tokens).strip()
    )


def extract_merchant_upi(line):

    parts = line.split("/")

    if len(parts) >= 4:

        merchant = parts[3]

        merchant = re.sub(
            r"[^a-zA-Z]",
            "",
            merchant
        )

        return merchant.lower()

    return ""


tab1, tab2 = st.tabs([
    " SMS",
    "PDF"
])

with tab1:

    st.info(
        "Enter a transaction SMS"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Use Example"):

            sms_input_opts = [
                "ALERT: INR 899.00 spent on your card XX1234 at AMAZON SHOPPING on 05-Nov Ref#112233",

                "Dear UPI User, your A/c XXXXXX0205 debit by Rs.291.00 on 03-02-26 transfer from Uber Ref No 640008628633 -SBI",

                "Dear Customer, your A/c XXXXXX8765 is DEBITED by Rs. 5000.00 on 01-11-2023 from HOSPITAL Ref No 123456789",

                "Rs107364.54 received in A/C XXXX Ref 273853009199"
            ]

            st.session_state.sms_input = sms_input_opts[
                random.randint(
                    0,
                    len(sms_input_opts) - 1
                )
            ]

    with col2:

        analyze = st.button(
            "Analyze SMS"
        )

    sms = st.text_area(
        "SMS Input",
        value=st.session_state.get(
            "sms_input",
            ""
        )
    )

    if analyze and sms.strip():

        cleaned_text = clean_text(sms)

        decoded = ner_predict(sms)

        amount, merchant, txn_type = extract_entities(
            decoded
        )

        txn_type = detect_type(
            cleaned_text
        )

        merchant = extract_merchant_rule(
            cleaned_text
        )

        amount = clean_amount(
            amount
        )

        category = classify(sms)

        if txn_type != "credit" and amount:

            try:

                st.session_state.all_transactions.append({
                    "Amount": float(
                        amount.replace(",", "")
                    ),
                    "Category": category
                })

            except:
                pass

        st.subheader("Result")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Amount",
                amount or "-"
            )

            st.metric(
                "Type",
                txn_type or "-"
            )

        with c2:

            st.metric(
                "Merchant",
                merchant or "-"
            )

            st.metric(
                "Category",
                category
            )


with tab2:

    st.info(
        "Upload bank statement (works for text + scanned PDFs)"
    )

    file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if file:

        file_bytes = file.read()

        with st.spinner(
            "Processing PDF..."
        ):

            text = ""

            try:

                with pdfplumber.open(
                    io.BytesIO(file_bytes)
                ) as pdf:

                    for page in pdf.pages:

                        page_text = page.extract_text()

                        if page_text:
                            text += page_text

            except:
                pass

            if not text.strip():

                images = convert_from_bytes(
                    file_bytes,
                    poppler_path=r"C:\Program Files (x86)\poppler-25.12.0\Library\bin"
                )

                for img in images:
                    text += pytesseract.image_to_string(
                        img
                    ) + "\n"

            lines = text.split("\n")

            transactions = []

            buffer = ""

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                if "UPI/" in line:

                    if buffer:
                        transactions.append(
                            buffer
                        )

                    buffer = line

                else:

                    buffer += " " + line

            if buffer:
                transactions.append(
                    buffer
                )

            results = []

            for txn in transactions:

                cleaned = clean_text(txn)

                decoded = ner_predict(txn)

                amount, merchant, txn_type = extract_entities(
                    decoded
                )

                txn_type = detect_type(
                    cleaned
                )

                merchant = extract_merchant_rule(
                    cleaned
                )

                if not merchant and "upi" in txn.lower():
                    merchant = extract_merchant_upi(txn)

                numbers = re.findall(
                    r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b",
                    txn
                )

                if numbers:

                    for num in numbers:

                        val = float(
                            num.replace(",", "")
                        )

                        if val < 100000:

                            amount = str(val)

                            break

                else:
                    amount = ""

                category = classify(txn)

                if amount:

                    try:

                        st.session_state.all_transactions.append({
                            "Amount": float(
                                amount.replace(",", "")
                            ),
                            "Category": category
                        })

                    except:
                        pass

                results.append({
                    "Text": txn,
                    "Amount": amount,
                    "Merchant": merchant,
                    "Type": txn_type,
                    "Category": category
                })

            if results:

                st.success(
                    f"Found {len(results)} transactions"
                )

                st.dataframe(
                    pd.DataFrame(results)
                )

            else:

                st.warning(
                    "No transactions detected"
                )


st.markdown("---")

st.subheader(
    "Live Spending Tracker"
)

if st.session_state.all_transactions:

    df = pd.DataFrame(
        st.session_state.all_transactions
    )

    total = df["Amount"].sum()

    st.metric(
        "Total Spend",
        f" ₹{total:,.2f}"
    )

    cat_df = df.groupby(
        "Category"
    )["Amount"].sum().reset_index()

    st.plotly_chart({
        "data": [{
            "labels": cat_df["Category"],
            "values": cat_df["Amount"],
            "type": "pie"
        }]
    })

else:

    st.info(
        "No transactions yet. Analyze SMS or upload PDF."
    )