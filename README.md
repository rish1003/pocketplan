# PocketPlan

PocketPlan is a financial text understanding system for analyzing banking SMS messages and bank statement PDFs.

The system combines natural language processing, traditional machine learning, transformer-based models, named entity recognition, rule-based refinement, and PDF/OCR processing.

## Features

- Banking SMS transaction analysis
- Bank statement PDF processing
- OCR support for scanned PDFs
- Transaction amount extraction
- Merchant extraction
- Transaction type detection
- Transaction category classification
- Named Entity Recognition using DistilBERT
- DistilBERT-based transaction classification
- SVM and Logistic Regression baselines
- Live spending tracker
- Category-wise spending visualization
- Streamlit interface

## System Pipeline

1. User provides a banking SMS or uploads a bank statement PDF.
2. Input text is cleaned and preprocessed.
3. NER extracts entities such as amount, merchant, and transaction type.
4. Rule-based methods refine the extracted information.
5. The classification model predicts the transaction category.
6. Results are displayed through the Streamlit interface.
7. Processed transactions are maintained using Streamlit session state.
8. Spending and category-wise distributions are visualized.

## Models

The project uses:

- SVM for baseline transaction classification
- Logistic Regression for traditional ML classification
- DistilBERT for transaction classification
- DistilBERT-based NER for entity extraction

The trained model artifacts are stored separately in the `models/` directory.

## Dataset

The project generates a synthetic dataset of 50,000 SMS-like financial transaction records using transaction data sampled from the `mitulshah/transaction-categorization` dataset.

The generated records contain:

- SMS text
- Amount
- Receiver
- Category
- Transaction type
- Country
- Currency

## Results

| Model | Accuracy |
|---|---:|
| SVM | 82.01% |
| Logistic Regression | 86.93% |
| DistilBERT | 94.33% |

The NER model achieved an overall F1-score of 0.909.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas
- NumPy
- Streamlit
- pdfplumber
- pdf2image
- pytesseract

## Running the Application

Install the required dependencies:

```bash
pip install -r requirements.txt