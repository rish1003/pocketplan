import pandas as pd
import random
from datasets import load_dataset

# load dataset
dataset = load_dataset("mitulshah/transaction-categorization", split="train")
df = dataset.to_pandas()

df = df[[
    "transaction_description",
    "category",
    "country",
    "currency"
]].dropna()

df = df.sample(50000, random_state=42)

BASE_RANGES = {
    "Food & Dining": (150, 2500),
    "Transportation": (100, 5000),
    "Shopping & Retail": (500, 15000),
    "Entertainment & Recreation": (200, 5000),
    "Healthcare & Medical": (500, 20000),
    "Utilities & Services": (800, 8000),
    "Financial Services": (500, 50000),
    "Income": (15000, 200000),
    "Government & Legal": (1000, 50000),
    "Charity & Donations": (100, 5000),
}

CURRENCY_SCALE = {
    "INR": 1,
    "USD": 80,
    "GBP": 100,
    "CAD": 60,
    "AUD": 55
}

CURRENCY_SYMBOL = {
    "USD": "$",
    "GBP": "£",
    "CAD": "$",
    "AUD": "$",
    "INR": "Rs"
}

FIRST_NAMES = [
    "Rahul", "Amit", "Priya", "Neha", "Arjun", "Sneha",
    "John", "Emily", "Michael", "Sarah", "David", "Emma",
    "Liam", "Olivia", "Noah", "Sophia",
    "James", "Charlotte", "Daniel", "Ava"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Patel",
    "Smith", "Johnson", "Brown", "Taylor",
    "Williams", "Jones", "Miller", "Davis",
    "Wilson", "Anderson", "Thomas", "Moore"
]

def generate_person_name():
    if random.random() < 0.5:
        return random.choice(FIRST_NAMES)
    else:
        return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def sample_amount(category, currency):
    low, high = BASE_RANGES.get(category, (500, 5000))
    scale = CURRENCY_SCALE.get(currency, 1)
    return round(random.uniform(low / scale, high / scale), 2)

def infer_category(amount, currency):
    best = None
    min_diff = float("inf")

    for cat, (low, high) in BASE_RANGES.items():
        scale = CURRENCY_SCALE.get(currency, 1)
        mean = ((low + high) / 2) / scale
        diff = abs(amount - mean)

        if diff < min_diff:
            min_diff = diff
            best = cat

    return best

def get_txn_type(category):
    if category == "Income":
        return "credit"
    elif category == "Financial Services":
        return random.choice(["debit", "credit"])
    else:
        return "debit"

def generate_sms(country, amount, receiver, currency, txn_type):
    ref = random.randint(10**10, 10**12)
    card = random.randint(1000, 9999)
    symbol = CURRENCY_SYMBOL[currency]

    if txn_type == "debit":
        templates = [
            f"Trx. of {symbol}{amount} on card ending *{card} at {receiver}, {country} is Approved",
            f"{symbol}{amount} spent on card *{card} at {receiver}",
            f"{symbol}{amount} debited on card ending *{card} at {receiver} Ref {ref}",
            f"Txn of {symbol}{amount} at {receiver} using card *{card} successful",
            f"{symbol}{amount} charged at {receiver} on card *{card} Ref {ref}"
        ]
    else:
        templates = [
            f"{symbol}{amount} credited to your account Ref {ref}",
            f"{symbol}{amount} received in A/C XXXX Ref {ref}",
            f"Credit of {symbol}{amount} received from {receiver}",
            f"{symbol}{amount} deposited successfully in your account",
            f"{symbol}{amount} credited via bank transfer Ref {ref}"
        ]

    return random.choice(templates)

def add_noise(text):
    if random.random() < 0.3:
        text = text.lower()

    text = text.replace("transaction", "txn")
    text = text.replace("account", "a/c")

    return text

def generate_sample(row):
    desc = row["transaction_description"]
    category = row["category"]
    country = row["country"]
    currency = row["currency"]

    amount = sample_amount(category, currency)

    if random.random() < 0.3:
        receiver = generate_person_name()
        category = infer_category(amount, currency)
    else:
        receiver = desc.upper()

    txn_type = get_txn_type(category)

    sms = generate_sms(
        country,
        amount,
        receiver,
        currency,
        txn_type
    )

    sms = add_noise(sms)

    return {
        "sms_text": sms,
        "amount": amount,
        "receiver": receiver,
        "category": category,
        "transaction_type": txn_type,
        "country": country,
        "currency": currency
    }

data = []

for _, row in df.iterrows():
    data.append(generate_sample(row))

final_df = pd.DataFrame(data)

final_df.to_csv("synthetic_sms.csv", index=False)

print(f"Generated {len(final_df)} synthetic SMS records.")
print(final_df.head())