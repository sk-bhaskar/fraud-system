import pandas as pd
import requests
import time
import random

df = pd.read_csv("data/raw/creditcard.csv")

normal_df = df[df["Class"] == 0].drop("Class", axis=1)
fraud_df = df[df["Class"] == 1].drop("Class", axis=1)

print(f"Normal transactions available: {len(normal_df)}")
print(f"Fraud transactions available: {len(fraud_df)}")
print("-" * 60)

# Send 5 normal transactions
print("Sending NORMAL transactions...\n")
for i in range(5):
    row = normal_df.sample(1, random_state=random.randint(1, 100000)).iloc[0].tolist()

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": row}
    )

    print(f"Normal Transaction {i+1}: {response.json()}")
    time.sleep(1)

print("\n" + "-" * 60)
print("Sending FRAUD transactions...\n")

# Send 5 fraud transactions
for i in range(5):
    row = fraud_df.sample(1, random_state=random.randint(1, 100000)).iloc[0].tolist()

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"features": row}
    )

    print(f"Fraud Transaction {i+1}: {response.json()}")
    time.sleep(1)