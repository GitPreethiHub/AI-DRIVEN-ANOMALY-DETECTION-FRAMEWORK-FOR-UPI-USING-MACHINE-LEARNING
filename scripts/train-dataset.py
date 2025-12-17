import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
DATA_PATH = r"C:\Users\Hp\OneDrive\Desktop\UPI-detection\AI-DRIVEN-ANOMALY-DETECTION-FRAMEWORK-FOR-UPI-USING-MACHINE-LEARNING\data\upi_fraud_dataset.csv"
OUTPUT_PATH = r"C:\Users\Hp\OneDrive\Desktop\UPI-detection\AI-DRIVEN-ANOMALY-DETECTION-FRAMEWORK-FOR-UPI-USING-MACHINE-LEARNING\data\train_dataset.csv"

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------
print("Loading dataset...")
dataset = pd.read_csv(DATA_PATH)
dataset.columns = dataset.columns.str.strip()
print("Original shape:", dataset.shape)
print("Columns:", dataset.columns.tolist())

# --------------------------------------------------
# 3. CLASS DISTRIBUTION
# --------------------------------------------------
fraud = dataset[dataset["fraud_risk"] == 1]
valid = dataset[dataset["fraud_risk"] == 0]

print("Fraud records:", len(fraud))
print("Valid records:", len(valid))

# --------------------------------------------------
# 4. BALANCE DATASET (AUTO UNDER / OVER SAMPLING)
# --------------------------------------------------
fraud_count = len(fraud)
valid_count = len(valid)

replace = valid_count < fraud_count
valid_sampled = valid.sample(
    fraud_count,
    replace=replace,
    random_state=42
)

dataset = pd.concat([valid_sampled, fraud], axis=0)
dataset.reset_index(drop=True, inplace=True)

print("Balanced dataset shape:", dataset.shape)

# --------------------------------------------------
# 5. TIME FEATURES
# --------------------------------------------------
# If datetime exists, derive features
if "trans_datetime" in dataset.columns:
    dataset["trans_datetime"] = pd.to_datetime(dataset["trans_datetime"])
    dataset["trans_hour"] = dataset["trans_datetime"].dt.hour
    dataset["trans_day"] = dataset["trans_datetime"].dt.day
    dataset["trans_month"] = dataset["trans_datetime"].dt.month
    dataset["trans_year"] = dataset["trans_datetime"].dt.year
else:
    # If already present, ensure they exist
    for col in ["trans_hour", "trans_day", "trans_month", "trans_year"]:
        if col not in dataset.columns:
            dataset[col] = 0

# --------------------------------------------------
# 6. AGE CALCULATION
# --------------------------------------------------
if "age" not in dataset.columns:
    if "dob" in dataset.columns and "trans_datetime" in dataset.columns:
        dataset["dob"] = pd.to_datetime(dataset["dob"])
        dataset["age"] = np.round(
            (dataset["trans_datetime"] - dataset["dob"]) / np.timedelta64(1, "Y")
        )
    else:
        dataset["age"] = 0

# --------------------------------------------------
# 7. DROP REDUNDANT / IDENTITY COLUMNS (SAFE)
# --------------------------------------------------
columns_to_drop = [
    "trans_id",
    "Id",
    "trans_datetime",
    "merchant",
    "card_holder_name",
    "gender",
    "dob",
    "upi_number"
]

dataset.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# --------------------------------------------------
# 8. LABEL ENCODING (CATEGORICAL)
# --------------------------------------------------
labelencoder_cat = LabelEncoder()
labelencoder_state = LabelEncoder()

if dataset["category"].dtype == "object":
    dataset["category"] = labelencoder_cat.fit_transform(dataset["category"])

if dataset["state"].dtype == "object":
    dataset["state"] = labelencoder_state.fit_transform(dataset["state"])

# --------------------------------------------------
# 9. FINAL SANITY CHECK
# --------------------------------------------------
print("\nFinal dataset info:")
print(dataset.info())
print("\nClass distribution:")
print(dataset["fraud_risk"].value_counts())

# --------------------------------------------------
# 10. SAVE FINAL DATASET
# --------------------------------------------------
dataset.to_csv(OUTPUT_PATH, index=False)
print("\nTraining dataset saved at:")
print(OUTPUT_PATH)
