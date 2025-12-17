import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------
# Use the prepared training dataset
DATA_PATH = r"C:\Users\Hp\OneDrive\Desktop\UPI-detection\AI-DRIVEN-ANOMALY-DETECTION-FRAMEWORK-FOR-UPI-USING-MACHINE-LEARNING\data\train_dataset.csv"

print("Loading dataset from:", DATA_PATH)
dataset = pd.read_csv(DATA_PATH, index_col=0)
print("Dataset shape:", dataset.shape)

# --------------------------------------------------
# 2. SPLIT FEATURES & LABEL
# --------------------------------------------------
# train_dataset.csv columns:
# trans_hour,trans_day,trans_month,trans_year,category,age,trans_amount,state,zip,fraud_risk
# → 9 feature columns + 1 target column (fraud_risk)
X = dataset.iloc[:, :-1].values  # all columns except the last as features
y = dataset.iloc[:, -1].values   # last column as label

# --------------------------------------------------
# 3. TRAIN–TEST SPLIT
# --------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=0, stratify=y
)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

fraud = np.count_nonzero(y_train == 1)
valid = np.count_nonzero(y_train == 0)
print("Fraud cases:", fraud)
print("Valid cases:", valid)

# --------------------------------------------------
# 4. FEATURE SCALING
# --------------------------------------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# --------------------------------------------------
# 5. MODEL TRAINING
# --------------------------------------------------
scores = {}

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=0)
lr.fit(x_train, y_train)
scores["Logistic Regression"] = accuracy_score(y_test, lr.predict(x_test))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
scores["KNN"] = accuracy_score(y_test, knn.predict(x_test))

# SVM
svm = SVC(kernel="linear", random_state=0)
svm.fit(x_train, y_train)
scores["SVM"] = accuracy_score(y_test, svm.predict(x_test))

# Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
scores["Naive Bayes"] = accuracy_score(y_test, nb.predict(x_test))

# Decision Tree
dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
dt.fit(x_train, y_train)
scores["Decision Tree"] = accuracy_score(y_test, dt.predict(x_test))

# --------------------------------------------------
# 6. RANDOM FOREST (DEPLOYMENT MODEL)
# --------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    class_weight="balanced",
    random_state=0,
    n_jobs=-1
)

rf.fit(x_train, y_train)
scores["Random Forest"] = accuracy_score(y_test, rf.predict(x_test))

# --------------------------------------------------
# 7. NEURAL NETWORK (FOR COMPARISON ONLY)
# --------------------------------------------------
input_dim = X.shape[1]

nn = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

nn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

nn.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)
loss, acc_nn = nn.evaluate(x_test, y_test, verbose=0)
scores["Neural Network (MLP)"] = acc_nn

# --------------------------------------------------
# 8. ACCURACY COMPARISON & BEST MODEL SELECTION
# --------------------------------------------------
results_df = pd.DataFrame({
    "Algorithm": scores.keys(),
    "Accuracy (%)": [v * 100 for v in scores.values()]
}).sort_values("Accuracy (%)", ascending=False)

print("\nAccuracy Comparison:")
print(results_df)

best_algorithm = results_df.iloc[0]["Algorithm"]
best_accuracy = results_df.iloc[0]["Accuracy (%)"]
print(f"\nBest-performing algorithm on this dataset: {best_algorithm} "
      f"with accuracy {best_accuracy:.2f}%")

# --------------------------------------------------
# 9. VISUALIZATION
# --------------------------------------------------
plt.figure(figsize=(10, 5))
sns.barplot(x="Accuracy (%)", y="Algorithm", data=results_df)
plt.title("UPI Fraud Detection Model Comparison")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 10. SAVE RANDOM FOREST + SCALER (FINAL OUTPUT)
# --------------------------------------------------
MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_upi_fraud_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

joblib.dump(rf, RF_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nRandom Forest model saved at:", RF_MODEL_PATH)
print("Scaler saved at:", SCALER_PATH)
