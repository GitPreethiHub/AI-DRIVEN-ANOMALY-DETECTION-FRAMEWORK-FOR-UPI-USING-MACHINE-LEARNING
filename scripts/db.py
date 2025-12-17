import sqlite3
from datetime import datetime

DB_NAME = "transactions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            step INTEGER,
            type INTEGER,
            amount REAL,
            oldbalanceOrg REAL,
            newbalanceOrig REAL,
            oldbalanceDest REAL,
            newbalanceDest REAL,
            fraud_prediction INTEGER,
            fraud_probability REAL,
            action TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_transaction(data: dict):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO transactions (
            timestamp, step, type, amount, oldbalanceOrg, newbalanceOrig,
            oldbalanceDest, newbalanceDest, fraud_prediction, fraud_probability, action
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        data["step"],
        data["type"],
        data["amount"],
        data["oldbalanceOrg"],
        data["newbalanceOrig"],
        data["oldbalanceDest"],
        data["newbalanceDest"],
        data["fraud_prediction"],
        data["fraud_probability"],
        data["action"]
    ))

    conn.commit()
    conn.close()
init_db()