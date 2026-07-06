import sqlite3
from pathlib import Path

DB_PATH = Path("outputs/history.db")


def initialize_database():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS verification_history(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        timestamp TEXT,

        prediction TEXT,

        confidence REAL,

        similarity REAL,

        trust_score REAL,

        status TEXT,

        proof_hash TEXT

    )

    """)

    conn.commit()

    conn.close()


def save_verification(

    timestamp,

    prediction,

    confidence,

    similarity,

    trust_score,

    status,

    proof_hash

):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(

        """

        INSERT INTO verification_history(

        timestamp,

        prediction,

        confidence,

        similarity,

        trust_score,

        status,

        proof_hash

        )

        VALUES(?,?,?,?,?,?,?)

        """,

        (

            timestamp,

            prediction,

            confidence,

            similarity,

            trust_score,

            status,

            proof_hash

        )

    )

    conn.commit()

    conn.close()


def load_history():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(

        """

        SELECT *

        FROM verification_history

        ORDER BY id DESC

        """

    )

    rows = cursor.fetchall()

    conn.close()

    return rows


initialize_database()