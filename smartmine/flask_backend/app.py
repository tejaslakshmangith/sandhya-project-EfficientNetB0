"""
SmartMine Flask Backend — database layer and inference proxy.

Runs on port 5000. Proxies image uploads to the FastAPI EfficientNetB0
inference engine (port 8000), stores every prediction in a local SQLite
database, and exposes history/stats endpoints for the Next.js frontend.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "smartmine.db")
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")

ALLOWED_MIME_PREFIXES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set to Row."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the predictions table if it does not already exist."""
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                filename TEXT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept a multipart/form-data image upload, proxy it to the FastAPI
    inference engine, persist the result, and return enriched JSON.

    Form fields
    -----------
    file       : image file (required)
    session_id : optional UUID string; a new one is generated if absent
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    uploaded_file = request.files["file"]
    mime = uploaded_file.mimetype or ""

    if not any(mime.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        return jsonify({"error": f"Unsupported file type: {mime}. Please upload a JPG, PNG, WebP, GIF, BMP, or TIFF image."}), 400

    session_id = request.form.get("session_id") or str(uuid.uuid4())
    filename = uploaded_file.filename or "unknown"

    # -- proxy to FastAPI --
    file_bytes = uploaded_file.read()
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/predict",
            files={"file": (filename, file_bytes, mime)},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "AI inference engine is unavailable. Please start the FastAPI backend on port 8000."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "AI inference engine timed out."}), 503

    if not resp.ok:
        return jsonify({"error": f"Inference engine error {resp.status_code}: {resp.text}"}), resp.status_code

    inference = resp.json()
    predicted_class = inference.get("class", "unknown")
    confidence = float(inference.get("confidence", 0.0))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"

    # -- persist to DB --
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (session_id, filename, predicted_class, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, filename, predicted_class, confidence, timestamp),
        )
        prediction_id = cursor.lastrowid
        conn.commit()

    return jsonify(
        {
            "class": predicted_class,
            "confidence": confidence,
            "session_id": session_id,
            "prediction_id": prediction_id,
            "timestamp": timestamp,
        }
    )


@app.route("/api/history", methods=["GET"])
def history():
    """
    Return recent predictions ordered by newest first.

    Query params
    ------------
    session_id : filter to a specific session (optional)
    limit      : max results, default 20, capped at 100
    """
    session_id = request.args.get("session_id")
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        limit = 20

    with get_db() as conn:
        if session_id:
            rows = conn.execute(
                """
                SELECT id, session_id, filename, predicted_class AS class, confidence, timestamp
                FROM predictions
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, session_id, filename, predicted_class AS class, confidence, timestamp
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.route("/api/stats", methods=["GET"])
def stats():
    """
    Return aggregate prediction statistics.

    Response fields
    ---------------
    total          : total number of predictions
    safe           : count of 'safe' predictions
    unsafe         : count of 'unsafe' predictions
    avg_confidence : mean confidence across all predictions (0–1)
    today          : number of predictions made today (UTC)
    """
    today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN predicted_class = 'safe' THEN 1 ELSE 0 END) AS safe,
                SUM(CASE WHEN predicted_class = 'unsafe' THEN 1 ELSE 0 END) AS unsafe,
                AVG(confidence) AS avg_confidence,
                SUM(CASE WHEN timestamp LIKE ? THEN 1 ELSE 0 END) AS today
            FROM predictions
            """,
            (f"{today_prefix}%",),
        ).fetchone()

    return jsonify(
        {
            "total": row["total"] or 0,
            "safe": row["safe"] or 0,
            "unsafe": row["unsafe"] or 0,
            "avg_confidence": round(row["avg_confidence"] or 0.0, 4),
            "today": row["today"] or 0,
        }
    )


@app.route("/api/history/<int:prediction_id>", methods=["DELETE"])
def delete_prediction(prediction_id: int):
    """Delete a single prediction record by its integer primary key."""
    with get_db() as conn:
        result = conn.execute(
            "DELETE FROM predictions WHERE id = ?", (prediction_id,)
        )
        conn.commit()

    if result.rowcount == 0:
        return jsonify({"error": f"Prediction {prediction_id} not found"}), 404

    return jsonify({"deleted": prediction_id})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5000)
else:
    # Also initialise when imported by a WSGI server
    init_db()
