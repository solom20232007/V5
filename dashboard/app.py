"""Dashboard Web (Flask) — تشغيل محلي:

    pip install flask
    python dashboard/app.py            # ثم افتح http://127.0.0.1:8050

بيقرا ملفات الـ CSV/JSON من جذر المشروع في كل طلب (بيانات لحظية).
/api/data يرجّع نفس البيانات JSON لو عايز تربطها بأداة تانية.
"""
import os
import sys

from flask import Flask, Response, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import collect, render  # noqa: E402

app = Flask(__name__)


@app.get("/")
def index():
    return Response(render(collect()), mimetype="text/html")


@app.get("/api/data")
def api_data():
    return jsonify(collect())


if __name__ == "__main__":
    app.run(host=os.getenv("DASH_HOST", "127.0.0.1"), port=int(os.getenv("DASH_PORT", "8050")))
