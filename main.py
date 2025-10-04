
from flask import Flask, request, jsonify
import pandas as pd
from src.star_manager import StarManager

app = Flask(__name__)
manager = StarManager()

@app.get("/stars")
def get_star_lightcurve():
    
    mission = (request.args.get("mission") or "").strip()
    target_id = (request.args.get("id") or "").strip()
    df = manager.fetch_lightcurve(mission, target_id)
    return jsonify(df.to_dict(orient="records"))




if __name__ == "__main__":
    # For local dev only
    app.run(host="127.0.0.1", port=5000, debug=True)


