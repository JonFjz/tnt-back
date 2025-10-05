
from flask import Flask, request, jsonify
import pandas as pd
from src.star_manager import Processor
from src.file_processor import FileProcessor
from src.services.model_service import ModelService
import os
app = Flask(__name__)


# manager = StarManager()
model_service = ModelService()
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)



@app.get("/analyze")
def analyze():
    mission = (request.args.get("mission") or "").strip()
    target_id = (request.args.get("id") or "").strip()
    Oilookup = request.args.get("Oilookup")
    parameters = request.args.get("parameters") or "{}"

    file = request.files.get("file")
    if file:
        file_processor = FileProcessor(file)
        
    response = Processor(mission, target_id, parameters, Oilookup, file_processor.file_path if file else None)
    
    #call drings function to analyze the data
    # call the ai model here to analyze the data
    return {"file_path": response.file_path, "response": response.response, "stellar": response.stellar, "products": response.products}

@app.route('/train-model', methods=['POST'])
def train_model():
    try:
        # Get request data
        data = request.get_json()

        # Call service function - don't pass request, just pass the data
        result = model_service.train_model_user(data)

        # Check if result is tuple with error status
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            return jsonify(result[0]), result[1]

        # Return successful result
        return jsonify(result)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get request data
        data = request.get_json()
        
        # Call service function
        result = model_service.predict(data)
        
        # Check if result is tuple with error status
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
            return jsonify(result[0]), result[1]
        
        # Return successful result
        return jsonify(result)
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=5000, debug=True)


