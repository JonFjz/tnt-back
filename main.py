
from flask import Flask, request, jsonify
import pandas as pd
from src.star_manager import Processor
from src.file_processor import FileProcessor
from src.services.model_service import ModelService
from src.services.mast_search_service import MastSearchService
import os

app = Flask(__name__)

# Initialize services
model_service = ModelService()
mast_search_service = MastSearchService()
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)



@app.get("/search")
def search_stars():
    """
    Search for stars using MAST catalog with filters matching your frontend form.
    """
    # Get parameters 
    ra = request.args.get("ra", type=float)
    dec = request.args.get("dec", type=float) 
    radius = request.args.get("radius", 15.0, type=float)
    mag_min = request.args.get("mag_min", 6.0, type=float)
    mag_max = request.args.get("mag_max", 15.0, type=float) 
    temp_min = request.args.get("temp_min", 3000.0, type=float)
    temp_max = request.args.get("temp_max", 7500.0, type=float)
    dist_min = request.args.get("dist_min", 10.0, type=float)
    dist_max = request.args.get("dist_max", 500.0, type=float)
    
    # Execute the search using MAST service
    result = mast_search_service.search_stars_with_filters(
        ra=ra,
        dec=dec, 
        radius=radius,
        mag_min=mag_min,
        mag_max=mag_max,
        temp_min=temp_min,
        temp_max=temp_max,
        dist_min=dist_min, 
        dist_max=dist_max
    )
    
    return jsonify(result)

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


