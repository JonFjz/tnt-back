
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from src.star_manager import StarProcessor
from src.file_processor import FileProcessor
from src.services.model_service import ModelService
from src.services.mast_search_service import MastSearchService
from src.data_mapper import build_model_payload_from_row
from src.exoplanet_processor import ExoplanetParameterProcessor
from src.exoplanet_paramter_processor import ExoplanetParameterProcessorTEST
from src.data_mapper_manual import build_payload
import os
import math, json
from flask import jsonify
import numpy as np

app = Flask(__name__)

# Initialize services
model_service = ModelService()
mast_search_service = MastSearchService()
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://tnt.thot.info:5173"
).split(",")

# one line CORS: no credentials, allow GET/POST/OPTIONS
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})



def _to_native(v):
    # Convert numpy scalars → Python
    if np is not None:
        if isinstance(v, (np.generic,)):
            return v.item()
    return v

def _clean_value(v):
    v = _to_native(v)
    # Replace NaN/±Inf → None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v

def deep_clean(obj):
    if isinstance(obj, dict):
        return {k: deep_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [deep_clean(v) for v in obj]
    return _clean_value(obj)


@app.route("/analyze", methods=["GET"])
def analyze():
    

    mission = (request.args.get("mission") or "").strip()
    target_id = (request.args.get("id") or "").strip()
    oi_lookup = request.args.get("oi_lookup", default=1, type=int)
    parameters = request.args.get("parameters") or "{}"
    optimization_type = (request.args.get("optimization_type") or "recall").strip()
    model_name = (request.args.get("model_name") or "default_model").strip()

    file = request.files.get("file")
    if file:
        file_processor = FileProcessor(file)
        
    sp = StarProcessor(mission, target_id, oi_lookup, parameters,  file_processor.file_path if file else None)
    output_json = None
    stellar_all_data = sp.stellar

    if sp.manualSearch:
        stellar_data = sp.stellar["stellar"] if sp.stellar else {}
       
        processor = ExoplanetParameterProcessor(
            fits_path=sp.file_path,
            mission=mission,
            catalog={
                'st_teff': stellar_data.get('st_teff'),
                'st_tefferr1': stellar_data.get('st_tefferr1'),
                'st_tefferr2': stellar_data.get('st_tefferr2'),
                'st_rad': stellar_data.get('st_rad'),
                'st_raderr1': stellar_data.get('st_raderr1'),
                'st_raderr2': stellar_data.get('st_raderr2'),
                'st_mass': stellar_data.get('st_mass'),
                'st_masserr1': stellar_data.get('st_masserr1'),
                'st_masserr2': stellar_data.get('st_masserr2'),
                'st_logg': stellar_data.get('st_logg'),
                'st_loggerr1': stellar_data.get('st_loggerr1'),
                'st_loggerr2': stellar_data.get('st_loggerr2'),
                'st_dist': stellar_data.get('st_dist'),
                'st_disterr1': stellar_data.get('st_disterr1'),
                'st_disterr2': stellar_data.get('st_disterr2'),
                'st_tmag': stellar_data.get('st_tmag'),
                'st_tmagerr1': stellar_data.get('st_tmagerr1'),
                'st_tmagerr2': stellar_data.get('st_tmagerr2')
            }
        )
        #we need this for front 
        output_json = processor.process()
       
        payload = build_payload(output_json, optimization_type=optimization_type, model_name=model_name)
    else:
        payload = build_model_payload_from_row(
            mission=mission,
            row=sp.response,
            optimization_type=optimization_type,
            model_name=model_name,
            overrides={},
        )
        payload = payload.to_dict()

    model_result = model_service.predict(payload)

    response = {"processed_json": output_json,  "manual_search": stellar_all_data, "model_result": model_result}
    return jsonify(deep_clean(response))

#we need an endpoint that also send the iamges to the front end


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


