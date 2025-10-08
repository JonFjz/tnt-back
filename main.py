
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

app = Flask(__name__)

# Initialize services
model_service = ModelService()
mast_search_service = MastSearchService()
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://192.168.100.66:5173"
).split(",")

# one line CORS: no credentials, allow GET/POST/OPTIONS
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

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
        
    response = StarProcessor(mission, target_id, oi_lookup, parameters,  file_processor.file_path if file else None)
    output_json = None
    stellar_all_data = {}

    if response.manualSearch:
        # Use actual stellar data from the StarProcessor response
        stellar_data = response.stellar["stellar"] if response.stellar else {}
        stellar_all_data = response.stellar

        TOI_REQUIRED = {
            "pl_trandurherr1","pl_trandurherr2","pl_orbpererr1","pl_orbper",
            "pl_tranmiderr1","pl_trandeperr2","pl_trandeperr1",
            "pl_tranmid","pl_trandep","pl_trandurh",
            "st_tmagerr2","st_tmagerr1","st_tmag",
            "st_disterr2","st_disterr1","st_dist",
            "st_loggerr2","st_logg",
            "st_teff","st_tefferr2","st_tefferr1",
            "st_rad","st_raderr1","st_raderr2",
            "eng_transit_probability",
            "eng_prad_srad_ratio",
            "eng_period_duration_ratio",
            "eng_duration_period_ratio"
        }

        KOI_REQUIRED = {
            "koi_prad","koi_prad_err1",
            "koi_dor","koi_ror",
            "koi_num_transits",
            "koi_duration_err1",
            "koi_period_err2",
            "koi_srad_err1",
            "koi_insol",
            "koi_model_snr",
            "koi_srho",
            "koi_max_mult_ev",
            "koi_teq",
            "eng_transit_probability"
        }

        processor = ExoplanetParameterProcessorTEST(
            fits_path=response.file_path,
            mission=mission,
            catalog=stellar_data,
            overrides = {} ,  # may include period/duration/depth
            prefer_flux="pdcsap",
            required_fields=TOI_REQUIRED if mission.lower() == "tess" else KOI_REQUIRED

        )
        #we need this for front 
        features = processor.run()  # <-- dict from ExoplanetParameterProcessorTEST

        items_like = [{
            "mission": mission.lower(),        # "tess" or "kepler"
            "parameters": features             # your feature dict
        }]
       
        payload = build_payload(items_like, optimization_type=optimization_type, allowed_features=TOI_REQUIRED if mission.lower() == "tess" else KOI_REQUIRED, model_name=model_name)
    else:
        payload = build_model_payload_from_row(
            mission=mission,
            row=response.response,
            optimization_type=optimization_type,
            model_name=model_name,
            overrides={},
        )
        payload = payload.to_dict()

    model_result = model_service.predict(payload)

    response = {"processed_json": output_json,  "manual_search": stellar_all_data, "model_result": model_result}
    return response

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


