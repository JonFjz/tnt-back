
from flask import Flask, request, jsonify
import pandas as pd
from src.star_manager import Processor
from src.file_processor import FileProcessor
app = Flask(__name__)


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

@app.get("/train")
def train_model():
    return 2



if __name__ == "__main__":
   
    app.run(host="127.0.0.1", port=5000, debug=True)


