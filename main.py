
from flask import Flask, jsonify
from src.controllers.search_controller import SearchController
from src.controllers.stars_controller import StarsController

app = Flask(__name__)

# Initialize controllers
search_controller = SearchController()
stars_controller = StarsController()

@app.get("/")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "TNT Backend API is running"})

@app.get("/stars")
def get_star_lightcurve():
    """Get lightcurve data for a specific star."""
    return stars_controller.get_star_lightcurve()

@app.get("/search")
def search_stars():
    """Search for stars using MAST catalog with various filters."""
    return search_controller.search_stars()




if __name__ == "__main__":
    # Use 0.0.0.0 to allow external connections (required for Docker)
    app.run(host="0.0.0.0", port=5000, debug=True)


