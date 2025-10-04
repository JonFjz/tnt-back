# src/controllers/stars_controller.py
from flask import request, jsonify
from src.star_manager import StarManager

class StarsController:
    """Controller for handling star lightcurve requests."""
    
    def __init__(self):
        self.manager = StarManager()
    
    def get_star_lightcurve(self):
        """
        Get lightcurve data for a specific star.
        
        Query Parameters:
        - mission: Mission name (required)
        - id: Target ID (required)
        
        Returns:
        JSON response with lightcurve data
        """
        try:
            mission = (request.args.get("mission") or "").strip()
            target_id = (request.args.get("id") or "").strip()
            
            if not mission or not target_id:
                return jsonify({
                    "status": "error",
                    "message": "Both 'mission' and 'id' parameters are required",
                    "data": []
                }), 400
            
            df = self.manager.fetch_lightcurve(mission, target_id)
            return jsonify(df.to_dict(orient="records"))
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error fetching lightcurve: {str(e)}",
                "data": []
            }), 500