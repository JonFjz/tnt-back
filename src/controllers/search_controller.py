# src/controllers/search_controller.py
from flask import request, jsonify
from src.mast_service import MASTService

class SearchController:
    """Controller for handling star search requests."""
    
    def __init__(self):
        self.mast_service = MASTService()
    
    def search_stars(self):
        """
        Search for stars using MAST catalog with various filters.
        
        Query Parameters:
        - ra: Right Ascension in degrees (required)
        - dec: Declination in degrees (required)
        - radius: Search radius in arcminutes (default: 15.0)
        - mag_min: Minimum magnitude (default: 8)
        - mag_max: Maximum magnitude (default: 14)
        - temp_min: Minimum temperature in Kelvin (default: 5000)
        - temp_max: Maximum temperature in Kelvin (default: 6000)
        - dist_min: Minimum distance in parsecs (default: 10)
        - dist_max: Maximum distance in parsecs (default: 500)
        - observation: Mission name ('TESS' or 'K2') (optional)
        - has_toi: Whether to filter for stars with TOI identifiers (default: false)
        - page_size: Number of results per page (default: 2000)
        
        Returns:
        JSON response with status, count, and star data
        """
        try:
            # Parse and validate parameters
            params = self._parse_search_parameters()
            if 'error' in params:
                return jsonify(params), 400
            
            # Execute the search
            search_results = self.mast_service.filter_stars(**params['search_params'])
            
            # Format and return results
            formatted_results = self.mast_service.format_search_results(search_results)
            return jsonify(formatted_results)
            
        except ValueError as ve:
            return self._error_response(f"Invalid parameter value: {str(ve)}"), 400
        except Exception as e:
            return self._error_response(f"Internal server error: {str(e)}"), 500
    
    def _parse_search_parameters(self):
        """Parse and validate search parameters from request."""
        # Parse required parameters
        ra = request.args.get("ra", type=float)
        dec = request.args.get("dec", type=float)
        
        if ra is None or dec is None:
            return self._error_response("Both 'ra' and 'dec' parameters are required")
        
        # Parse optional parameters with defaults
        radius_arcmin = request.args.get("radius", 15.0, type=float)
        mag_min = request.args.get("mag_min", 8.0, type=float)
        mag_max = request.args.get("mag_max", 14.0, type=float)
        temp_min = request.args.get("temp_min", 5000.0, type=float)
        temp_max = request.args.get("temp_max", 6000.0, type=float)
        dist_min = request.args.get("dist_min", 10.0, type=float)
        dist_max = request.args.get("dist_max", 500.0, type=float)
        observation = request.args.get("observation")
        has_toi = request.args.get("has_toi", "false").lower() in ("true", "1", "yes")
        page_size = request.args.get("page_size", 2000, type=int)
        
        # Validate parameter ranges
        validation_error = self._validate_parameter_ranges(mag_min, mag_max, temp_min, temp_max, dist_min, dist_max)
        if validation_error:
            return validation_error
        
        # Return validated parameters
        return {
            'search_params': {
                'ra': ra,
                'dec': dec,
                'radius_arcmin': radius_arcmin,
                'mag_range': (mag_min, mag_max),
                'temp_range_k': (temp_min, temp_max),
                'dist_range_pc': (dist_min, dist_max),
                'observation': observation,
                'has_toi': has_toi,
                'page_size': page_size
            }
        }
    
    def _validate_parameter_ranges(self, mag_min, mag_max, temp_min, temp_max, dist_min, dist_max):
        """Validate that min/max parameter ranges are valid."""
        if mag_min >= mag_max:
            return self._error_response("mag_min must be less than mag_max")
            
        if temp_min >= temp_max:
            return self._error_response("temp_min must be less than temp_max")
            
        if dist_min >= dist_max:
            return self._error_response("dist_min must be less than dist_max")
        
        return None
    
    def _error_response(self, message):
        """Create a standardized error response."""
        return {
            "error": True,
            "status": "error",
            "message": message,
            "count": 0,
            "data": []
        }