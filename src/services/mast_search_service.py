# src/services/mast_search_service.py
import sys
import json
import requests
from urllib.parse import quote as urlencode
from typing import Dict, Any, Optional, Tuple
import numpy as np

class MastSearchService:
    """Service for searching stars using MAST API based on the tutorial."""
    
    def __init__(self):
        self.base_url = 'https://mast.stsci.edu/api/v0/invoke'
        self.version = ".".join(map(str, sys.version_info[:3]))
        
    def mast_query(self, request: Dict[str, Any]) -> Tuple[Dict, str]:
        """Perform a MAST query from the tutorial.
        
        Parameters
        ----------
        request : dict
            The MAST request json object
            
        Returns
        -------
        Tuple[Dict, str]
            Tuple of (HTTP headers, response content)
        """
        
        # Create Http Header Variables
        headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain",
            "User-agent": f"python-requests/{self.version}"
        }

        # Encoding the request as a json string
        req_string = json.dumps(request)
        req_string = urlencode(req_string)
        
        # Perform the HTTP request
        resp = requests.post(self.base_url, data=f"request={req_string}", headers=headers)
        
        # Pull out the headers and response content
        head = resp.headers
        content = resp.content.decode('utf-8')

        return head, content

    def search_stars_with_filters(self, ra: float, dec: float, radius: float,
                                 mag_min: float = 6, mag_max: float = 15,
                                 temp_min: float = 3000, temp_max: float = 7500,
                                 dist_min: float = 10, dist_max: float = 500) -> Dict[str, Any]:
        """
        Search for stars using the TIC catalog with filters matching your UI.
        
        Parameters matching your frontend form:
        ----------
        ra : float
            Right Ascension in degrees (0-360)
        dec : float
            Declination in degrees (-90 to 90)  
        radius : float
            Search radius in arcminutes (0.01 to 30)
        mag_min : float
            Minimum magnitude (6 default, brighter stars)
        mag_max : float  
            Maximum magnitude (15 default, fainter stars)
        temp_min : float
            Minimum temperature in Kelvin (3000 default)
        temp_max : float
            Maximum temperature in Kelvin (7500 default) 
        dist_min : float
            Minimum distance in parsecs (10 default)
        dist_max : float
            Maximum distance in parsecs (500 default)
            
        Returns
        -------
        Dict containing search results
        """
        
        try:
            # Convert radius from arcminutes to degrees
            radius_deg = radius / 60.0
            
            # Use the TIC catalog filtered position search from the tutorial
            service = "Mast.Catalogs.Filtered.Tic.Position"
            
            # Build filters array
            filters = []
            
            # Magnitude filter (TESS magnitude)
            filters.append({
                "paramName": "Tmag",
                "values": [{"min": mag_min, "max": mag_max}]
            })
            
            # Temperature filter  
            filters.append({
                "paramName": "Teff", 
                "values": [{"min": temp_min, "max": temp_max}]
            })
            
            # Distance filter (try 'd' column)
            filters.append({
                "paramName": "d",
                "values": [{"min": dist_min, "max": dist_max}] 
            })
            
            # Build the mashup request following tutorial format
            mashup_request = {
                "service": service,
                "format": "json", 
                "params": {
                    "columns": "*",
                    "filters": filters,
                    "ra": ra,
                    "dec": dec, 
                    "radius": radius_deg
                },
                "pagesize": 1000,
                "page": 1
            }
            
            print(f"MAST Request: {json.dumps(mashup_request, indent=2)}")
            
            # Execute the query
            headers, out_string = self.mast_query(mashup_request)
            
            # Parse the response
            result = json.loads(out_string)
            
            print(f"MAST Response status: {result.get('status', 'Unknown')}")
            
            if result.get('status') == 'COMPLETE':
                data = result.get('data', [])
                return {
                    "status": "success",
                    "message": f"Found {len(data)} stars matching your criteria",
                    "count": len(data),
                    "data": data,
                    "query_params": {
                        "ra": ra,
                        "dec": dec, 
                        "radius_arcmin": radius,
                        "mag_range": [mag_min, mag_max],
                        "temp_range_k": [temp_min, temp_max],
                        "dist_range_pc": [dist_min, dist_max]
                    }
                }
            else:
                error_msg = result.get('msg', 'Unknown error from MAST API')
                return {
                    "status": "error",
                    "message": f"MAST API error: {error_msg}",
                    "count": 0,
                    "data": []
                }
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response string: {out_string[:500]}...")
            return {
                "status": "error", 
                "message": "Failed to parse MAST API response",
                "count": 0,
                "data": []
            }
        except Exception as e:
            print(f"Search error: {str(e)}")
            return {
                "status": "error",
                "message": f"Internal search error: {str(e)}",
                "count": 0, 
                "data": []
            }