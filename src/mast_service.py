# src/mast_service.py
import sys
import json
import requests
from urllib.parse import quote as urlencode
from typing import Dict, Any, Optional, Tuple, List
from astropy.table import Table
import numpy as np


class MASTService:
    """Service class for interacting with the MAST (Mikulski Archive for Space Telescopes) API."""
    
    def __init__(self):
        self.base_url = 'https://mast.stsci.edu/api/v0/invoke'
        self.version = ".".join(map(str, sys.version_info[:3]))
        
    def _mast_query(self, request: Dict[str, Any]) -> Tuple[Dict, str]:
        """Perform a MAST query.
        
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

    def filter_stars(self, ra: float, dec: float, radius_arcmin: float, 
                    mag_range: Tuple[float, float], 
                    temp_range_k: Tuple[float, float], 
                    dist_range_pc: Tuple[float, float],
                    observation: Optional[str] = None, 
                    has_toi: bool = False,
                    page_size: int = 2000) -> Optional[Dict[str, Any]]:
        """
        Constructs and executes a MAST query to filter stars based on specified criteria.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees.
        dec : float
            Declination in degrees.
        radius_arcmin : float
            Search radius in arcminutes.
        mag_range : Tuple[float, float]
            (min_mag, max_mag).
        temp_range_k : Tuple[float, float]
            (min_temp, max_temp) in Kelvin.
        dist_range_pc : Tuple[float, float]
            (min_dist, max_dist) in parsecs.
        observation : str, optional
            Satellite name ('TESS' or 'K2'). Defaults to None.
        has_toi : bool, optional
            If True, filters for stars with a TOI identifier. Defaults to False.
        page_size : int, optional
            Number of results per page. Defaults to 2000.

        Returns
        -------
        Optional[Dict[str, Any]]
            The JSON response from the MAST API containing the filtered star data,
            or None if the query failed.
        """

        # Convert radius from arcminutes to degrees for the API
        radius_deg = radius_arcmin / 60.0

        # This service allows us to query the TIC catalog with multiple filters
        service = "Mast.Catalogs.Filtered.Tic.Position"

        # Build the list of filters based on the function arguments
        filters = []

        # Magnitude Filter
        filters.append({
            "paramName": "Tmag",
            "values": [{"min": mag_range[0], "max": mag_range[1]}]
        })

        # Temperature Filter
        filters.append({
            "paramName": "Teff",
            "values": [{"min": temp_range_k[0], "max": temp_range_k[1]}]
        })

        # Distance Filter - Try common distance column names in TIC
        # Note: Distance may not always be available, so we'll try different column names
        try:
            # Try 'd' first (common in TIC), then 'plx' for parallax-derived distance
            filters.append({
                "paramName": "d", 
                "values": [{"min": dist_range_pc[0], "max": dist_range_pc[1]}]
            })
        except:
            # If 'd' doesn't work, we'll skip distance filtering for now
            print("Warning: Distance filtering may not be available for this catalog")

        # TOI/KOI Filter. This searches for any entry that has a TOI ID.
        # Note: TOI information is often in separate catalogs
        if has_toi:
            # Skip TOI filtering for now as it requires cross-referencing with TOI catalog
            print("Note: TOI filtering requires additional catalog cross-reference - skipping for now")
            
        # Build the final request object
        mashup_request = {
            "service": service,
            "format": "json",
            "params": {
                "columns": "*",  # Request all available columns
                "filters": filters,
                "ra": ra,
                "dec": dec, 
                "radius": radius_deg
            },
            'pagesize': page_size,
            'page': 1
        }

        # Add observation/mission filter if specified.
        # Note: Filtering observations (e.g. TESS data products) and catalog properties
        # are often separate queries. This example filters the TIC catalog.
        # To find actual observations for these stars, a second query would be needed.
        if observation:
            # For simplicity in this example, we'll note that TIC is the catalog for TESS.
            # Filtering by 'K2' would require querying a different catalog or cross-referencing.
            print(f"Note: Querying the TESS Input Catalog (TIC) for {observation} mission.")

        # Debug: Print the request being sent
        print(f"Sending MAST request: {json.dumps(mashup_request, indent=2)}")

        # Execute the query
        try:
            headers, out_string = self._mast_query(mashup_request)
            print(f"MAST response: {out_string[:500]}...")  # Print first 500 chars
            filtered_data = json.loads(out_string)
            return filtered_data
        except json.JSONDecodeError:
            print("Error: Could not decode JSON response.")
            print("Response string:", out_string)
            return None
        except Exception as e:
            print(f"Error executing MAST query: {str(e)}")
            return None

    def get_star_table(self, search_results: Dict[str, Any]) -> Optional[Table]:
        """
        Convert MAST search results to an Astropy Table.
        
        Parameters
        ----------
        search_results : Dict[str, Any]
            The JSON response from a MAST search
            
        Returns
        -------
        Optional[Table]
            An Astropy Table containing the star data, or None if conversion failed
        """
        try:
            if search_results and search_results.get('status') == 'COMPLETE':
                return Table(search_results['data'])
            return None
        except Exception as e:
            print(f"Error creating Astropy Table: {str(e)}")
            return None

    def format_search_results(self, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format MAST search results for API response.
        
        Parameters
        ----------
        search_results : Dict[str, Any]
            The JSON response from a MAST search
            
        Returns
        -------
        Dict[str, Any]
            Formatted response with status, count, and data
        """
        if not search_results:
            return {
                "status": "error",
                "message": "No results returned from MAST query",
                "count": 0,
                "data": []
            }
            
        if search_results.get('status') == 'COMPLETE':
            data = search_results.get('data', [])
            return {
                "status": "success",
                "message": f"Found {len(data)} stars matching the criteria",
                "count": len(data),
                "data": data
            }
        else:
            return {
                "status": "error",
                "message": search_results.get('msg', 'Unknown error from MAST API'),
                "count": 0,
                "data": []
            }