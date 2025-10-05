"""
Quick test script to verify visualization generation
"""

from src.exoplanet_processor import ExoplanetParameterProcessor
import json
from pathlib import Path

# Test with the FITS file
fits_path = r"C:\Users\drind\Downloads\tess2025014115807-s0088-0000000114018671-0285-s_lc.fits"

catalog = {
    'st_teff': 7070.0,
    'st_tefferr1': 100.0,
    'st_tefferr2': 100.0,
    'st_rad': 2.00601,
    'st_raderr1': 0.1,
    'st_raderr2': 0.1,
    'st_mass': 1.565,
    'st_masserr1': 0.05,
    'st_masserr2': 0.05,
    'st_logg': 4.0,
    'st_loggerr1': 0.1,
    'st_loggerr2': 0.1,
    'st_dist': 100.0,
    'st_disterr1': 5.0,
    'st_disterr2': 5.0,
    'st_tmag': 10.5,
    'st_tmagerr': 0.05,
    'st_tmagerr2': 0.05
}

print("=" * 70)
print("TESTING VISUALIZATION GENERATION")
print("=" * 70)

try:
    # Process
    processor = ExoplanetParameterProcessor(fits_path, "tess", catalog)
    output_json_path = processor.process()
    
    print("\n" + "=" * 70)
    print("CHECKING OUTPUT")
    print("=" * 70)
    
    # Load and inspect JSON
    with open(output_json_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n✓ JSON loaded from: {output_json_path}")
    
    # Check visualizations
    if results[0].get('visualizations'):
        print(f"\n✓ Found {len(results[0]['visualizations'])} visualizations in JSON:")
        for viz in results[0]['visualizations']:
            viz_path = Path(viz['path'])
            exists = "✓ EXISTS" if viz_path.exists() else "✗ MISSING"
            print(f"  {exists}: {viz['type']}")
            print(f"           {viz['path']}")
    else:
        print("\n✗ No visualizations found in JSON output!")
    
    # Check parameters
    params = results[0]['parameters']
    non_null = sum(1 for v in params.values() if v is not None)
    print(f"\n✓ Parameters: {non_null}/{len(params)} non-null")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
