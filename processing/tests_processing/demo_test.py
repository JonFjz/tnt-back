"""
Demo/Test Script for ExoplanetParameterProcessor

This script demonstrates the processor with sample data.
Use this to quickly test the implementation or show colleagues.

Usage:
    python demo_test.py

Author: NASA Space Apps Challenge Team
Date: October 5, 2025
"""

import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from exoplanet_processor import ExoplanetParameterProcessor


def demo_with_sample_data():
    """
    Demo with hardcoded sample catalog.
    
    NOTE: You need to provide your own FITS file!
    """
    print("\n" + "=" * 80)
    print("ExoplanetParameterProcessor - Demo Script")
    print("=" * 80)
    
    # ========== CONFIGURATION ==========
    # UPDATE THESE VALUES FOR YOUR DATA
    
    fits_path = "path/to/your/fits_file.fits"  # <-- UPDATE THIS!
    mission = "tess"  # or "kepler"
    
    # Sample TESS catalog (Sun-like star)
    sample_catalog = {
        # Stellar effective temperature
        'st_teff': 5778.0,
        'st_tefferr1': 100.0,
        'st_tefferr2': 100.0,
        
        # Stellar radius
        'st_rad': 1.0,
        'st_raderr1': 0.05,
        'st_raderr2': 0.05,
        
        # Stellar mass
        'st_mass': 1.0,
        'st_masserr1': 0.03,
        'st_masserr2': 0.03,
        
        # Surface gravity
        'st_logg': 4.4,
        'st_loggerr1': 0.1,
        'st_loggerr2': 0.1,
        
        # Distance
        'st_dist': 100.0,
        'st_disterr1': 5.0,
        'st_disterr2': 5.0,
        
        # TESS magnitude
        'st_tmag': 10.0,
        'st_tmagerr1': 0.05,
        'st_tmagerr2': 0.05
    }
    
    # ========== VALIDATION ==========
    
    fits_file = Path(fits_path)
    
    if not fits_file.exists():
        print("\n" + "!" * 80)
        print("ERROR: FITS file not found!")
        print(f"Path: {fits_path}")
        print("\nTo run this demo:")
        print("1. Download a TESS or Kepler FITS file")
        print("2. Update 'fits_path' in this script")
        print("3. Update 'mission' ('tess' or 'kepler')")
        print("4. Run again: python demo_test.py")
        print("!" * 80 + "\n")
        return False
    
    # ========== PROCESSING ==========
    
    print(f"\n✓ FITS file found: {fits_file.name}")
    print(f"✓ Mission: {mission.upper()}")
    print(f"✓ Catalog: {len(sample_catalog)} parameters")
    print("\nStarting processing...\n")
    
    try:
        # Initialize processor
        processor = ExoplanetParameterProcessor(
            fits_path=str(fits_path),
            mission=mission,
            catalog=sample_catalog
        )
        
        # Process!
        output_json_path = processor.process()
        
        # ========== SUCCESS ==========
        
        print("\n" + "=" * 80)
        print("✓✓✓ DEMO SUCCESSFUL ✓✓✓")
        print("=" * 80)
        print(f"\nOutput JSON: {output_json_path}")
        print(f"Visualizations: {fits_file.parent / 'visualizations'}")
        
        # Show sample of results
        import json
        with open(output_json_path, 'r') as f:
            results = json.load(f)
        
        print("\n" + "-" * 80)
        print("Sample Results (first 10 parameters):")
        print("-" * 80)
        
        params = results[0]['parameters']
        for i, (key, value) in enumerate(params.items()):
            if i >= 10:
                break
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key:30s} = {value:.6f}")
                else:
                    print(f"  {key:30s} = {value}")
            else:
                print(f"  {key:30s} = null")
        
        print(f"\n  ... and {len(params) - 10} more parameters")
        
        # Statistics
        non_null = sum(1 for v in params.values() if v is not None)
        print("\n" + "-" * 80)
        print(f"Total parameters: {len(params)}")
        print(f"Computed: {non_null}")
        print(f"Null: {len(params) - non_null}")
        print("-" * 80)
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Make sure the FITS file path is correct.")
        return False
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("Check mission type and catalog data.")
        return False
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_with_missing_catalog():
    """
    Demo showing null handling with incomplete catalog.
    """
    print("\n" + "=" * 80)
    print("ExoplanetParameterProcessor - Null Handling Demo")
    print("=" * 80)
    
    fits_path = "path/to/your/fits_file.fits"  # <-- UPDATE THIS!
    
    # Incomplete catalog (many missing values)
    incomplete_catalog = {
        'st_teff': 5778.0,      # Only temperature provided
        'st_rad': None,         # Missing
        'st_mass': None,        # Missing
        'st_logg': None,        # Missing
        'st_dist': None,        # Missing
        'st_tmag': 10.0         # Only magnitude provided
    }
    
    fits_file = Path(fits_path)
    
    if not fits_file.exists():
        print("\n⚠ FITS file not found (update 'fits_path' in script)")
        print("This demo shows how the processor handles missing data.\n")
        return False
    
    try:
        processor = ExoplanetParameterProcessor(
            fits_path=str(fits_path),
            mission="tess",
            catalog=incomplete_catalog
        )
        
        output = processor.process()
        
        print("\n✓ Processing succeeded despite missing values!")
        print(f"✓ Output: {output}")
        print("\nDependent parameters will be null in JSON.")
        print("Non-dependent parameters still computed.\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "=" * 80)
    print("Checking Dependencies")
    print("=" * 80 + "\n")
    
    dependencies = {
        'lightkurve': '2.5.1',
        'astropy': '7.1.0',
        'numpy': '2.3.3',
        'matplotlib': '3.10.6'
    }
    
    all_ok = True
    
    for package, min_version in dependencies.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package:15s} {version:10s} (need {min_version}+)")
        except ImportError:
            print(f"✗ {package:15s} NOT INSTALLED")
            all_ok = False
    
    if all_ok:
        print("\n✓ All dependencies installed!\n")
    else:
        print("\n✗ Missing dependencies!")
        print("Run: pip install -r requirements.txt\n")
    
    return all_ok


def print_usage():
    """Print usage instructions."""
    print("\n" + "=" * 80)
    print("ExoplanetParameterProcessor - Demo/Test Script")
    print("=" * 80)
    print("""
This script demonstrates the ExoplanetParameterProcessor class.

BEFORE RUNNING:
1. Update 'fits_path' in the demo_with_sample_data() function
2. Update 'mission' ('tess' or 'kepler')
3. Optionally update catalog values

USAGE:
    python demo_test.py

WHAT IT DOES:
- Checks dependencies
- Loads your FITS file
- Runs BLS transit search
- Computes all parameters
- Generates visualizations
- Saves JSON output
- Displays sample results

OUTPUT:
- JSON file: <fits_name>_parameters.json
- Visualizations: visualizations/*.png

For more examples, see:
- QUICKSTART.md
- example_usage.py
- EXOPLANET_PROCESSOR_DOCS.md
""")
    print("=" * 80 + "\n")


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    print_usage()
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies before continuing.")
        sys.exit(1)
    
    # Run main demo
    success = demo_with_sample_data()
    
    if success:
        print("\n" + "=" * 80)
        print("Demo completed successfully! 🎉")
        print("=" * 80)
        print("""
Next steps:
1. Check the output JSON file
2. View the visualizations in the visualizations/ folder
3. Modify catalog values and rerun
4. Try with different FITS files
5. Integrate into your pipeline

For more information:
- Quick Start: QUICKSTART.md
- Full Documentation: EXOPLANET_PROCESSOR_DOCS.md
- Examples: example_usage.py
""")
    else:
        print("\n" + "=" * 80)
        print("Demo failed - see error messages above")
        print("=" * 80)
        print("""
Common issues:
1. FITS file path incorrect → Update 'fits_path' in script
2. Missing dependencies → Run: pip install -r requirements.txt
3. Invalid catalog data → Check catalog keys and values

Need help? See QUICKSTART.md for troubleshooting.
""")
    
    sys.exit(0 if success else 1)
