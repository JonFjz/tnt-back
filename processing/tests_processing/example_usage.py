"""
Example usage of ExoplanetParameterProcessor for TESS and Kepler data.

This script demonstrates how to:
1. Load a FITS file (LC or TPF)
2. Provide catalog data as a dictionary
3. Process the data to extract exoplanet parameters
4. Generate visualizations for frontend display
5. Output JSON with parameters

Author: NASA Space Apps Challenge Team
Date: October 5, 2025
"""

import json
from pathlib import Path
from exoplanet_processor import ExoplanetParameterProcessor


def example_tess_processing():
    """Example: Process a TESS light curve file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: TESS Processing")
    print("=" * 80)
    
    # Sample TIC catalog data (replace with actual values from MAST)
    tic_catalog = {
        # Stellar parameters
        'st_teff': 7070.0,          # Effective temperature (K)
        'st_tefferr1': 100.0,       # Upper error
        'st_tefferr2': -100.0,      # Lower error
        'st_rad': 2.00601,          # Stellar radius (solar radii)
        'st_raderr1': 0.1,
        'st_raderr2': -0.1,
        'st_mass': 1.565,           # Stellar mass (solar masses)
        'st_masserr1': 0.05,
        'st_masserr2': -0.05,
        'st_logg': 4.0,             # Surface gravity (log g)
        'st_loggerr1': 0.1,
        'st_loggerr2': -0.1,
        'st_dist': 100.0,           # Distance (parsecs)
        'st_disterr1': 5.0,
        'st_disterr2': -5.0,
        'st_tmag': 10.5,            # TESS magnitude
        'st_tmagerr1': 0.05,
        'st_tmagerr2': -0.05
    }
    
    # Initialize processor
    processor = ExoplanetParameterProcessor(
        fits_path="path/to/tess_lc.fits",  # Replace with actual FITS path
        mission="tess",
        catalog=tic_catalog
    )
    
    # Process and generate output
    try:
        output_json_path = processor.process()
        print(f"\n✓ Processing complete!")
        print(f"✓ Output saved to: {output_json_path}")
        
        # Load and display results
        with open(output_json_path, 'r') as f:
            results = json.load(f)
        
        print(f"\n✓ Computed parameters:")
        params = results[0]['parameters']
        for key, value in list(params.items())[:10]:  # Show first 10
            print(f"  {key}: {value}")
        print(f"  ... and {len(params) - 10} more parameters")
        
    except FileNotFoundError:
        print("\n⚠ Example FITS file not found. Replace with actual file path.")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_kepler_processing():
    """Example: Process a Kepler light curve file."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Kepler Processing")
    print("=" * 80)
    
    # Sample Kepler DR25 catalog data
    kepler_catalog = {
        # Stellar parameters (can use TIC or Kepler-specific columns)
        'koi_steff': 5778.0,        # Or 'st_teff' for TIC
        'koi_steff_err1': 80.0,
        'koi_steff_err2': -80.0,
        'koi_srad': 1.0,            # Or 'st_rad'
        'koi_srad_err1': 0.05,
        'koi_srad_err2': -0.05,
        'koi_smass': 1.0,           # Or 'st_mass'
        'koi_smass_err1': 0.03,
        'koi_smass_err2': -0.03,
        'koi_slogg': 4.4,           # Or 'st_logg'
        'koi_slogg_err1': 0.1,
        'koi_slogg_err2': -0.1,
        'koi_srho': 1.4,            # Stellar density
        'koi_srho_err1': 0.1,
        # Photometry
        'koi_hmag': 12.5,
        'koi_imag': 12.3,
        'koi_zmag': 12.1,
        # Coordinates
        'dec': 45.0
    }
    
    # Initialize processor
    processor = ExoplanetParameterProcessor(
        fits_path="path/to/kepler_llc.fits",  # Replace with actual FITS path
        mission="kepler",
        catalog=kepler_catalog
    )
    
    # Process and generate output
    try:
        output_json_path = processor.process()
        print(f"\n✓ Processing complete!")
        print(f"✓ Output saved to: {output_json_path}")
        
        # Load and display results
        with open(output_json_path, 'r') as f:
            results = json.load(f)
        
        print(f"\n✓ Computed parameters:")
        params = results[0]['parameters']
        for key, value in list(params.items())[:10]:  # Show first 10
            print(f"  {key}: {value}")
        print(f"  ... and {len(params) - 10} more parameters")
        
    except FileNotFoundError:
        print("\n⚠ Example FITS file not found. Replace with actual file path.")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_tpf_processing():
    """Example: Process a Target Pixel File (TPF)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Target Pixel File (TPF) Processing")
    print("=" * 80)
    
    tic_catalog = {
        'st_teff': 6200.0,
        'st_rad': 1.5,
        'st_mass': 1.2,
        'st_logg': 4.2,
        'st_dist': 150.0,
        'st_tmag': 11.0
    }
    
    # TPF will be automatically converted to light curve
    processor = ExoplanetParameterProcessor(
        fits_path="path/to/tess_tp.fits",  # TPF file
        mission="tess",
        catalog=tic_catalog
    )
    
    try:
        output_json_path = processor.process()
        print(f"\n✓ TPF converted and processed!")
        print(f"✓ Check visualizations folder for aperture mask plot")
        
    except FileNotFoundError:
        print("\n⚠ Example TPF file not found. Replace with actual file path.")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_null_handling():
    """Example: Demonstrate null handling when catalog values missing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Null Handling (Missing Catalog Values)")
    print("=" * 80)
    
    # Catalog with missing/null values
    incomplete_catalog = {
        'st_teff': 5500.0,
        'st_rad': None,           # Missing stellar radius
        'st_mass': None,          # Missing stellar mass
        'st_logg': 4.5,
        # Other values missing
    }
    
    processor = ExoplanetParameterProcessor(
        fits_path="path/to/tess_lc.fits",
        mission="tess",
        catalog=incomplete_catalog
    )
    
    try:
        output_json_path = processor.process()
        
        with open(output_json_path, 'r') as f:
            results = json.load(f)
        
        print(f"\n✓ Processing completed despite missing values")
        print(f"✓ Parameters dependent on missing values will be null:")
        
        params = results[0]['parameters']
        null_params = [k for k, v in params.items() if v is None]
        print(f"  Null parameters: {len(null_params)}/{len(params)}")
        print(f"  Examples: {null_params[:5]}")
        
    except FileNotFoundError:
        print("\n⚠ Example FITS file not found. Replace with actual file path.")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def load_catalog_from_json(json_path: str) -> dict:
    """
    Helper function to load catalog from JSON file.
    
    Args:
        json_path: Path to TIC/DR25 JSON catalog file
    
    Returns:
        Dictionary with catalog parameters
    """
    with open(json_path, 'r') as f:
        catalog = json.load(f)
    return catalog


def batch_processing_example():
    """Example: Process multiple FITS files in batch."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Batch Processing")
    print("=" * 80)
    
    fits_files = [
        "path/to/file1.fits",
        "path/to/file2.fits",
        "path/to/file3.fits"
    ]
    
    catalog_files = [
        "path/to/catalog1.json",
        "path/to/catalog2.json",
        "path/to/catalog3.json"
    ]
    
    results = []
    
    for fits_file, catalog_file in zip(fits_files, catalog_files):
        try:
            # Load catalog
            catalog = load_catalog_from_json(catalog_file)
            
            # Process
            processor = ExoplanetParameterProcessor(
                fits_path=fits_file,
                mission="tess",  # or detect from filename
                catalog=catalog
            )
            
            output_path = processor.process()
            results.append(output_path)
            print(f"  ✓ Processed: {Path(fits_file).name}")
            
        except Exception as e:
            print(f"  ✗ Failed {Path(fits_file).name}: {e}")
    
    print(f"\n✓ Batch processing complete: {len(results)}/{len(fits_files)} successful")


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ExoplanetParameterProcessor - Usage Examples")
    print("=" * 80)
    
    # Run examples (comment out as needed)
    example_tess_processing()
    example_kepler_processing()
    example_tpf_processing()
    example_null_handling()
    # batch_processing_example()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
    print("\nNote: Update file paths to actual FITS and catalog files to run successfully.")
