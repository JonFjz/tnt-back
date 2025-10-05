# ExoplanetParameterProcessor Documentation

## Overview

`ExoplanetParameterProcessor` is a Python class designed to process TESS and Kepler FITS light curve files to derive exoplanet and stellar parameters for machine learning models. It performs Box Least Squares (BLS) transit detection, combines FITS data with catalog information, and outputs comprehensive parameter sets with visualization plots.

## Features

- ✅ **Dual Mission Support**: Processes both TESS and Kepler data
- ✅ **TPF Conversion**: Automatically converts Target Pixel Files to light curves
- ✅ **BLS Transit Detection**: Uses `lightkurve` BLS for period/depth/duration extraction
- ✅ **Comprehensive Parameters**: Computes 31 TESS features and 43 Kepler features
- ✅ **Null Handling**: Gracefully handles missing catalog values
- ✅ **Visualizations**: Generates plots for aperture, raw LC, normalized LC, and folded LC
- ✅ **JSON Output**: Machine-readable format for ML models and frontend integration
- ✅ **Verbose Logging**: Detailed processing steps for debugging

---

## Installation

### Requirements

```bash
# All dependencies already in requirements.txt
pip install -r requirements.txt
```

**Key Dependencies:**
- `lightkurve >= 2.5.1`
- `astropy >= 7.1.0`
- `numpy >= 2.3.3`
- `matplotlib >= 3.10.6`

---

## Quick Start

### Basic Usage

```python
from exoplanet_processor import ExoplanetParameterProcessor

# Initialize with FITS file, mission type, and catalog
processor = ExoplanetParameterProcessor(
    fits_path="tess2025..._lc.fits",
    mission="tess",  # or "kepler"
    catalog={
        'st_teff': 7070.0,
        'st_rad': 2.006,
        'st_mass': 1.565,
        # ... more catalog parameters
    }
)

# Process and generate output
output_json = processor.process()
print(f"Results saved to: {output_json}")
```

---

## Class Reference

### Constructor

```python
ExoplanetParameterProcessor(fits_path: str, mission: str, catalog: dict)
```

**Parameters:**
- `fits_path` (str): Path to FITS file (LC or TPF format)
- `mission` (str): Mission type - `'tess'` or `'kepler'`
- `catalog` (dict): Dictionary with stellar/catalog parameters

**Raises:**
- `ValueError`: If mission type is invalid
- `FileNotFoundError`: If FITS file doesn't exist

---

### Methods

#### `process() -> str`

Main processing pipeline. Executes all steps sequentially.

**Returns:** Path to output JSON file

**Raises:** `Exception` if any processing step fails

**Processing Steps:**
1. Load FITS (LC or TPF → LC conversion)
2. Run BLS transit search
3. Process mission-specific parameters
4. Compute engineered features
5. Generate visualizations
6. Save JSON output

---

### FITS File Handling

#### Supported Formats

| Format | Extension | Handling |
|--------|-----------|----------|
| Light Curve | `LIGHTCURVE` | Direct load, uses PDCSAP_FLUX or SAP_FLUX |
| Target Pixel File | `PIXELS` or `TARGETTABLES` | Converts to LC via aperture photometry |

#### TPF Aperture Selection

Priority order:
1. **Pipeline aperture** (mission's official mask)
2. **Threshold aperture** (3-sigma above background)

---

### Parameter Sets

#### TESS Parameters (31 total)

**Transit Parameters (13):**
- `pl_orbper`, `pl_orbpererr1`, `pl_orbpererr2` - Orbital period
- `pl_trandurh`, `pl_trandurherr1`, `pl_trandurherr2` - Transit duration (hours)
- `pl_trandep`, `pl_trandeperr1`, `pl_trandeperr2` - Transit depth
- `pl_tranmid`, `pl_tranmiderr1`, `pl_tranmiderr2` - Transit midpoint
- `pl_eqt` - Equilibrium temperature

**Stellar Parameters (10):**
- `st_teff`, `st_tefferr1`, `st_tefferr2` - Effective temperature
- `st_rad`, `st_raderr1`, `st_raderr2` - Stellar radius
- `st_dist`, `st_disterr1`, `st_disterr2` - Distance
- `st_tmag`, `st_tmagerr1`, `st_tmagerr2` - TESS magnitude
- `st_logg`, `st_loggerr1`, `st_loggerr2` - Surface gravity

**Engineered Features (8):**
- `eng_period_duration_ratio` - Period / (duration/24)
- `eng_prad_srad_ratio` - sqrt(depth)
- `eng_transit_shape` - sqrt(depth) / duration
- `eng_duration_period_ratio` - (duration/24) / period
- `eng_stellar_density` - (P/D)³
- `eng_transit_probability` - R_star / P^(2/3)
- `eng_temp_ratio` - T_eq / T_eff
- `eng_signal_quality` - depth / 10^(Tmag/2.5)
- `eng_habitable_zone` - 1 if 200K ≤ T_eq ≤ 350K, else 0

#### Kepler Parameters (43 total)

**Transit Parameters (14 required):**
- `koi_period`, `koi_duration`, `koi_depth`, `koi_time0`
- `koi_num_transits`, `koi_model_snr`, `koi_max_mult_ev`
- `koi_prad`, `koi_ror`, `koi_dor`
- `koi_insol`, `koi_teq`
- Plus errors for each

**Stellar Parameters (22 additional):**
- `koi_steff`, `koi_srad`, `koi_smass`, `koi_slogg`, `koi_srho`
- Photometry: `koi_hmag`, `koi_imag`, `koi_zmag`
- Coordinates: `dec`, `koi_fwm_srao`, `koi_fwm_sra_err`
- Plus errors

**Engineered Features (7):**
- `eng_period_duration_ratio` - Period / duration
- `eng_prad_srad_ratio` - R_p / (R_star * 109)
- `eng_snr_depth_ratio` - SNR / depth
- `eng_stellar_density` - R_star³ / M_star
- `eng_transit_probability` - R_star / (a/R_star)
- `eng_temp_ratio` - T_eq / T_eff
- `eng_habitable_zone` - HZ indicator

---

### Visualizations

Output directory: `<FITS_directory>/visualizations/`

**Generated Plots:**

1. **Aperture Mask** (TPF only)
   - Filename: `<FITS_stem>_aperture.png`
   - Shows selected pixels for photometry

2. **Raw Light Curve**
   - Filename: `<FITS_stem>_raw_lc.png`
   - Unprocessed flux vs. time

3. **Normalized Light Curve**
   - Filename: `<FITS_stem>_normalized_lc.png`
   - Quality-filtered, outlier-removed, normalized flux

4. **Folded Light Curve**
   - Filename: `<FITS_stem>_folded_lc.png`
   - Phase-folded at detected period with binned data

---

### JSON Output Format

```json
[
  {
    "mission": "tess",
    "parameters": {
      "pl_orbper": 3.52474,
      "pl_trandep": 0.0045,
      "pl_trandurh": 2.3,
      "st_teff": 7070.0,
      "st_rad": 2.006,
      "eng_period_duration_ratio": 36.6,
      "eng_habitable_zone": 0,
      "st_mass": null,
      ...
    }
  }
]
```

**Key Points:**
- Array format (future-proof for multi-planet systems)
- `null` for uncomputable parameters
- All numeric values as floats
- No NaN values (converted to null)

---

## Null Handling

### Philosophy

- **Graceful Degradation**: Missing catalog values don't crash the processor
- **Propagation**: If input parameter is null → dependent outputs are null
- **Logging**: Every null value triggers a warning with reason

### Example

```python
catalog = {
    'st_teff': 5500.0,
    'st_rad': None,  # Missing
    'st_mass': None   # Missing
}

# Processing continues, but:
# - pl_rad = null (needs st_rad)
# - pl_eqt = null (needs st_rad and derived a/R_star)
# - eng_period_duration_ratio = computed (doesn't need st_rad)
```

---

## Error Handling

| Error Type | Handling | Result |
|------------|----------|--------|
| Invalid mission | Raise ValueError | Stops execution |
| FITS not found | Raise FileNotFoundError | Stops execution |
| TPF conversion fails | Try pipeline → threshold → raise | Stops if both fail |
| BLS finds no transit | Log warning, set nulls | Continues with nulls |
| Catalog key missing | Return None, log warning | Continues with nulls |
| Math error (div/0, sqrt<0) | Return None, log | Continues with nulls |

---

## Logging

### Configuration

```python
# Automatic logging setup per instance
# Format: YYYY-MM-DD HH:MM:SS - ExoProcessor_<fits_stem> - LEVEL - message
```

### Log Levels

- `DEBUG`: Detailed internal operations (extension names, flux columns)
- `INFO`: Major processing steps (✓ markers)
- `WARNING`: Null values, fallback actions (⚠ markers)
- `ERROR`: Critical failures (✗ markers)

### Example Output

```
2025-10-05 12:30:45 - ExoProcessor_tess2025... - INFO - ======================================================================
2025-10-05 12:30:45 - ExoProcessor_tess2025... - INFO - STEP 1/5: Loading FITS file
2025-10-05 12:30:45 - ExoProcessor_tess2025... - INFO - ======================================================================
2025-10-05 12:30:46 - ExoProcessor_tess2025... - INFO - ✓ Detected Light Curve FITS
2025-10-05 12:30:46 - ExoProcessor_tess2025... - INFO -   Using flux column: pdcsap_flux
2025-10-05 12:30:46 - ExoProcessor_tess2025... - WARNING - ⚠ Catalog key 'st_mass' is null or missing
```

---

## Advanced Usage

### Custom Error Estimation

Currently uses placeholder errors (1% for period, 5% for duration). To implement bootstrap:

```python
# In _run_bls() method, replace:
result['period_err'] = 0.01 * result['period']

# With bootstrap sampling:
from scipy.stats import bootstrap
# ... bootstrap implementation ...
```

### Batch Processing

```python
fits_files = ["file1.fits", "file2.fits", "file3.fits"]
catalogs = [catalog1, catalog2, catalog3]

for fits_file, catalog in zip(fits_files, catalogs):
    processor = ExoplanetParameterProcessor(fits_file, "tess", catalog)
    processor.process()
```

### Integration with MAST API

```python
from astroquery.mast import Catalogs

# Fetch TIC catalog
tic_data = Catalogs.query_object("TIC 25155310", catalog="TIC")
catalog = tic_data[0].to_dict()  # Convert to dict

processor = ExoplanetParameterProcessor(fits_path, "tess", catalog)
```

---

## Performance

### Typical Processing Times

| Input Type | Cadences | Processing Time |
|------------|----------|-----------------|
| TESS LC | ~10,000 | 10-15 seconds |
| TESS TPF | ~10,000 | 20-30 seconds (aperture extraction) |
| Kepler LC | ~50,000 | 30-45 seconds |

**Bottlenecks:**
- BLS periodogram: O(N × M) where N = cadences, M = period grid points
- TPF aperture photometry: Depends on TPF size

---

## Troubleshooting

### Common Issues

**1. "No significant transit detected"**
- **Cause**: Noisy data, no real transit, or period outside search range
- **Solution**: Check raw light curve plot, adjust BLS period range in `_run_bls()`

**2. "Pipeline aperture unavailable"**
- **Cause**: TPF file doesn't have mission aperture mask
- **Solution**: Automatically falls back to threshold aperture (logged)

**3. Many null parameters**
- **Cause**: Incomplete catalog data
- **Solution**: Verify catalog source, use TIC for TESS or DR25 for Kepler

**4. Import errors**
- **Cause**: Missing dependencies
- **Solution**: `pip install -r requirements.txt`

---

## Contributing

### Code Structure

```
exoplanet_processor.py
├── __init__(): Initialization and validation
├── _load_fits(): FITS loading (LC/TPF)
├── _run_bls(): BLS transit detection
├── _process_tess(): TESS parameter calculations
├── _process_kepler(): Kepler parameter calculations
├── _compute_engineered_tess(): TESS engineered features
├── _compute_engineered_kepler(): Kepler engineered features
├── _generate_visualizations(): Plot generation
└── _save_json(): JSON output
```

### Adding New Features

1. Add calculation in `_process_tess()` or `_process_kepler()`
2. Use `_safe_catalog_get()` for catalog lookups
3. Use `_safe_div()`, `_safe_sqrt()` for math operations
4. Log each step with appropriate level
5. Return `None` if inputs missing

---

## License

NASA Space Apps Challenge 2025 - Open Source

---

## Contact

For issues, questions, or contributions, contact the development team.

---

## Version History

- **v1.0.0** (2025-10-05): Initial release
  - TESS and Kepler support
  - 31 TESS features, 43 Kepler features
  - Visualization generation
  - Null handling
  - Comprehensive logging
