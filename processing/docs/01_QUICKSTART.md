# 🚀 QUICK START GUIDE - ExoplanetParameterProcessor

## For Colleagues: Run This First!

This guide gets you up and running with the `ExoplanetParameterProcessor` in 5 minutes.

---

## ✅ Step 1: Verify Installation

```bash
# Check Python version (need 3.8+)
python --version

# Install dependencies (already in requirements.txt)
pip install -r requirements.txt

# Key packages:
# - lightkurve==2.5.1
# - astropy==7.1.0
# - numpy==2.3.3
# - matplotlib==3.10.6
```

---

## ✅ Step 2: Basic Usage (Copy & Paste)

### TESS Example

```python
from exoplanet_processor import ExoplanetParameterProcessor

# Your FITS file path
fits_path = "path/to/your/tess_lc.fits"

# Your TIC catalog data (from MAST or local JSON)
catalog = {
    'st_teff': 5778.0,      # Stellar temperature (K)
    'st_tefferr1': 100.0,   # Upper error
    'st_tefferr2': 100.0,   # Lower error
    'st_rad': 1.0,          # Stellar radius (solar radii)
    'st_raderr1': 0.05,
    'st_raderr2': 0.05,
    'st_mass': 1.0,         # Stellar mass (solar masses)
    'st_masserr1': 0.03,
    'st_masserr2': 0.03,
    'st_logg': 4.4,         # Surface gravity
    'st_loggerr1': 0.1,
    'st_loggerr2': 0.1,
    'st_dist': 100.0,       # Distance (parsecs)
    'st_disterr1': 5.0,
    'st_disterr2': 5.0,
    'st_tmag': 10.0,        # TESS magnitude
    'st_tmagerr1': 0.05,
    'st_tmagerr2': 0.05
}

# Process!
processor = ExoplanetParameterProcessor(
    fits_path=fits_path,
    mission="tess",
    catalog=catalog
)

output_json = processor.process()
print(f"✓ Done! Results: {output_json}")
```

### Kepler Example

```python
from exoplanet_processor import ExoplanetParameterProcessor

fits_path = "path/to/your/kepler_llc.fits"

catalog = {
    'koi_steff': 5778.0,    # Or use 'st_teff' from TIC
    'koi_steff_err1': 80.0,
    'koi_steff_err2': 80.0,
    'koi_srad': 1.0,        # Or use 'st_rad'
    'koi_srad_err1': 0.05,
    'koi_srad_err2': 0.05,
    'koi_smass': 1.0,
    'koi_smass_err1': 0.03,
    'koi_smass_err2': 0.03,
    'koi_slogg': 4.4,
    'koi_srho': 1.4,        # Stellar density
    'koi_hmag': 12.5,
    'koi_imag': 12.3,
    'dec': 45.0
}

processor = ExoplanetParameterProcessor(
    fits_path=fits_path,
    mission="kepler",
    catalog=catalog
)

output_json = processor.process()
print(f"✓ Done! Results: {output_json}")
```

---

## ✅ Step 3: What You Get

### Output Files

After running `processor.process()`, you'll get:

1. **JSON File**: `<fits_filename>_parameters.json`
   ```json
   [
     {
       "mission": "tess",
       "parameters": {
         "pl_orbper": 3.524,
         "pl_trandep": 0.0045,
         "st_teff": 5778.0,
         "eng_habitable_zone": 1,
         ...
       }
     }
   ]
   ```

2. **Visualization Folder**: `visualizations/`
   - `*_aperture.png` (if TPF)
   - `*_raw_lc.png`
   - `*_normalized_lc.png`
   - `*_folded_lc.png`

### Console Output

You'll see detailed logging:
```
======================================================================
ExoplanetParameterProcessor initialized
FITS: tess2025..._lc.fits
Mission: TESS
======================================================================
======================================================================
STEP 1/5: Loading FITS file
======================================================================
✓ Detected Light Curve FITS
  Using flux column: pdcsap_flux
  Quality filter: 10000 → 9500 cadences
✓ FITS loaded successfully: 9500 good cadences
======================================================================
STEP 2/5: Running BLS transit search
======================================================================
  Period search range: 0.50 - 6.67 days
  Running BLS periodogram...
✓ Period detected: 3.524740 days
✓ Duration: 2.3000 hours
✓ BLS power: 0.3456
✓ Transit depth: 4500.00 ppm
======================================================================
STEP 3/5: Processing TESS parameters
======================================================================
  Transit parameters: period=3.524740, depth=0.0045
  Stellar parameters: Teff=5778.0K, R=1.0Rsun
  Derived T_eq: 850.5K
  Derived planet radius: 1.85 R_Earth
✓ TESS parameters processed
  Computing TESS engineered features...
  ✓ Engineered features computed: 9/9
======================================================================
STEP 4/5: Generating visualizations
======================================================================
  ✓ Saved raw light curve: tess2025..._raw_lc.png
  ✓ Saved normalized light curve: tess2025..._normalized_lc.png
  ✓ Saved folded light curve: tess2025..._folded_lc.png
✓ Visualizations generated
======================================================================
STEP 5/5: Saving JSON output
======================================================================
  ✓ Saved 31/34 parameters to: tess2025..._parameters.json
  ✓ Null parameters: 3
======================================================================
✓✓✓ PROCESSING COMPLETE ✓✓✓
======================================================================
```

---

## ✅ Step 4: Common Scenarios

### Scenario 1: Target Pixel File (TPF)

```python
# Same code, just use TPF file path
processor = ExoplanetParameterProcessor(
    fits_path="tess_tp.fits",  # TPF file
    mission="tess",
    catalog=catalog
)
processor.process()
# Automatically converts TPF → LC using aperture photometry
# Also saves aperture mask visualization!
```

### Scenario 2: Missing Catalog Data

```python
# It's okay if some values are missing!
catalog = {
    'st_teff': 5778.0,
    'st_rad': None,    # Missing - that's fine!
    'st_mass': None,   # Missing - that's fine!
    'st_logg': 4.4
}

processor = ExoplanetParameterProcessor(fits_path, "tess", catalog)
output = processor.process()

# Dependent parameters will be null in JSON
# Non-dependent parameters still computed
```

### Scenario 3: Load Catalog from JSON File

```python
import json

# Load catalog from file
with open("tic_catalog.json", "r") as f:
    catalog = json.load(f)

processor = ExoplanetParameterProcessor(fits_path, "tess", catalog)
processor.process()
```

### Scenario 4: Batch Processing

```python
import os

fits_dir = "path/to/fits_files/"
fits_files = [f for f in os.listdir(fits_dir) if f.endswith(".fits")]

for fits_file in fits_files:
    try:
        # Load corresponding catalog
        catalog = load_catalog_for_file(fits_file)  # Your function
        
        processor = ExoplanetParameterProcessor(
            fits_path=os.path.join(fits_dir, fits_file),
            mission="tess",
            catalog=catalog
        )
        
        output = processor.process()
        print(f"✓ Processed: {fits_file}")
        
    except Exception as e:
        print(f"✗ Failed {fits_file}: {e}")
```

---

## ✅ Step 5: Understanding the Output

### JSON Structure

```json
[
  {
    "mission": "tess",
    "parameters": {
      // Transit Parameters (13)
      "pl_orbper": 3.524,           // Period (days)
      "pl_orbpererr1": 0.035,       // Period error
      "pl_trandurh": 2.3,           // Duration (hours)
      "pl_trandep": 0.0045,         // Depth (fraction)
      "pl_tranmid": 2459000.5,      // Midpoint (BTJD)
      
      // Stellar Parameters (10)
      "st_teff": 5778.0,            // Temperature (K)
      "st_rad": 1.0,                // Radius (solar radii)
      "st_mass": 1.0,               // Mass (solar masses)
      "st_dist": 100.0,             // Distance (pc)
      "st_tmag": 10.0,              // TESS magnitude
      
      // Derived Parameters (3)
      "pl_eqt": 850.5,              // Equilibrium temp (K)
      "pl_rad": 1.85,               // Planet radius (Earth radii)
      "a_over_rstar": 10.5,         // a/R_star ratio
      
      // Engineered Features (8)
      "eng_period_duration_ratio": 36.6,
      "eng_prad_srad_ratio": 0.067,
      "eng_transit_probability": 0.012,
      "eng_habitable_zone": 0,      // 1 if 200-350K, else 0
      "eng_stellar_density": 1.2,
      ...
      
      // null if uncomputable
      "some_parameter": null
    }
  }
]
```

### Visualization Files

**1. Raw Light Curve** (`*_raw_lc.png`)
- Original flux vs. time
- Shows all data before filtering

**2. Normalized Light Curve** (`*_normalized_lc.png`)
- Quality-filtered, outlier-removed
- Median normalized to 1.0

**3. Folded Light Curve** (`*_folded_lc.png`)
- Phase-folded at detected period
- Shows transit clearly
- Red points = binned data

**4. Aperture Mask** (`*_aperture.png`, TPF only)
- Shows which pixels used for photometry
- Color indicates flux level

---

## 🐛 Troubleshooting

### Error: "FITS file not found"
```python
# Check file path
import os
print(os.path.exists("your_file.fits"))  # Should be True

# Use absolute path
fits_path = os.path.abspath("your_file.fits")
```

### Error: "Import lightkurve could not be resolved"
```bash
pip install lightkurve astropy numpy matplotlib
```

### Warning: "No significant transit detected"
- Light curve may be too noisy
- Transit may be too shallow
- Check raw light curve visualization
- Try adjusting period range (see docs)

### Warning: "Unusual period: 0.05 days"
- This is just a warning, not an error
- Parameter may be physically unusual
- Check if real or artifact

### Many null parameters
- Check your catalog data completeness
- Use TIC for TESS, DR25 for Kepler
- Non-dependent parameters still computed

---

## 📞 Need Help?

### Documentation
- **Full Docs**: `EXOPLANET_PROCESSOR_DOCS.md`
- **Examples**: `example_usage.py`
- **Enhancements**: `STEP5_ENHANCEMENTS.md`

### Quick Checks
```python
# Check class is imported correctly
from exoplanet_processor import ExoplanetParameterProcessor
print(ExoplanetParameterProcessor.__doc__)

# Check FITS file format
import lightkurve as lk
lc = lk.read("your_file.fits")
print(lc)

# Check catalog structure
print(catalog.keys())
print({k: v for k, v in catalog.items() if k.startswith('st_')})
```

---

## 🎯 Key Features to Show Colleagues

1. **Dual Format Support**: Handles both LC and TPF files automatically
2. **Smart Aperture**: Uses pipeline mask, falls back to threshold
3. **Null Safe**: Missing catalog values don't crash the processor
4. **Fast**: Adaptive BLS is 33-67% faster than fixed-range
5. **Validated**: Checks for unphysical parameters automatically
6. **Visualizations**: 4 plots ready for frontend display
7. **ML Ready**: JSON output with 31 TESS / 43 Kepler features

---

## 🚀 Quick Demo Script

Save this as `demo.py`:

```python
from exoplanet_processor import ExoplanetParameterProcessor

# Minimal example
catalog = {
    'st_teff': 5778.0, 'st_rad': 1.0, 'st_mass': 1.0,
    'st_logg': 4.4, 'st_dist': 100.0, 'st_tmag': 10.0
}

processor = ExoplanetParameterProcessor(
    fits_path="your_file.fits",  # UPDATE THIS
    mission="tess",               # or "kepler"
    catalog=catalog
)

try:
    output = processor.process()
    print(f"\n{'='*70}")
    print(f"SUCCESS! Output: {output}")
    print(f"Check the visualizations/ folder for plots!")
    print(f"{'='*70}\n")
except Exception as e:
    print(f"\nERROR: {e}\n")
```

Run:
```bash
python demo.py
```

---

## ✅ Checklist for Colleagues

Before showing the class:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Have a test FITS file ready
- [ ] Have corresponding catalog data (or use sample above)
- [ ] Know the mission type ('tess' or 'kepler')
- [ ] Have write permissions in directory (for output)

To demonstrate:
- [ ] Show basic usage (copy-paste from Step 2)
- [ ] Show console output (detailed logging)
- [ ] Show JSON output (parameter structure)
- [ ] Show visualizations (4 plots)
- [ ] Show null handling (missing catalog values)
- [ ] Show TPF conversion (if available)

---

**You're ready to go!** 🎉

For more details, see:
- `EXOPLANET_PROCESSOR_DOCS.md` - Complete API reference
- `example_usage.py` - 5 detailed examples
- `README.md` - Project overview
