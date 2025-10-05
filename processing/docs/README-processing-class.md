# TNT-Back: Exoplanet Parameter Processor

Backend system for processing TESS and Kepler FITS light curve files to extract exoplanet and stellar parameters for machine learning models.

## � Documentation

All documentation is organized in the [`docs/`](./docs) folder:

| # | Document | Description |
|---|----------|-------------|
| 00 | [**HANDOFF**](./docs/00_HANDOFF.md) | 🎯 **START HERE** - Quick demo guide for colleagues |
| 01 | [**QUICKSTART**](./docs/01_QUICKSTART.md) | 5-minute getting started guide with examples |
| 02 | [**API REFERENCE**](./docs/02_API_REFERENCE.md) | Complete API documentation & parameter reference |
| 03 | [**SOLUTION SUMMARY**](./docs/03_SOLUTION_SUMMARY.md) | Executive overview of the complete solution |
| 04 | [**ENHANCEMENTS**](./docs/04_ENHANCEMENTS.md) | Optimizations & performance improvements |
| 05 | [**PROJECT STRUCTURE**](./docs/05_PROJECT_STRUCTURE.md) | Visual project organization & file tree |
| 06 | [**DOCUMENTATION INDEX**](./docs/06_DOCUMENTATION_INDEX.md) | Navigation hub for all documentation |

> **💡 Quick Links:**
> - First time? → [00_HANDOFF.md](./docs/00_HANDOFF.md)
> - Need examples? → [01_QUICKSTART.md](./docs/01_QUICKSTART.md)
> - API details? → [02_API_REFERENCE.md](./docs/02_API_REFERENCE.md)

## �🚀 Features

- **Dual Mission Support**: Process both TESS and Kepler data
- **Automatic TPF Conversion**: Handles Target Pixel Files with aperture photometry
- **BLS Transit Detection**: Box Least Squares algorithm for period/depth/duration
- **Comprehensive Parameters**: 43 TESS features + 50 Kepler features
- **ML-Ready Output**: JSON format for model training and inference
- **Frontend Visualizations**: Generates plots (raw LC, normalized LC, folded LC, aperture)
- **Robust Null Handling**: Gracefully handles missing catalog data
- **Verbose Logging**: Detailed processing steps for debugging

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `lightkurve >= 2.5.1` - FITS handling and BLS
- `astropy >= 7.1.0` - FITS file operations
- `numpy >= 2.3.3` - Numerical computations
- `matplotlib >= 3.10.6` - Visualization generation

## 🔧 Quick Start

### Basic Usage

```python
from exoplanet_processor import ExoplanetParameterProcessor

# Initialize processor
processor = ExoplanetParameterProcessor(
    fits_path="tess2025232030459-s0096-0000000025155310-0293-s_lc.fits",
    mission="tess",  # or "kepler"
    catalog={
        'st_teff': 7070.0,
        'st_rad': 2.006,
        'st_mass': 1.565,
        'st_logg': 4.0,
        'st_dist': 100.0,
        'st_tmag': 10.5
    }
)

# Process and generate output
output_json = processor.process()
print(f"Results: {output_json}")
```

### Output Structure

```json
[
  {
    "mission": "tess",
    "parameters": {
      "pl_orbper": 3.52474,
      "pl_trandep": 0.0045,
      "pl_trandurh": 2.3,
      "st_teff": 7070.0,
      "eng_period_duration_ratio": 36.6,
      "eng_habitable_zone": 0,
      ...
    }
  }
]
```

## 📊 Generated Visualizations

All plots saved to `visualizations/` directory:

1. **Aperture Mask** (TPF only) - Shows pixel selection
2. **Raw Light Curve** - Unprocessed flux vs. time
3. **Normalized Light Curve** - Quality-filtered, normalized flux
4. **Folded Light Curve** - Phase-folded at detected period

## 🔬 Parameter Sets

### TESS (31 parameters)
- **Transit**: period, duration, depth, midpoint + errors
- **Stellar**: T_eff, radius, mass, distance, T_mag + errors
- **Derived**: equilibrium temp, planet radius, a/R_star
- **Engineered**: 8 ML features (period/duration ratio, transit probability, HZ indicator, etc.)

### Kepler (43 parameters)
- **Transit**: period, duration, depth, time0, num_transits, SNR + errors
- **Stellar**: T_eff, radius, mass, log g, density + errors
- **Photometry**: H, I, Z magnitudes
- **Derived**: planet radius, a/R_star, insolation, T_eq
- **Engineered**: 7 ML features

## 📖 Documentation

- **Full Documentation**: [EXOPLANET_PROCESSOR_DOCS.md](EXOPLANET_PROCESSOR_DOCS.md)
- **Usage Examples**: [example_usage.py](example_usage.py)
- **Class Reference**: See docstrings in [exoplanet_processor.py](exoplanet_processor.py)

## 🛠️ Advanced Features

### TPF Aperture Selection
- Priority: Pipeline mask → Threshold mask (3-sigma)
- Automatic fallback with logging
- Visualizes selected pixels

### Null Handling
- Missing catalog values → dependent parameters set to `null`
- No crashes, graceful degradation
- Comprehensive warnings logged

### Error Propagation
- Quadrature sum for error propagation
- Placeholder errors (1% period, 5% duration)
- Extensible for bootstrap methods

## 🔍 Example Workflows

### 1. TESS Light Curve
```python
processor = ExoplanetParameterProcessor("tess_lc.fits", "tess", tic_catalog)
processor.process()
```

### 2. TESS Target Pixel File
```python
processor = ExoplanetParameterProcessor("tess_tp.fits", "tess", tic_catalog)
# Automatically converts TPF → LC with aperture photometry
processor.process()
```

### 3. Kepler Long Cadence
```python
processor = ExoplanetParameterProcessor("kepler_llc.fits", "kepler", koi_catalog)
processor.process()
```

### 4. Batch Processing
```python
for fits_file, catalog in zip(fits_files, catalogs):
    processor = ExoplanetParameterProcessor(fits_file, "tess", catalog)
    processor.process()
```

## ⚙️ Configuration

### BLS Settings
Modify in `_run_bls()`:
```python
periodogram = self.light_curve.to_periodogram(
    method='bls',
    period=np.arange(0.5, 20, 0.001)  # Search range: 0.5-20 days
)
```

### Quality Filtering
Modify in `_load_fits()`:
```python
lc = lc[lc.quality == 0]  # Keep only good quality
lc = lc.remove_outliers(sigma=5)  # 5-sigma clipping
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No transit detected | Check raw LC plot; transit may be too shallow or period out of range |
| Pipeline aperture unavailable | Automatic fallback to threshold (normal for some TPFs) |
| Many null parameters | Verify catalog completeness (use TIC for TESS, DR25 for Kepler) |
| Import errors | Run `pip install -r requirements.txt` |

## 📈 Performance

| Input Type | Cadences | Time |
|------------|----------|------|
| TESS LC | ~10,000 | 10-15s |
| TESS TPF | ~10,000 | 20-30s |
| Kepler LC | ~50,000 | 30-45s |

## 🤝 Integration

### With MAST API
```python
from astroquery.mast import Catalogs
tic_data = Catalogs.query_object("TIC 25155310", catalog="TIC")
catalog = tic_data[0].to_dict()
```

### With Frontend
- JSON output: Direct API response
- Visualizations: Serve PNG files from `visualizations/`
- Parameter selection: Filter JSON by required features

## 📝 Project Structure

```
tnt-back/
├── exoplanet_processor.py      # Main class implementation
├── example_usage.py             # Usage examples
├── EXOPLANET_PROCESSOR_DOCS.md  # Full documentation
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container setup
└── README.md                    # This file
```

## 🎯 NASA Space Apps Challenge 2025

This project is part of the NASA Space Apps Challenge, focusing on exoplanet characterization and machine learning model development.

## 📄 License

Open Source - NASA Space Apps Challenge 2025

## 👥 Team

Developed by the TNT team for exoplanet detection and characterization.

---

**For detailed API reference and advanced usage, see [EXOPLANET_PROCESSOR_DOCS.md](EXOPLANET_PROCESSOR_DOCS.md)**

A simple Python backend application using Flask.

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/JonFjz/tnt-back.git
cd tnt-back
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Local Development
```bash
python main.py
```

The application will be available at `http://localhost:5000`

#### Using Docker
```bash
# Build the Docker image
docker build -t tnt-back .

# Run the container
docker run -p 5000:5000 tnt-back
```

### API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint

## Project Structure

```
tnt-back/
├── main.py           # Main application entry point
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker configuration
├── .gitignore        # Git ignore rules
└── README.md         # This file
```