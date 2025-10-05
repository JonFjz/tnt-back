"""
ExoplanetParameterProcessor: Process TESS/Kepler FITS files to derive exoplanet parameters.

This class handles:
- FITS loading (Light Curve and Target Pixel File)
- BLS transit detection
- TESS/Kepler parameter calculations
- Engineered feature computation
- Visualization generation for frontend
- JSON output with null handling

Author: NASA Space Apps Challenge Team
Date: October 5, 2025
"""

import logging
import json
import numpy as np
import warnings
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import lightkurve as lk
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
matplotlib.use('Agg')  # Non-interactive backend for saving plots


class ExoplanetParameterProcessor:
    """
    Process single-star TESS or Kepler FITS light curve to derive exoplanet parameters.
    
    Attributes:
        fits_path (Path): Path to FITS file (LC or TPF)
        mission (str): 'tess' or 'kepler'
        catalog (dict): TIC/DR25 catalog data
        logger (logging.Logger): Verbose logger
        light_curve (lk.LightCurve): Processed light curve
        is_tpf (bool): True if input was TPF
        aperture_mask (np.ndarray): Aperture mask if TPF
        bls_result (dict): BLS transit parameters
        parameters (dict): Final computed parameters
    """
    
    def __init__(self, fits_path: str, mission: str, catalog: dict):
        """
        Initialize the processor.
        
        Args:
            fits_path: Path to FITS file (LC or TPF)
            mission: 'tess' or 'kepler'
            catalog: Dictionary with stellar/catalog parameters
        
        Raises:
            ValueError: If mission type invalid
            FileNotFoundError: If FITS file doesn't exist
        """
        self.fits_path = Path(fits_path)
        self.mission = mission.lower()
        self.catalog = catalog
        
        # Initialize attributes
        self.light_curve = None
        self.light_curve_raw = None  # Keep raw for visualization
        self.is_tpf = False
        self.aperture_mask = None
        self.tpf = None  # Store TPF object for plotting
        self.bls_result = {}
        self.parameters = {}
        self.folded_lc = None  # Store for plotting
        self.visualization_paths = []  # Track generated visualization files
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Validate inputs
        self._validate_inputs()
        
        self.logger.info("=" * 70)
        self.logger.info(f"ExoplanetParameterProcessor initialized")
        self.logger.info(f"FITS: {self.fits_path.name}")
        self.logger.info(f"Mission: {self.mission.upper()}")
        self.logger.info("=" * 70)
    
    def _setup_logging(self) -> logging.Logger:
        """Configure verbose logging."""
        logger = logging.getLogger(f"ExoProcessor_{self.fits_path.stem}")
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_inputs(self):
        """Validate mission type and FITS file existence."""
        if self.mission not in ['tess', 'kepler']:
            raise ValueError(f"Invalid mission: {self.mission}. Must be 'tess' or 'kepler'")
        
        if not self.fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")
        
        self.logger.debug("✓ Inputs validated")
    
    def _safe_catalog_get(self, key: str, default=None) -> Optional[float]:
        """
        Safely retrieve catalog value, return None if missing/null.
        
        Args:
            key: Catalog key (e.g., 'st_teff')
            default: Default value if key missing
        
        Returns:
            Float value or None if missing/null/NaN
        """
        value = self.catalog.get(key, default)
        
        # Check for None, NaN, or empty string
        if value is None or value == '':
            self.logger.warning(f"⚠ Catalog key '{key}' is null or missing")
            return None
        
        # Check for NaN
        try:
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                self.logger.warning(f"⚠ Catalog key '{key}' is NaN")
                return None
        except (TypeError, ValueError):
            pass
        
        return float(value)
    
    def _propagate_error(
        self, 
        value: Optional[float], 
        rel_errors: List[Tuple[Optional[float], Optional[float]]]
    ) -> Optional[float]:
        """
        Propagate relative errors using quadrature sum: σ_f = |f| * sqrt(Σ(σ_i/x_i)²)
        
        Args:
            value: The computed value
            rel_errors: List of (error, value) tuples for error propagation
        
        Returns:
            Propagated error or None if any input is None
        """
        if value is None or value == 0:
            return None
        
        # Check if any error or value is None
        for err, val in rel_errors:
            if err is None or val is None or val == 0:
                return None
        
        try:
            sum_sq = sum((err / val) ** 2 for err, val in rel_errors if val != 0)
            return abs(value) * np.sqrt(sum_sq)
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
    
    def _safe_div(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Safe division with null checking."""
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
    
    def _safe_sqrt(self, value: Optional[float]) -> Optional[float]:
        """Safe square root with null and negative checking."""
        if value is None or value < 0:
            return None
        return np.sqrt(value)
    
    def _safe_power(self, base: Optional[float], exponent: float) -> Optional[float]:
        """Safe exponentiation with null checking."""
        if base is None or base < 0:
            return None
        try:
            return base ** exponent
        except (ValueError, OverflowError):
            return None
    
    def _compute_semi_major_axis_ratio(self, period: Optional[float], duration: Optional[float]) -> Optional[float]:
        """
        Compute semi-major axis to stellar radius ratio (a/R_star).
        
        Formula: a/R_star ≈ (P / (π * T_dur)) * sqrt(1 - b²)
        Assumes impact parameter b ≈ 0 (central transit)
        
        Args:
            period: Orbital period (days)
            duration: Transit duration (days)
        
        Returns:
            a/R_star ratio or None
        """
        if period is None or duration is None or duration == 0:
            return None
        return self._safe_div(period, np.pi * duration)
    
    def _compute_equilibrium_temperature(
        self, 
        t_eff: Optional[float], 
        a_over_rstar: Optional[float],
        albedo: float = 0.3
    ) -> Optional[float]:
        """
        Compute planet equilibrium temperature.
        
        Formula: T_eq = T_eff * (R_star/a)^0.5 * (1 - A)^0.25
        
        Args:
            t_eff: Stellar effective temperature (K)
            a_over_rstar: Semi-major axis to stellar radius ratio
            albedo: Bond albedo (default 0.3)
        
        Returns:
            Equilibrium temperature (K) or None
        """
        if t_eff is None or a_over_rstar is None or a_over_rstar == 0:
            return None
        
        try:
            t_eq = t_eff * (1.0 / a_over_rstar) ** 0.5 * (1 - albedo) ** 0.25
            return t_eq if t_eq > 0 else None
        except (ValueError, OverflowError):
            return None
    
    def _compute_planet_radius(
        self, 
        stellar_radius: Optional[float], 
        depth: Optional[float],
        r_sun_to_r_earth: float = 109.0
    ) -> Optional[float]:
        """
        Compute planet radius from transit depth.
        
        Formula: R_p = R_star * sqrt(depth) * 109
        
        Args:
            stellar_radius: Stellar radius (solar radii)
            depth: Transit depth (fraction)
            r_sun_to_r_earth: Solar to Earth radius ratio (default 109)
        
        Returns:
            Planet radius (Earth radii) or None
        """
        if stellar_radius is None or depth is None or depth < 0:
            return None
        
        sqrt_depth = self._safe_sqrt(depth)
        if sqrt_depth is None:
            return None
        
        return stellar_radius * sqrt_depth * r_sun_to_r_earth
    
    def _compute_insolation_flux(
        self,
        t_eff: Optional[float],
        stellar_radius: Optional[float],
        a_over_rstar: Optional[float],
        t_sun: float = 5778.0
    ) -> Optional[float]:
        """
        Compute insolation flux relative to Earth.
        
        Formula: S = (T_eff/T_sun)^4 * (R_star/a)^2
        
        Args:
            t_eff: Stellar effective temperature (K)
            stellar_radius: Stellar radius (solar radii)
            a_over_rstar: Semi-major axis to stellar radius ratio
            t_sun: Solar temperature (K, default 5778)
        
        Returns:
            Insolation flux (Earth = 1.0) or None
        """
        if None in [t_eff, stellar_radius, a_over_rstar] or a_over_rstar == 0:
            return None
        
        try:
            temp_ratio = (t_eff / t_sun) ** 4
            distance_ratio = (stellar_radius / a_over_rstar) ** 2
            return temp_ratio * distance_ratio
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
    
    def _is_in_habitable_zone(self, t_eq: Optional[float], t_min: float = 200, t_max: float = 350) -> Optional[int]:
        """
        Check if planet is in habitable zone based on equilibrium temperature.
        
        Args:
            t_eq: Equilibrium temperature (K)
            t_min: Minimum HZ temperature (K, default 200)
            t_max: Maximum HZ temperature (K, default 350)
        
        Returns:
            1 if in HZ, 0 if not, None if t_eq is None
        """
        if t_eq is None:
            return None
        return int(t_min <= t_eq <= t_max)
    
    def _validate_light_curve(self, lc: lk.LightCurve) -> bool:
        """
        Validate light curve has sufficient data for processing.
        
        Args:
            lc: Light curve to validate
        
        Returns:
            True if valid, False otherwise
        """
        min_cadences = 100  # Minimum cadences for meaningful BLS
        
        if len(lc) < min_cadences:
            self.logger.error(f"✗ Light curve too short: {len(lc)} cadences (minimum: {min_cadences})")
            return False
        
        # Check for all NaN or constant flux
        if np.all(np.isnan(lc.flux.value)):
            self.logger.error("✗ All flux values are NaN")
            return False
        
        flux_std = np.nanstd(lc.flux.value)
        if flux_std == 0 or np.isnan(flux_std):
            self.logger.error("✗ Flux has zero variance (constant or all NaN)")
            return False
        
        # Check time span
        timespan = (lc.time[-1] - lc.time[0]).value
        if timespan < 1.0:  # Less than 1 day
            self.logger.warning(f"⚠ Very short timespan: {timespan:.2f} days")
        
        return True
    
    def _load_fits(self) -> lk.LightCurve:
        """
        Load FITS file as LightCurve.
        
        Handles:
        - Light Curve FITS (LIGHTCURVE extension) → direct load
        - Target Pixel File (PIXELS/TARGETTABLES extension) → convert via aperture photometry
        
        Returns:
            lightkurve.LightCurve with quality-filtered, normalized flux
        
        Raises:
            ValueError: If FITS format unrecognized
        """
        self.logger.info("=" * 70)
        self.logger.info("STEP 1/5: Loading FITS file")
        self.logger.info("=" * 70)
        
        # Inspect FITS extensions
        with fits.open(self.fits_path) as hdul:
            ext_names = [hdu.name for hdu in hdul]
            self.logger.debug(f"FITS extensions: {ext_names}")
        
        try:
            # Try Light Curve first
            if 'LIGHTCURVE' in ext_names:
                self.logger.info("✓ Detected Light Curve FITS")
                lc = lk.read(self.fits_path)
                
                # Determine flux column
                if 'pdcsap_flux' in lc.colnames:
                    flux_col = 'pdcsap_flux'
                elif 'sap_flux' in lc.colnames:
                    flux_col = 'sap_flux'
                else:
                    flux_col = 'flux'
                
                self.logger.info(f"  Using flux column: {flux_col}")
                self.is_tpf = False
            
            # Handle Target Pixel File
            elif 'PIXELS' in ext_names or 'TARGETTABLES' in ext_names:
                self.logger.info("✓ Detected Target Pixel File (TPF)")
                self.tpf = lk.read(self.fits_path)
                
                self.logger.info(f"  TPF dimensions: {self.tpf.shape[1]}x{self.tpf.shape[2]} pixels")
                self.logger.info(f"  TPF cadences: {len(self.tpf.time)}")
                
                # Try pipeline aperture first, then threshold
                try:
                    self.aperture_mask = self.tpf.pipeline_mask
                    lc = self.tpf.to_lightcurve(aperture_mask='pipeline')
                    self.logger.info("  ✓ Using PIPELINE aperture mask")
                except (AttributeError, ValueError) as e:
                    self.logger.warning(f"  Pipeline aperture unavailable: {e}")
                    self.logger.info("  Falling back to THRESHOLD aperture")
                    self.aperture_mask = self.tpf.create_threshold_mask(threshold=3)
                    lc = self.tpf.to_lightcurve(aperture_mask=self.aperture_mask)
                    self.logger.info("  ✓ Using THRESHOLD aperture (3-sigma)")
                
                self.logger.info(f"  Aperture: {self.aperture_mask.sum()} pixels selected")
                self.is_tpf = True
                flux_col = 'flux'
            
            else:
                raise ValueError(f"Unknown FITS format. Extensions: {ext_names}")
            
            # Store raw light curve for visualization
            self.light_curve_raw = lc.copy()
            
            # Quality filtering
            original_len = len(lc)
            if 'quality' in lc.colnames:
                lc = lc[lc.quality == 0]
                self.logger.info(f"  Quality filter: {original_len} → {len(lc)} cadences")
            
            # Remove NaNs and outliers
            lc = lc.remove_nans()
            self.logger.info(f"  After removing NaNs: {len(lc)} cadences")
            
            # Sigma clipping for outliers (5-sigma)
            lc = lc.remove_outliers(sigma=5)
            self.logger.info(f"  After 5-sigma clipping: {len(lc)} cadences")
            
            # Normalize
            lc = lc.normalize()
            self.logger.info(f"  ✓ Flux normalized (median = 1.0)")
            
            self.logger.info(f"✓ FITS loaded successfully: {len(lc)} good cadences")
            
            # Validate light curve quality
            if not self._validate_light_curve(lc):
                raise ValueError("Light curve validation failed - insufficient data quality")
            
            return lc
        
        except Exception as e:
            self.logger.error(f"✗ Failed to load FITS: {e}")
            raise
    
    def _run_bls(self, period_min: float = None, period_max: float = None) -> Dict[str, Optional[float]]:
        """
        Run Box Least Squares (BLS) periodogram to detect transits.
        
        Args:
            period_min: Minimum period to search (days). Auto-detected if None.
            period_max: Maximum period to search (days). Auto-detected if None.
        
        Returns:
            Dictionary with transit parameters:
            - period: Orbital period (days)
            - duration: Transit duration (days for Kepler, hours for TESS)
            - depth: Transit depth (fraction)
            - midpoint: Transit midpoint/epoch (BTJD/BKJD)
            - power: BLS peak power
            - snr: Signal-to-noise ratio
            - num_transits: Number of transits detected
            - period_err, duration_err, midpoint_err: Error estimates
        """
        self.logger.info("=" * 70)
        self.logger.info("STEP 2/5: Running BLS transit search")
        self.logger.info("=" * 70)
        
        result = {
            'period': None,
            'duration': None,
            'duration_hours': None,
            'depth': None,
            'midpoint': None,
            'power': None,
            'snr': None,
            'num_transits': None,
            'period_err': None,
            'duration_err': None,
            'midpoint_err': None,
            'depth_err': None
        }
        
        try:
            # Auto-detect period range from baseline
            timespan = (self.light_curve.time[-1] - self.light_curve.time[0]).value
            
            if period_min is None:
                # Minimum: 3 transits required or 0.5 days (whichever is larger)
                period_min = max(0.5, timespan / 100)
            
            if period_max is None:
                # Maximum: At least 3 transits visible
                period_max = min(timespan / 3, 20)
            
            self.logger.info(f"  Period search range: {period_min:.2f} - {period_max:.2f} days")
            self.logger.info(f"  Timespan: {timespan:.2f} days")
            
            # Adaptive resolution: finer at short periods
            if period_max - period_min < 5:
                resolution = 0.0001  # High resolution for short periods
            elif period_max - period_min < 10:
                resolution = 0.001   # Medium resolution
            else:
                resolution = 0.01    # Coarse resolution for long periods
            
            self.logger.info(f"  Period resolution: {resolution} days")
            
            # Run BLS periodogram
            self.logger.info("  Running BLS periodogram...")
            period_grid = np.arange(period_min, period_max, resolution)
            periodogram = self.light_curve.to_periodogram(method='bls', period=period_grid)
            
            # Check if significant peak detected
            power_threshold = 0.05  # Minimum power for transit detection
            max_power = periodogram.max_power
            
            if max_power < power_threshold:
                self.logger.warning(f"⚠ No significant transit detected (max power: {max_power:.4f} < {power_threshold})")
                return result
            
            # Extract parameters
            result['period'] = float(periodogram.period_at_max_power.value)
            result['duration'] = float(periodogram.duration_at_max_power.value)  # days
            result['duration_hours'] = result['duration'] * 24  # hours
            result['power'] = float(max_power)
            result['midpoint'] = float(periodogram.transit_time_at_max_power.value)
            
            self.logger.info(f"  ✓ Period detected: {result['period']:.6f} days")
            self.logger.info(f"  ✓ Duration: {result['duration_hours']:.4f} hours")
            self.logger.info(f"  ✓ BLS power: {result['power']:.4f}")
            self.logger.info(f"  ✓ Transit midpoint: {result['midpoint']:.6f} BTJD")
            
            # Fold light curve at detected period
            self.folded_lc = self.light_curve.fold(period=result['period'], epoch_time=result['midpoint'])
            
            # Compute transit depth (1 - minimum flux in folded LC)
            in_transit = np.abs(self.folded_lc.phase.value) < (result['duration'] / result['period'] / 2)
            if in_transit.sum() > 0:
                transit_flux = self.folded_lc.flux[in_transit]
                result['depth'] = float(1.0 - np.min(transit_flux.value))
                result['depth_err'] = float(np.std(transit_flux.value))
                
                self.logger.info(f"  ✓ Transit depth: {result['depth'] * 1e6:.2f} ppm")
            else:
                self.logger.warning("⚠ No in-transit points found in folded light curve")
                result['depth'] = None
            
            # Count transits
            timespan = (self.light_curve.time[-1] - self.light_curve.time[0]).value
            result['num_transits'] = int(timespan / result['period'])
            self.logger.info(f"  ✓ Number of transits: {result['num_transits']}")
            
            # Estimate SNR
            if result['depth'] is not None:
                noise = np.std(self.light_curve.flux.value)
                result['snr'] = result['depth'] / noise if noise > 0 else None
                self.logger.info(f"  ✓ SNR: {result['snr']:.2f}" if result['snr'] else "  ⚠ SNR: N/A")
            
            # Placeholder error estimates (10% for period, 5% for duration, 1% for midpoint)
            result['period_err'] = 0.01 * result['period']
            result['duration_err'] = 0.05 * result['duration']
            result['midpoint_err'] = 0.01
            
            self.logger.info("✓ BLS transit search completed")
            
        except Exception as e:
            self.logger.error(f"✗ BLS failed: {e}")
            self.logger.warning("  Returning null transit parameters")
        
        return result
    
    def _process_tess(self) -> Dict[str, Optional[float]]:
        """
        Process TESS parameters: 23 required features + derived parameters.
        
        Returns:
            Dictionary with TESS parameters (null if uncomputable)
        """
        self.logger.info("=" * 70)
        self.logger.info("STEP 3/5: Processing TESS parameters")
        self.logger.info("=" * 70)
        
        params = {}
        
        # ========== TRANSIT PARAMETERS FROM BLS ==========
        params['pl_orbper'] = self.bls_result.get('period')
        params['pl_orbpererr1'] = self.bls_result.get('period_err')
        params['pl_orbpererr2'] = self.bls_result.get('period_err')  # Symmetric for now
        
        params['pl_trandurh'] = self.bls_result.get('duration_hours')
        params['pl_trandurherr1'] = self.bls_result.get('duration_err')
        params['pl_trandurherr2'] = self.bls_result.get('duration_err')
        
        params['pl_trandep'] = self.bls_result.get('depth')
        params['pl_trandeperr1'] = self.bls_result.get('depth_err')
        params['pl_trandeperr2'] = self.bls_result.get('depth_err')
        
        params['pl_tranmid'] = self.bls_result.get('midpoint')
        params['pl_tranmiderr1'] = self.bls_result.get('midpoint_err')
        params['pl_tranmiderr2'] = self.bls_result.get('midpoint_err')
        
        self.logger.info(f"  Transit parameters: period={params['pl_orbper']}, depth={params['pl_trandep']}")
        
        # ========== STELLAR PARAMETERS FROM CATALOG ==========
        params['st_teff'] = self._safe_catalog_get('st_teff')
        params['st_tefferr1'] = self._safe_catalog_get('st_tefferr1')
        params['st_tefferr2'] = self._safe_catalog_get('st_tefferr2')
        
        params['st_rad'] = self._safe_catalog_get('st_rad')
        params['st_raderr1'] = self._safe_catalog_get('st_raderr1')
        params['st_raderr2'] = self._safe_catalog_get('st_raderr2')
        
        params['st_mass'] = self._safe_catalog_get('st_mass')
        params['st_masserr1'] = self._safe_catalog_get('st_masserr1')
        params['st_masserr2'] = self._safe_catalog_get('st_masserr2')
        
        params['st_logg'] = self._safe_catalog_get('st_logg')
        params['st_loggerr1'] = self._safe_catalog_get('st_loggerr1')
        params['st_loggerr2'] = self._safe_catalog_get('st_loggerr2')
        
        params['st_dist'] = self._safe_catalog_get('st_dist')
        params['st_disterr1'] = self._safe_catalog_get('st_disterr1')
        params['st_disterr2'] = self._safe_catalog_get('st_disterr2')
        
        params['st_tmag'] = self._safe_catalog_get('st_tmag')
        params['st_tmagerr1'] = self._safe_catalog_get('st_tmagerr1')
        params['st_tmagerr2'] = self._safe_catalog_get('st_tmagerr2')
        
        self.logger.info(f"  Stellar parameters: Teff={params['st_teff']}K, R={params['st_rad']}Rsun")
        
        # ========== DERIVED PARAMETERS ==========
        # Semi-major axis to stellar radius ratio
        duration_days = params['pl_trandurh'] / 24.0 if params['pl_trandurh'] else None
        params['a_over_rstar'] = self._compute_semi_major_axis_ratio(params['pl_orbper'], duration_days)
        
        if params['a_over_rstar']:
            self.logger.info(f"  Derived a/R_star: {params['a_over_rstar']:.2f}")
        else:
            self.logger.warning("  ⚠ Cannot compute a/R_star (missing period or duration)")
        
        # Planet equilibrium temperature
        params['pl_eqt'] = self._compute_equilibrium_temperature(
            params['st_teff'], 
            params['a_over_rstar']
        )
        
        if params['pl_eqt']:
            self.logger.info(f"  Derived T_eq: {params['pl_eqt']:.1f}K")
        else:
            self.logger.warning("  ⚠ Cannot compute T_eq (missing dependencies)")
        
        # Planet radius
        params['pl_rad'] = self._compute_planet_radius(params['st_rad'], params['pl_trandep'])
        
        if params['pl_rad']:
            params['pl_raderr1'] = self._propagate_error(
                params['pl_rad'],
                [(params['st_raderr1'], params['st_rad']), (params['pl_trandeperr1'] * 0.5, params['pl_trandep'])]
            )
            self.logger.info(f"  Derived planet radius: {params['pl_rad']:.2f} R_Earth")
        else:
            params['pl_rad'] = None
            params['pl_raderr1'] = None
            self.logger.warning("  ⚠ Cannot compute planet radius (missing stellar radius or depth)")
        
        self.logger.info("✓ TESS parameters processed")
        return params
    
    def _compute_engineered_tess(self, params: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Compute 8 engineered features for TESS.
        
        Args:
            params: Dictionary with base TESS parameters
        
        Returns:
            Dictionary with engineered features
        """
        self.logger.info("  Computing TESS engineered features...")
        
        eng = {}
        
        # 1. Period to duration ratio
        if params['pl_orbper'] and params['pl_trandurh']:
            eng['eng_period_duration_ratio'] = params['pl_orbper'] / (params['pl_trandurh'] / 24)
            self.logger.debug(f"    eng_period_duration_ratio: {eng['eng_period_duration_ratio']:.2f}")
        else:
            eng['eng_period_duration_ratio'] = None
        
        # 2. Planet to star radius ratio
        eng['eng_prad_srad_ratio'] = self._safe_sqrt(params['pl_trandep'])
        
        # 3. Transit shape parameter
        if params['pl_trandep'] and params['pl_trandurh']:
            eng['eng_transit_shape'] = self._safe_sqrt(params['pl_trandep']) / params['pl_trandurh']
        else:
            eng['eng_transit_shape'] = None
        
        # 4. Duration to period ratio
        if params['pl_trandurh'] and params['pl_orbper']:
            eng['eng_duration_period_ratio'] = (params['pl_trandurh'] / 24) / params['pl_orbper']
        else:
            eng['eng_duration_period_ratio'] = None
        
        # 5. Stellar density proxy
        if eng['eng_period_duration_ratio']:
            eng['eng_stellar_density'] = eng['eng_period_duration_ratio'] ** 3
        else:
            eng['eng_stellar_density'] = None
        
        # 6. Transit probability
        if params['st_rad'] and params['pl_orbper']:
            eng['eng_transit_probability'] = params['st_rad'] / (params['pl_orbper'] ** (2/3))
        else:
            eng['eng_transit_probability'] = None
        
        # 7. Temperature ratio
        if params['pl_eqt'] and params['st_teff']:
            eng['eng_temp_ratio'] = params['pl_eqt'] / params['st_teff']
        else:
            eng['eng_temp_ratio'] = None
        
        # 8. Signal quality
        if params['pl_trandep'] and params['st_tmag']:
            try:
                eng['eng_signal_quality'] = params['pl_trandep'] / (10 ** (params['st_tmag'] / 2.5))
            except (ValueError, OverflowError):
                eng['eng_signal_quality'] = None
        else:
            eng['eng_signal_quality'] = None
        
        # 9. Habitable zone indicator
        eng['eng_habitable_zone'] = self._is_in_habitable_zone(params['pl_eqt'])
        
        self.logger.info(f"  ✓ Engineered features computed: {sum(v is not None for v in eng.values())}/9")
        
        return eng
    
    def _process_kepler(self) -> Dict[str, Optional[float]]:
        """
        Process Kepler parameters: 14 required + 22 additional features.
        
        Returns:
            Dictionary with Kepler parameters (null if uncomputable)
        """
        self.logger.info("=" * 70)
        self.logger.info("STEP 3/5: Processing Kepler parameters")
        self.logger.info("=" * 70)
        
        params = {}
        
        # ========== TRANSIT PARAMETERS FROM BLS ==========
        params['koi_period'] = self.bls_result.get('period')
        params['koi_period_err1'] = self.bls_result.get('period_err')
        params['koi_period_err2'] = self.bls_result.get('period_err')
        
        params['koi_duration'] = self.bls_result.get('duration')  # days
        params['koi_duration_err1'] = self.bls_result.get('duration_err')
        params['koi_duration_err2'] = self.bls_result.get('duration_err')
        
        params['koi_depth'] = self.bls_result.get('depth')
        params['koi_depth_err1'] = self.bls_result.get('depth_err')
        params['koi_depth_err2'] = self.bls_result.get('depth_err')
        
        params['koi_time0'] = self.bls_result.get('midpoint')
        params['koi_time0bk_err1'] = self.bls_result.get('midpoint_err')
        params['koi_time0bk_err2'] = self.bls_result.get('midpoint_err')
        
        params['koi_num_transits'] = self.bls_result.get('num_transits')
        params['koi_model_snr'] = self.bls_result.get('snr')
        params['koi_max_mult_ev'] = self.bls_result.get('power')
        
        self.logger.info(f"  Transit parameters: period={params['koi_period']}, depth={params['koi_depth']}")
        
        # ========== STELLAR PARAMETERS FROM CATALOG ==========
        params['koi_steff'] = self._safe_catalog_get('koi_steff') or self._safe_catalog_get('st_teff')
        params['koi_steff_err1'] = self._safe_catalog_get('koi_steff_err1') or self._safe_catalog_get('st_tefferr1')
        params['koi_steff_err2'] = self._safe_catalog_get('koi_steff_err2') or self._safe_catalog_get('st_tefferr2')
        
        params['koi_srad'] = self._safe_catalog_get('koi_srad') or self._safe_catalog_get('st_rad')
        params['koi_srad_err1'] = self._safe_catalog_get('koi_srad_err1') or self._safe_catalog_get('st_raderr1')
        params['koi_srad_err2'] = self._safe_catalog_get('koi_srad_err2') or self._safe_catalog_get('st_raderr2')
        
        params['koi_smass'] = self._safe_catalog_get('koi_smass') or self._safe_catalog_get('st_mass')
        params['koi_smass_err1'] = self._safe_catalog_get('koi_smass_err1') or self._safe_catalog_get('st_masserr1')
        params['koi_smass_err2'] = self._safe_catalog_get('koi_smass_err2') or self._safe_catalog_get('st_masserr2')
        
        params['koi_slogg'] = self._safe_catalog_get('koi_slogg') or self._safe_catalog_get('st_logg')
        params['koi_slogg_err1'] = self._safe_catalog_get('koi_slogg_err1') or self._safe_catalog_get('st_loggerr1')
        params['koi_slogg_err2'] = self._safe_catalog_get('koi_slogg_err2') or self._safe_catalog_get('st_loggerr2')
        
        # Photometry
        params['koi_hmag'] = self._safe_catalog_get('koi_hmag')
        params['koi_imag'] = self._safe_catalog_get('koi_imag')
        params['koi_zmag'] = self._safe_catalog_get('koi_zmag')
        
        # Coordinates
        params['dec'] = self._safe_catalog_get('dec')
        params['koi_srho'] = self._safe_catalog_get('koi_srho')
        params['koi_srho_err1'] = self._safe_catalog_get('koi_srho_err1')
        
        self.logger.info(f"  Stellar parameters: Teff={params['koi_steff']}K, R={params['koi_srad']}Rsun")
        
        # ========== DERIVED PARAMETERS ==========
        # Planet to star radius ratio
        params['koi_ror'] = self._safe_sqrt(params['koi_depth'])
        params['koi_ror_err1'] = self._propagate_error(
            params['koi_ror'],
            [(params['koi_depth_err1'] * 0.5, params['koi_depth'])]
        ) if params['koi_depth'] else None
        
        # Planet radius in Earth radii
        params['koi_prad'] = self._compute_planet_radius(params['koi_srad'], params['koi_depth'])
        
        if params['koi_prad']:
            params['koi_prad_err1'] = self._propagate_error(
                params['koi_prad'],
                [(params['koi_srad_err1'], params['koi_srad']), (params['koi_depth_err1'] * 0.5, params['koi_depth'])]
            )
            params['koi_prad_err2'] = params['koi_prad_err1']
            self.logger.info(f"  Derived planet radius: {params['koi_prad']:.2f} R_Earth")
        else:
            params['koi_prad'] = None
            params['koi_prad_err1'] = None
            params['koi_prad_err2'] = None
        
        # Semi-major axis to stellar radius ratio
        params['koi_dor'] = self._compute_semi_major_axis_ratio(params['koi_period'], params['koi_duration'])
        
        if params['koi_dor']:
            params['koi_dor_err1'] = self._propagate_error(
                params['koi_dor'],
                [(params['koi_period_err1'], params['koi_period']), (params['koi_duration_err1'], params['koi_duration'])]
            )
            self.logger.info(f"  Derived a/R_star (koi_dor): {params['koi_dor']:.2f}")
        else:
            params['koi_dor'] = None
            params['koi_dor_err1'] = None
        
        # Planet equilibrium temperature
        params['koi_teq'] = self._compute_equilibrium_temperature(params['koi_steff'], params['koi_dor'])
        
        if params['koi_teq']:
            self.logger.info(f"  Derived T_eq: {params['koi_teq']:.1f}K")
        else:
            self.logger.warning("  ⚠ Cannot compute T_eq (missing dependencies)")
        
        # Insolation flux (relative to Earth)
        params['koi_insol'] = self._compute_insolation_flux(
            params['koi_steff'], 
            params['koi_srad'], 
            params['koi_dor']
        )
        
        if params['koi_insol']:
            self.logger.info(f"  Derived insolation: {params['koi_insol']:.2f} S_Earth")
        else:
            self.logger.warning("  ⚠ Cannot compute insolation (missing dependencies)")
        
        # Centroid offsets (rough approximation from PSF_CENTR if available)
        # This is a very rough estimate and should be taken with caution
        if hasattr(self.light_curve, 'centroid_col') and hasattr(self.light_curve, 'centroid_row'):
            params['koi_fwm_srao'] = float(np.mean(self.light_curve.centroid_col.value))
            params['koi_fwm_sra_err'] = float(np.std(self.light_curve.centroid_col.value))
            self.logger.debug(f"  Centroid offset approximated from light curve")
        else:
            params['koi_fwm_srao'] = None
            params['koi_fwm_sra_err'] = None
            self.logger.warning("  ⚠ Centroid data not available")
        
        self.logger.info("✓ Kepler parameters processed")
        return params
    
    def _compute_engineered_kepler(self, params: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """
        Compute 7 engineered features for Kepler.
        
        Args:
            params: Dictionary with base Kepler parameters
        
        Returns:
            Dictionary with engineered features
        """
        self.logger.info("  Computing Kepler engineered features...")
        
        eng = {}
        
        # 1. Period to duration ratio
        if params['koi_period'] and params['koi_duration']:
            eng['eng_period_duration_ratio'] = params['koi_period'] / params['koi_duration']
        else:
            eng['eng_period_duration_ratio'] = None
        
        # 2. Planet to star radius ratio (normalized)
        if params['koi_prad'] and params['koi_srad']:
            eng['eng_prad_srad_ratio'] = params['koi_prad'] / (params['koi_srad'] * 109)
        else:
            eng['eng_prad_srad_ratio'] = None
        
        # 3. SNR to depth ratio
        if params['koi_model_snr'] and params['koi_depth']:
            eng['eng_snr_depth_ratio'] = params['koi_model_snr'] / params['koi_depth']
        else:
            eng['eng_snr_depth_ratio'] = None
        
        # 4. Stellar density proxy
        if params['koi_srad'] and params['koi_smass']:
            eng['eng_stellar_density'] = (params['koi_srad'] ** 3) / params['koi_smass']
        else:
            eng['eng_stellar_density'] = None
        
        # 5. Transit probability
        if params['koi_srad'] and params['koi_dor']:
            eng['eng_transit_probability'] = params['koi_srad'] / params['koi_dor']
        else:
            eng['eng_transit_probability'] = None
        
        # 6. Temperature ratio
        eng['eng_temp_ratio'] = self._safe_div(params['koi_teq'], params['koi_steff'])
        
        # 7. Habitable zone indicator
        eng['eng_habitable_zone'] = self._is_in_habitable_zone(params['koi_teq'])
        
        self.logger.info(f"  ✓ Engineered features computed: {sum(v is not None for v in eng.values())}/7")
        
        return eng
    
    def _fig_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64 encoded string.
        
        Args:
            fig: Matplotlib figure object
        
        Returns:
            Base64 encoded string with data URI prefix
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_bytes = buffer.read()
        buffer.close()
        
        # Encode to base64 and add data URI prefix
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/png;base64,{base64_str}"
    
    def _generate_visualizations(self):
        """
        Generate visualization plots for frontend as Base64-encoded images.
        1. Aperture mask (if TPF)
        2. Raw light curve
        3. Flattened/normalized light curve
        4. Folded light curve with transit model
        
        Stores base64 data in self.visualization_paths for embedding in JSON DTO.
        """
        self.logger.info("=" * 70)
        self.logger.info("STEP 4/5: Generating visualizations (Base64 encoded)")
        self.logger.info("=" * 70)
        
        base_name = self.fits_path.stem
        
        # Clear previous visualization data
        self.visualization_paths = []
        
        # 1. Aperture mask (TPF only)
        if self.is_tpf and self.tpf is not None:
            try:
                fig = plt.figure(figsize=(8, 8))
                self.tpf.plot(aperture_mask=self.aperture_mask, ax=plt.gca())
                plt.title(f"Aperture Mask - {base_name}")
                
                # Convert to base64
                base64_data = self._fig_to_base64(fig)
                plt.close(fig)
                
                self.visualization_paths.append({
                    "type": "aperture",
                    "data": base64_data,
                    "filename": f"{base_name}_aperture.png"
                })
                self.logger.info(f"  ✓ Generated aperture plot (Base64)")
            except Exception as e:
                self.logger.warning(f"  ⚠ Failed to generate aperture plot: {e}")
                import traceback
                self.logger.warning(f"  Traceback: {traceback.format_exc()}")
        
        # 2. Raw light curve
        if self.light_curve_raw is not None:
            try:
                fig, ax = plt.subplots(figsize=(14, 5))
                self.light_curve_raw.plot(ax=ax, label='Raw Flux')
                ax.set_title(f"Raw Light Curve - {base_name}")
                ax.set_xlabel("Time (BTJD)")
                ax.set_ylabel("Flux")
                ax.legend()
                
                # Convert to base64
                base64_data = self._fig_to_base64(fig)
                plt.close(fig)
                
                self.visualization_paths.append({
                    "type": "raw_light_curve",
                    "data": base64_data,
                    "filename": f"{base_name}_raw_lc.png"
                })
                self.logger.info(f"  ✓ Generated raw light curve (Base64)")
            except Exception as e:
                self.logger.warning(f"  ⚠ Failed to generate raw LC plot: {e}")
                import traceback
                self.logger.warning(f"  Traceback: {traceback.format_exc()}")
        
        # 3. Flattened/normalized light curve
        if self.light_curve is not None:
            try:
                fig, ax = plt.subplots(figsize=(14, 5))
                self.light_curve.plot(ax=ax, label='Normalized Flux', color='blue', linewidth=0.5)
                ax.set_title(f"Normalized Light Curve - {base_name}")
                ax.set_xlabel("Time (BTJD)")
                ax.set_ylabel("Normalized Flux")
                ax.legend()
                ax.grid(alpha=0.3)
                
                # Convert to base64
                base64_data = self._fig_to_base64(fig)
                plt.close(fig)
                
                self.visualization_paths.append({
                    "type": "normalized_light_curve",
                    "data": base64_data,
                    "filename": f"{base_name}_normalized_lc.png"
                })
                self.logger.info(f"  ✓ Generated normalized light curve (Base64)")
            except Exception as e:
                self.logger.warning(f"  ⚠ Failed to generate normalized LC plot: {e}")
                import traceback
                self.logger.warning(f"  Traceback: {traceback.format_exc()}")
        
        # 4. Folded light curve with transit
        if self.folded_lc is not None and self.bls_result.get('period'):
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                self.folded_lc.scatter(ax=ax, s=1, c='black', alpha=0.3, label='Data')
                
                # Bin the folded light curve for clarity
                binned_lc = self.folded_lc.bin(time_bin_size=0.01)
                binned_lc.scatter(ax=ax, s=10, c='red', label='Binned (0.01 phase)')
                
                ax.set_title(f"Phase-Folded Light Curve - Period: {self.bls_result['period']:.6f} days")
                ax.set_xlabel("Phase")
                ax.set_ylabel("Normalized Flux")
                ax.legend()
                ax.grid(alpha=0.3)
                ax.set_xlim(-0.5, 0.5)
                
                # Convert to base64
                base64_data = self._fig_to_base64(fig)
                plt.close(fig)
                
                self.visualization_paths.append({
                    "type": "folded_light_curve",
                    "data": base64_data,
                    "filename": f"{base_name}_folded_lc.png"
                })
                self.logger.info(f"  ✓ Generated folded light curve (Base64)")
            except Exception as e:
                self.logger.warning(f"  ⚠ Failed to generate folded LC plot: {e}")
                import traceback
                self.logger.warning(f"  Traceback: {traceback.format_exc()}")
        
        self.logger.info(f"✓ Visualizations generated: {len(self.visualization_paths)} plots")
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate computed parameters for physical plausibility.
        
        Args:
            params: Dictionary of computed parameters
        
        Returns:
            Dictionary with 'warnings' and 'errors' lists
        """
        issues = {'warnings': [], 'errors': []}
        
        # Period checks
        if params.get('pl_orbper') or params.get('koi_period'):
            period = params.get('pl_orbper') or params.get('koi_period')
            if period and (period < 0.1 or period > 1000):
                issues['warnings'].append(f"Unusual period: {period:.2f} days")
        
        # Depth checks
        depth_key = 'pl_trandep' if self.mission == 'tess' else 'koi_depth'
        if params.get(depth_key):
            depth = params[depth_key]
            if depth < 0 or depth > 0.5:
                issues['warnings'].append(f"Unusual depth: {depth:.4f} (expect 0-0.5)")
        
        # Temperature checks
        if params.get('st_teff') or params.get('koi_steff'):
            t_eff = params.get('st_teff') or params.get('koi_steff')
            if t_eff and (t_eff < 2000 or t_eff > 50000):
                issues['warnings'].append(f"Unusual stellar temp: {t_eff:.0f}K")
        
        # Equilibrium temperature checks
        t_eq = params.get('pl_eqt') or params.get('koi_teq')
        if t_eq and (t_eq < 0 or t_eq > 5000):
            issues['warnings'].append(f"Unusual equilibrium temp: {t_eq:.0f}K")
        
        # Radius checks
        rad_key = 'pl_rad' if self.mission == 'tess' else 'koi_prad'
        if params.get(rad_key):
            radius = params[rad_key]
            if radius < 0.1 or radius > 30:  # Earth radii
                issues['warnings'].append(f"Unusual planet radius: {radius:.2f} R_Earth")
        
        # Check for all nulls
        non_null = sum(1 for v in params.values() if v is not None)
        if non_null == 0:
            issues['errors'].append("All parameters are null - check catalog data")
        
        return issues
    
    def _extract_model_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model-specific parameters for ML inference.
        
        For TESS (TOI): 23 parameters for recall model
        For Kepler (KOI): 14 parameters for precision model
        
        Args:
            params: Full parameter dictionary
        
        Returns:
            Dictionary with only model-required parameters
        """
        if self.mission == 'tess':
            # TESS TOI recall model parameters (23 parameters)
            toi_params = [
                'pl_trandurherr1', 'eng_transit_probability', 'pl_trandurherr2',
                'eng_prad_srad_ratio', 'pl_orbpererr1', 'pl_orbper',
                'eng_period_duration_ratio', 'eng_duration_period_ratio',
                'pl_tranmiderr1', 'pl_trandeperr2', 'pl_tranmid',
                'pl_trandep', 'pl_trandurh', 'pl_trandeperr1',
                'st_tmagerr2', 'st_disterr2', 'st_loggerr2',
                'st_disterr1', 'st_dist', 'st_teff',
                'st_tmagerr1', 'st_rad', 'st_tefferr2'
            ]
            model_data = {param: params.get(param) for param in toi_params}
            
        elif self.mission == 'kepler':
            # Kepler KOI precision model parameters (14 parameters)
            koi_params = [
                'koi_prad', 'koi_dor', 'koi_ror', 'koi_num_transits',
                'koi_duration_err1', 'koi_prad_err1', 'koi_period_err2',
                'koi_srad_err1', 'koi_insol', 'eng_transit_probability',
                'koi_model_snr', 'koi_srho', 'koi_max_mult_ev', 'koi_teq'
            ]
            model_data = {param: params.get(param) for param in koi_params}
        
        else:
            model_data = {}
        
        return model_data
    
    def _build_json_output(self) -> dict:
        """
        Build JSON output as a dictionary (DTO pattern).
        
        Returns:
            Dictionary containing mission, parameters, visualizations, and model data
        """
        self.logger.info("=" * 70)
        self.logger.info("STEP 5/5: Building JSON output")
        self.logger.info("=" * 70)
        
        # Convert None to null and handle numpy types
        def serialize(obj):
            if obj is None:
                return None
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean parameters
        cleaned_params = {k: serialize(v) for k, v in self.parameters['parameters'].items()}
        
        # Validate parameters
        validation = self._validate_parameters(cleaned_params)
        if validation['warnings']:
            self.logger.warning("  Parameter validation warnings:")
            for warning in validation['warnings']:
                self.logger.warning(f"    - {warning}")
        if validation['errors']:
            self.logger.error("  Parameter validation errors:")
            for error in validation['errors']:
                self.logger.error(f"    - {error}")
        
        # Extract model-specific data
        model_data = self._extract_model_data(cleaned_params)
        
        output_data = [{
            "mission": self.mission,
            "parameters": cleaned_params,
            "visualizations": self.visualization_paths,
            "data": [model_data]  # Model-ready parameters in array format
        }]
        
        # Count non-null parameters
        non_null = sum(1 for v in cleaned_params.values() if v is not None)
        total = len(cleaned_params)
        
        # Count model data parameters
        model_non_null = sum(1 for v in model_data.values() if v is not None)
        model_total = len(model_data)
        
        self.logger.info(f"  ✓ Built {non_null}/{total} parameters")
        self.logger.info(f"  ✓ Null parameters: {total - non_null}")
        self.logger.info(f"  ✓ Visualizations included: {len(self.visualization_paths)}")
        self.logger.info(f"  ✓ Model data: {model_non_null}/{model_total} parameters ready for ML inference")
        
        return output_data
    
    def process(self, return_json: bool = True, print_json: bool = True) -> dict:
        """
        Main processing pipeline.
        
        Args:
            return_json: If True, returns JSON dict (DTO pattern). If False, returns None.
            print_json: If True, prints formatted JSON to console.
        
        Returns:
            Dictionary containing mission, parameters, visualizations, and model data (DTO)
        
        Raises:
            Exception: If processing fails at any step
        """
        try:
            # Step 1: Load FITS
            self.light_curve = self._load_fits()
            
            # Step 2: Run BLS
            self.bls_result = self._run_bls()
            
            # Step 3: Process mission-specific parameters
            if self.mission == 'tess':
                tess_params = self._process_tess()
                eng_params = self._compute_engineered_tess(tess_params)
                tess_params.update(eng_params)
                self.parameters = {"mission": "tess", "parameters": tess_params}
            
            elif self.mission == 'kepler':
                kepler_params = self._process_kepler()
                eng_params = self._compute_engineered_kepler(kepler_params)
                kepler_params.update(eng_params)
                self.parameters = {"mission": "kepler", "parameters": kepler_params}
            
            # Step 4: Generate visualizations
            self._generate_visualizations()
            
            # Step 5: Build JSON output (DTO)
            output_json = self._build_json_output()
            
            self.logger.info("=" * 70)
            self.logger.info("✓✓✓ PROCESSING COMPLETE ✓✓✓")
            self.logger.info("=" * 70)
            
            # Print JSON to console if requested
            if print_json:
                print("\n" + "=" * 70)
                print("JSON OUTPUT:")
                print("=" * 70)
                print(json.dumps(output_json, indent=2))
                print("=" * 70 + "\n")
            
            return output_json if return_json else None
        
        except Exception as e:
            self.logger.error(f"✗✗✗ PROCESSING FAILED: {e} ✗✗✗")
            raise


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    # Example TESS usage
    processor = ExoplanetParameterProcessor(
        fits_path=r"C:\Users\drind\Downloads\tess2025014115807-s0088-0000000114018671-0285-s_lc.fits",
        mission="tess",
        catalog={
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
            'st_tmagerr1': 0.05,
            'st_tmagerr2': 0.05
        }
    )
    
    # Process and get JSON output (DTO pattern - no file saved!)
    # JSON will be automatically printed to console
    result = processor.process()
    
    # Access the data
    print(f"\n✓ Mission: {result[0]['mission']}")
    print(f"✓ Parameters computed: {len(result[0]['parameters'])}")
    print(f"✓ Visualizations generated: {len(result[0]['visualizations'])}")
    print(f"✓ Model data ready: {len(result[0]['data'][0])} parameters")
    
    # Use the model data for ML inference
    model_input = result[0]['data'][0]
    print(f"\n✓ Ready to send to ML model: {list(model_input.keys())[:5]}...")
