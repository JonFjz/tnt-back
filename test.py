import lightkurve as lk
lcfs = lk.search_lightcurve("KIC 11446443", mission="Kepler", author="Kepler")\
         .download_all(flux_column="pdcsap_flux")
lc = lcfs.stitch().remove_nans().remove_outliers(sigma=7).flatten(401)
print(lc)