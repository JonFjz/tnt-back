import pandas as pd
import lightkurve as lk
from lightkurve import search_targetpixelfile
from lightkurve import TessTargetPixelFile

class StarManager:
    def __init__(self):
        pass

    def fetch_lightcurve(self, mission: str, target_id: str):
        pixelFile = lk.search_targetpixelfile(target_id, author=mission, cadence="long", quarter=4).download()
        lc = pixelFile.to_lightcurve(aparture_mask=pixelFile.pipeline_mask)
        return lc.to_pandas()

