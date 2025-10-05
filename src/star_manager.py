import re, requests
from urllib.parse import quote_plus
from src.fits_downloader import LatestFITSFetcher
from src.stellar_data_fetcher import StarMetaFetcher

class StarProcessor:
    def __init__(self, mission, target_id, oi_lookup, parameters=None,  file_path=None):
        self.mission = mission
        self.target_id = target_id
        self.parameters = parameters
        self.oi_lookup  = oi_lookup
        self.file_path = file_path
        self.response= None
        self.found = None

        if self.mission.lower() == "tess":
            self.found = self.foundPlanet()

        if self.oi_lookup:
            self.response = self.checkOI()
          

        if self.response is None:
            self.getFitsData()
            self.getData() 
           

        # Check if the target exists in TOI and KOI
        # If not for KOI, get the data for TOI, download it, saves which was the latest that it used and save it in the data folder
        # For TESS, the user should put the file themselves
        # We give them the option to tweak parameters for the light curve

    def checkOI(self):
        """
        Query NASA Exoplanet Archive for TOI/KOI rows for this target.
        """
        base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        mission = (self.mission or "").strip().lower()
        target = (self.target_id or "").strip()

        self.oi_source = None
        self.oi_row = None
        self.found = False

        if mission == "tess":
            # Accept TIC ID (e.g., 25155310 or "TIC 25155310") OR TOI (e.g., "TOI-700.01")
            m = re.search(r'(?i)toi[\s-]*([0-9]+(?:\.[0-9]+)?)', target)
            if m:
                toi_val = m.group(1)  # e.g., "700.01"
                q = f"select * from toi where toi={toi_val}"
            else:
                m = re.search(r'(?i)(?:tic[:\\s-]*)?([0-9]+)', target)
                if not m:
                    raise ValueError("For TESS, use TIC ID (e.g., 25155310) or TOI (e.g., TOI-700.01).")
                tid = m.group(1)
                q = f"select * from toi where tid={tid}"

            url = f"{base}?query={quote_plus(q)}&format=json"
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            data = r.json()
            rows = data if isinstance(data, list) else data.get("data", [])
            self.oi_source = "TOI"
            self.oi_row = rows[0] if rows else None
            self.found = self.oi_row is not None
            return self.oi_row

        elif mission == "kepler":
            # Accept KIC (e.g., 11446443) OR KOI (e.g., "KOI-351.01" or "K00001.01")
            m = re.search(r'(?i)k(?:oi[-\\s]*)?([0-9]+)(?:[.-]([0-9]+))?', target)
            if m and (m.group(2) is not None or 'koi' in target.lower()):
                # Build kepoi_name like K01234.01
                n = int(m.group(1))
                c = m.group(2)
                if c is not None:
                    kepoi = f"K{n:05d}.{int(c):02d}"
                    q = f"select * from cumulative where kepoi_name='{kepoi}'"
                else:
                    kepoi_prefix = f"K{n:05d}.%"
                    q = f"select * from cumulative where kepoi_name like '{kepoi_prefix}'"
            else:
                m = re.search(r'(?i)(?:kic[:\\s-]*)?([0-9]+)', target)
                if not m:
                    raise ValueError("For Kepler, use KIC (e.g., 11446443) or KOI (e.g., KOI-351.01).")
                kepid = m.group(1)
                q = f"select * from cumulative where kepid={kepid}"

            url = f"{base}?query={quote_plus(q)}&format=json"
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            data = r.json()
            rows = data if isinstance(data, list) else data.get("data", [])
            self.oi_source = "KOI"
            self.oi_row = rows[0] if rows else None
            self.found = self.oi_row is not None
            return self.oi_row

        else:
            raise ValueError("Unsupported mission. Use 'Kepler' or 'TESS'.")

    def getFitsData(self):
        if self.file_path is None:
            mission = (self.mission or "").strip().lower()
            process = LatestFITSFetcher(
                mission=mission, 
                target_id=self.target_id,
                outdir="./tmp",
                prefer="spoc",  
                verbose=True,
            )
            path = process.download()
            self.file_path = str(path) if path else None
            # We'll call getData() separately if needed, not automatically here


    def getData(self):   
        mission = (self.mission or "").strip().lower()
        meta = StarMetaFetcher(mission, self.target_id) 
        self.stellar = meta.fetch_stellar()
        self.products = meta.list_mast_products()

    def getBestParametersAI(self):
        # Implement the logic to get the best parameters using AI (gpt)
        pass

    def foundPlanet(self) -> int:
        """
        Return 1 if TIC has ≥1 confirmed planet (NASA Exoplanet Archive),
        else 0. Assumes self.target_id is 'TIC 123...' or '123...'.
        """
        import re, requests

        base = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        t = str(self.target_id).strip()
        m = re.match(r'^(?:TIC\s*)?(\d+)$', t, flags=re.I)
        if not m:
            return 0  # you said inputs are standardized; bail if not a TIC

        tic = int(m.group(1))

        def _count(sql: str) -> int:
            r = requests.get(base, params={"query": sql, "format": "json"}, timeout=15)
            if not r.ok:
                return 0
            j = r.json()
            # TAP JSON looks like: {"fields":[{"name":"n"}], "data":[[<count>]]}
            if isinstance(j, dict) and "data" in j and j["data"]:
                return int(j["data"][0][0])
            # Rare/legacy: list-of-dicts fallback
            if isinstance(j, list) and j and isinstance(j[0], dict):
                return int(next(iter(j[0].values())))
            return 0

        try:
            if _count(f"SELECT COUNT(*) AS n FROM pscomppars WHERE tic_id={tic}") > 0:
                return 1
            if _count(f"SELECT COUNT(*) AS n FROM stellarhosts WHERE tic_id={tic} AND sy_pnum>0") > 0:
                return 1
            return 0
        except Exception:
            return 0


       