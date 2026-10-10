"""Read the PGCB CSV into an hourly table.

Source: Shekh & Rafi, "Hourly Electricity Generation, Demand, Load
Shedding, and Fuel Mix Dataset for Bangladesh (2015–2026)", Mendeley Data
v1, doi:10.17632/vpk8spw2mm.1, CC BY 4.0. Compiled from the PGCB ERP portal.

Facts from the dataset description (checked 2026-10-10):
- `datetime` is local Asia/Dhaka time.
- `demand_mw` is *observed* demand: during load shedding it is censored by
  the supply available, so it is served demand, not latent demand.
- Early rows (2015–2026 in places) include half-hourly readings; we keep
  the on-the-hour readings only.
"""

import hashlib
from pathlib import Path

import pandas as pd

from powerfc import config

# Mendeley Data v1 file, CC BY 4.0. The hash pins the exact bytes this
# project was built and evaluated on.
SOURCE_URL = (
    "https://data.mendeley.com/public-files/datasets/vpk8spw2mm/files/"
    "d0e7e374-f839-43cf-91d2-63c9f12b259e/file_downloaded"
)
SOURCE_SHA256 = "05c2350f5f3eb398d0b5ca9e9fcdcad31509fa6e1340f8fcef15ec64cbee4dc3"

NUMERIC = [
    "generation_mw",
    "demand_mw",
    "load_shedding",
    "gas",
    "liquid_fuel",
    "coal",
    "hydro",
    "solar",
    "wind",
    "india_bheramara_hvdc",
    "india_tripura",
    "india_adani",
    "nepal",
]


def read_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df[df["datetime"].dt.minute == 0]  # drop half-hourly readings
    df = df.drop_duplicates("datetime", keep="first").sort_values("datetime")
    df[NUMERIC] = df[NUMERIC].astype("float64")
    df["remarks"] = df["remarks"].astype("string")
    return df.reset_index(drop=True)


def fetch_raw(path: Path = config.RAW_CSV, url: str = SOURCE_URL, sha256: str = SOURCE_SHA256) -> dict:
    """Download the PGCB CSV if missing and verify its checksum."""
    import requests

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path} sha256 {digest} != expected {sha256}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
