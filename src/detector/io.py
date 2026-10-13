"""Read images straight out of the dataset zips (no extraction needed)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image

from .transforms import to_rgb


class ZipImageReader:
    """Keeps one open handle per zip; safe to create one per worker process."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._zips: dict[str, zipfile.ZipFile] = {}

    def read(self, zip_name: str, member: str):
        if not zip_name:  # loose file under data_dir (e.g. Grok outputs)
            return to_rgb(Image.open(self.data_dir / member))
        z = self._zips.get(zip_name)
        if z is None:
            z = self._zips[zip_name] = zipfile.ZipFile(self.data_dir / zip_name)
        return to_rgb(Image.open(io.BytesIO(z.read(member))))
