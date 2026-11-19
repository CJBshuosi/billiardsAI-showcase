import hashlib
import json
import subprocess
from pathlib import Path

from .contracts import MANIFEST, MEDIA_DURATIONS


class MediaCatalog:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.entries = json.loads((self.root / "manifest.json").read_text())

    def resolve(self, media_id: str) -> Path:
        if media_id not in MANIFEST or media_id not in self.entries:
            raise ValueError("MEDIA_UNAVAILABLE")
        item = self.entries[media_id]
        if item["filename"] != MANIFEST[media_id]:
            raise ValueError("MEDIA_UNAVAILABLE")
        path = (self.root / item["filename"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("MEDIA_UNAVAILABLE")
        return path

    def validate(self):
        try:
            if set(self.entries) != set(MANIFEST):
                raise ValueError("manifest mismatch")
            for media_id, item in self.entries.items():
                path = self.resolve(media_id)
                if item["duration_ms"] != MEDIA_DURATIONS[media_id]:
                    raise ValueError("manifest duration mismatch")
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError("media hash mismatch")
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "json",
                        str(path),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=3,
                )
                duration = float(json.loads(result.stdout)["format"]["duration"]) * 1000
                if abs(duration - item["duration_ms"]) > 70:
                    raise ValueError("media duration mismatch")
        except (OSError, subprocess.SubprocessError, KeyError, TypeError):
            raise ValueError("MEDIA_VALIDATION_FAILED") from None
