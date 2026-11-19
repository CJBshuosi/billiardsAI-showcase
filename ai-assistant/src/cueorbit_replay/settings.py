import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from sqlalchemy.engine import make_url

PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def guarded_url(dsn: str):
    url = make_url(dsn)
    if (
        url.drivername != "postgresql+asyncpg"
        or not re.fullmatch(r"cueorbit_replay_m1(?:_[a-z0-9_]+)?", url.database or "")
        or url.query
    ):
        raise ValueError("DEDICATED_DATABASE_REQUIRED")
    return url


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)
    model_name: str = field(default="", repr=False)
    host: str = "127.0.0.1"
    port: int = 8765
    test_database_url: str = field(default="", repr=False)
    rules_enabled: bool = False
    rules_database_url: str = field(default="", repr=False)
    rules_encoder_dir: Path | None = field(default=None, repr=False)
    rules_retrieval: str = field(default="vector_glossary_v1", repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        env_file: Path | None | Literal["project"] = "project",
        environ: Mapping[str, str] | None = None,
    ):
        """Read this checkout's .env, then overlay explicit process values.

        env_file=None disables file reads; tests may inject a temporary path and
        an isolated mapping. Empty process values deliberately override file
        values. Parsing does not mutate os.environ or expand unrelated values.
        """
        path = PROJECT_ENV_FILE if env_file == "project" else env_file
        values = (
            dotenv_values(dotenv_path=path, interpolate=False)
            if path is not None
            else {}
        )
        process = os.environ if environ is None else environ

        def value(name: str) -> str:
            return process[name] if name in process else values.get(name) or ""

        return cls(
            database_url=value("DATABASE_URL"),
            test_database_url=value("TEST_DATABASE_URL"),
            api_key=value("DEEPSEEK_API_KEY"),
            model_name=value("DEEPSEEK_MODEL"),
            rules_enabled=value("CUEORBIT_RULES_ENABLED").strip().lower()
            in ("1", "true", "yes"),
            rules_database_url=value("CUEORBIT_RULES_DATABASE_URL"),
            rules_encoder_dir=Path(value("CUEORBIT_ENCODER_DIR"))
            if value("CUEORBIT_ENCODER_DIR")
            else None,
            rules_retrieval=value("CUEORBIT_RULES_RETRIEVAL") or "vector_glossary_v1",
        )

    @property
    def live_ready(self):
        return bool(self.api_key.strip() and self.model_name.strip())
