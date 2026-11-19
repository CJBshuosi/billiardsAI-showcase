import argparse
import asyncio
from pathlib import Path

from .fixtures import load_builtin_bundles, load_bundle
from .repository import SessionRepository
from .settings import Settings


async def run(command, path):
    repo = SessionRepository(Settings.from_env().database_url)
    try:
        await repo.initialize()
        items = (
            load_builtin_bundles()
            if command == "seed-demo"
            else (load_bundle(Path(path)),)
            if command == "import"
            else ()
        )
        for bundle in items:
            print(bundle.bundle_id, (await repo.import_bundle(bundle)).status)
    finally:
        await repo.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init-db", "import", "seed-demo"])
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.command == "import" and not args.path:
        parser.error("import requires bundle path")
    asyncio.run(run(args.command, args.path))
