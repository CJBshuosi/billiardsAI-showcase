"""One owned local encoder process, one active request, no hidden queue."""

import asyncio
import multiprocessing
import os
from pathlib import Path

from .embeddings import LocalEncoder, validate_vector


class EmbeddingUnavailable(RuntimeError):
    pass


class EmbeddingBusy(EmbeddingUnavailable):
    pass


def _encoder_main(connection, root, factory):
    for name in (
        "DEEPSEEK_API_KEY",
        "DATABASE_URL",
        "TEST_DATABASE_URL",
        "CUEORBIT_RULES_DATABASE_URL",
    ):
        os.environ.pop(name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        encoder = factory(Path(root))
        validate_vector(encoder.embed_query("warmup"))
        connection.send(("ready", None, None))
    except Exception:
        connection.send(("startup_error", None, "MODEL_UNAVAILABLE"))
        connection.close()
        return
    while True:
        try:
            message = connection.recv()
        except (EOFError, OSError):
            break
        if message is None:
            break
        identity, text = message
        try:
            connection.send(
                ("result", identity, validate_vector(encoder.embed_query(text)))
            )
        except ValueError as error:
            code = (
                "ENCODER_TOKEN_LIMIT"
                if str(error) == "ENCODER_TOKEN_LIMIT"
                else "ENCODER_FAILURE"
            )
            connection.send(("invalid", identity, code))
        except Exception:
            connection.send(("error", identity, "ENCODER_FAILURE"))
    connection.close()


class EmbeddingWorker:
    def __init__(self, root: Path, *, encoder_factory=LocalEncoder, startup_timeout=20):
        self.root = root
        self.factory = encoder_factory
        self.startup_timeout = startup_timeout
        self.process = None
        self.connection = None
        self._ready = False
        self._busy = False
        self._counter = 0

    @property
    def ready(self):
        return self._ready and self.process is not None and self.process.is_alive()

    async def start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(
            target=_encoder_main,
            args=(child, str(self.root), self.factory),
            daemon=True,
        )
        self.process.start()
        child.close()
        try:
            message = await asyncio.wait_for(
                asyncio.to_thread(parent.recv), self.startup_timeout
            )
            if message[0] != "ready":
                raise EmbeddingUnavailable("MODEL_UNAVAILABLE")
            self._ready = True
        except BaseException:
            self._abort()
            raise

    def _abort(self):
        self._ready = False
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
        if self.connection is not None:
            self.connection.close()

    async def embed_query(self, text: str) -> list[float]:
        if not self.ready:
            raise EmbeddingUnavailable("EMBEDDING_UNAVAILABLE")
        if self._busy:
            raise EmbeddingBusy("EMBEDDING_BUSY")
        self._busy = True
        self._counter += 1
        identity = self._counter
        try:
            self.connection.send((identity, text))
            status, returned, value = await asyncio.to_thread(self.connection.recv)
            if returned != identity:
                raise EmbeddingUnavailable("LATE_ENCODER_RESULT")
            if status == "invalid":
                raise ValueError(value)
            if status != "result":
                raise EmbeddingUnavailable("ENCODER_FAILURE")
            return validate_vector(value)
        except asyncio.CancelledError:
            self._abort()
            raise
        except (EOFError, OSError):
            self._abort()
            raise EmbeddingUnavailable("EMBEDDING_UNAVAILABLE") from None
        finally:
            self._busy = False

    async def aclose(self):
        self._ready = False
        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.connection.send(None)
                except (OSError, EOFError):
                    pass
                await asyncio.to_thread(self.process.join, 0.5)
            if self.process.is_alive():
                self.process.terminate()
            await asyncio.to_thread(self.process.join, 1)
        if self.connection is not None:
            self.connection.close()
