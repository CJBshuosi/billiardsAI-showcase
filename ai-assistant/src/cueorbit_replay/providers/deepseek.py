import httpx
from pydantic import ValidationError

from ..contracts import AnswerDraft, ModelTurn
from ..model import ModelError
from .decoding import decode_turn
from .native import NativeChatTransport


class DeepSeekModel:
    def __init__(
        self, api_key: str, model_name: str, *, client: httpx.AsyncClient | None = None
    ):
        self.model_name = model_name
        self._transport = NativeChatTransport(api_key, model_name, client=client)

    async def next_turn(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        try:
            data = await self._transport.complete(
                messages, tool_schemas, timeout_s=timeout_s, required_tool=required_tool
            )
            return decode_turn(data, AnswerDraft, ModelTurn)
        except (httpx.TimeoutException, TimeoutError):
            raise TimeoutError("MODEL_TIMEOUT") from None
        except ModelError:
            raise
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise ModelError(
                "PROVIDER_NETWORK_ERROR"
                if isinstance(error, httpx.HTTPError)
                else "MODEL_ENVELOPE_INVALID"
            ) from None
