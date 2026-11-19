"""Shared native JSON transport. Response policies belong to each adapter."""

import json

import httpx

from ..model import ModelError


class NativeChatTransport:
    def __init__(self, api_key, model_name, *, client=None):
        if not api_key.strip() or not model_name.strip():
            raise ModelError("LIVE_MODEL_NOT_CONFIGURED")
        self._key = api_key
        self.model_name = model_name
        self._client = client

    async def complete(self, messages, tool_schemas, *, timeout_s, required_tool=None):
        payload = dict(
            model=self.model_name,
            messages=messages,
            tools=tool_schemas,
            thinking={"type": "disabled"},
            response_format={"type": "json_object"},
            max_tokens=1400,
            temperature=0,
        )
        if required_tool is not None:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": required_tool},
            }
        if self._client is not None:
            return await self._request(self._client, payload, timeout_s)
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
            return await self._request(client, payload, timeout_s)

    async def _request(self, client, payload, timeout_s):
        async with client.stream(
            "POST",
            "https://api.deepseek.com/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._key}"},
            timeout=timeout_s,
        ) as response:
            if response.status_code != 200:
                raise ModelError(f"PROVIDER_HTTP_{response.status_code}")
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > 256000:
                    raise ModelError("PROVIDER_RESPONSE_TOO_LARGE")
            try:
                return json.loads(body)
            except ValueError:
                raise ModelError("PROVIDER_JSON_INVALID") from None
