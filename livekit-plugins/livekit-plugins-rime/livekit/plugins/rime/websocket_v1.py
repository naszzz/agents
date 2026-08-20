# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp

from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, utils

RIME_V1_SUBPROTOCOL = "rime.v1.json"
_CANCEL_TIMEOUT = 1.0


class _WebSocket(Protocol):
    @property
    def protocol(self) -> str | None: ...

    @property
    def closed(self) -> bool: ...

    async def send_str(self, data: str) -> None: ...

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage: ...

    async def close(self) -> bool: ...

    def exception(self) -> BaseException | None: ...


@dataclass(frozen=True)
class RimeV1StartOptions:
    speaker: str
    language: str
    sample_rate: int
    time_scale_factor: float | None = None
    max_tokens: int | None = None
    text_lookahead_tokens: int | None = None

    def envelope(self, context_id: str) -> dict[str, object]:
        audio_parameters: dict[str, object] = {
            "audioFormat": "audio/pcm",
            "samplingRate": self.sample_rate,
        }
        if self.time_scale_factor is not None:
            audio_parameters["timeScaleFactor"] = self.time_scale_factor

        coda_parameters: dict[str, object] = {}
        if self.max_tokens is not None:
            coda_parameters["maxTokens"] = self.max_tokens
        if self.text_lookahead_tokens is not None:
            coda_parameters["textLookaheadTokens"] = self.text_lookahead_tokens

        start: dict[str, object] = {
            "speaker": self.speaker,
            "language": self.language,
            "text": "",
            "audioParameters": audio_parameters,
        }
        if coda_parameters:
            start["codaParameters"] = coda_parameters

        return {"contextId": context_id, "start": start}


@dataclass(frozen=True)
class RimeV1Text:
    text: str


@dataclass(frozen=True)
class RimeV1Flush:
    pass


RimeV1Input = RimeV1Text | RimeV1Flush


@dataclass(frozen=True)
class RimeV1Started:
    context_id: str
    request_id: str


@dataclass(frozen=True)
class RimeV1Audio:
    context_id: str
    data: bytes


RimeV1Output = RimeV1Started | RimeV1Audio


def _status_for_error_kind(kind: str) -> tuple[int, bool]:
    return {
        "invalid_input": (400, False),
        "unauthenticated": (401, False),
        "resource_exhausted": (429, True),
        "unimplemented": (501, False),
        "unavailable": (503, True),
        "internal": (500, True),
    }.get(kind, (500, True))


def _engine_error(data: object) -> APIStatusError:
    if not isinstance(data, dict):
        return APIStatusError(
            "Rime v1 returned an invalid error payload",
            status_code=500,
            retryable=False,
        )

    kind = data.get("kind")
    message = data.get("message")
    request_id = data.get("requestId")
    kind_text = kind if isinstance(kind, str) else "unknown"
    status_code, retryable = _status_for_error_kind(kind_text)
    message_text = message if isinstance(message, str) else "Rime synthesis failed"
    return APIStatusError(
        message_text,
        status_code=status_code,
        request_id=request_id if isinstance(request_id, str) else None,
        body={"kind": kind_text},
        retryable=retryable,
    )


class RimeV1Connection:
    """One ready Rime v1 WebSocket with at most one active context."""

    def __init__(self, websocket: _WebSocket) -> None:
        self._ws = websocket
        self._healthy = True
        self._ready = False
        self._active_context_id: str | None = None

    @classmethod
    async def connect(
        cls,
        *,
        session: aiohttp.ClientSession,
        url: str,
        api_key: str,
        timeout: float,
    ) -> RimeV1Connection:
        try:
            websocket = await asyncio.wait_for(
                session.ws_connect(
                    url,
                    headers={"Authorization": f"Api-Key {api_key}"},
                    protocols=(RIME_V1_SUBPROTOCOL,),
                    autoping=True,
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Rime v1 WebSocket connection timed out") from None
        except aiohttp.ClientResponseError as exc:
            raise APIStatusError(
                exc.message,
                status_code=exc.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as exc:
            raise APIConnectionError("Rime v1 WebSocket connection failed") from exc

        connection = cls(websocket)
        try:
            if websocket.protocol != RIME_V1_SUBPROTOCOL:
                raise connection._protocol_error(
                    f"Rime selected WebSocket subprotocol {websocket.protocol!r}, "
                    f"expected {RIME_V1_SUBPROTOCOL!r}"
                )
            await connection._wait_ready(timeout)
            return connection
        except BaseException:
            await connection.close()
            raise

    @property
    def reusable(self) -> bool:
        return (
            self._healthy
            and self._ready
            and self._active_context_id is None
            and not self._ws.closed
        )

    async def close(self) -> None:
        self._healthy = False
        if not self._ws.closed:
            await self._ws.close()

    async def _wait_ready(self, timeout: float) -> None:
        try:
            message = await asyncio.wait_for(self._receive_envelope(), timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError("Rime v1 did not send ready before the timeout") from None

        payload_name, payload = self._payload(message)
        context_id = message.get("contextId", "")
        if payload_name == "error" and context_id == "":
            raise _engine_error(payload)
        if payload_name != "ready" or context_id not in ("", None):
            raise self._protocol_error("Rime v1 must send ready before context events")
        if not isinstance(payload, dict) or payload.get("protocol") != 1:
            raise self._protocol_error("Rime v1 returned an unsupported protocol version")
        self._ready = True

    async def synthesize(
        self,
        *,
        start: RimeV1StartOptions,
        inputs: AsyncIterable[RimeV1Input],
    ) -> AsyncIterator[RimeV1Output]:
        if not self.reusable:
            raise APIConnectionError(
                "Rime v1 connection is not ready for a new context", retryable=False
            )

        context_id = utils.shortuuid()
        self._active_context_id = context_id
        terminal = False
        started = False
        send_task = asyncio.create_task(
            self._send_context(context_id=context_id, start=start, inputs=inputs),
            name="rime_v1_send_context",
        )

        try:
            while True:
                message = await self._receive_or_sender_failure(send_task)
                payload_name, payload = self._payload(message)
                response_context = message.get("contextId", "")

                if payload_name == "error" and response_context == "":
                    self._healthy = False
                    raise _engine_error(payload)
                if response_context != context_id:
                    raise self._protocol_error(
                        "Rime v1 returned an event for an unexpected context"
                    )

                if payload_name == "started":
                    if started or not isinstance(payload, dict):
                        raise self._protocol_error("Rime v1 returned an invalid started event")
                    request_id = payload.get("requestId")
                    if not isinstance(request_id, str):
                        raise self._protocol_error("Rime v1 started event has no requestId")
                    started = True
                    yield RimeV1Started(context_id=context_id, request_id=request_id)
                elif payload_name == "audio":
                    if not started or not isinstance(payload, str):
                        raise self._protocol_error("Rime v1 returned audio before started")
                    try:
                        audio = base64.b64decode(payload, validate=True)
                    except (binascii.Error, ValueError):
                        raise self._protocol_error(
                            "Rime v1 returned invalid base64 audio"
                        ) from None
                    yield RimeV1Audio(context_id=context_id, data=audio)
                elif payload_name == "done":
                    if not started:
                        raise self._protocol_error("Rime v1 returned done before started")
                    terminal = True
                    return
                elif payload_name == "cancelled":
                    terminal = True
                    raise APIStatusError(
                        "Rime synthesis was cancelled",
                        status_code=499,
                        request_id=None,
                        retryable=False,
                    )
                elif payload_name == "error":
                    terminal = True
                    raise _engine_error(payload)
                else:
                    raise self._protocol_error(
                        f"Rime v1 returned unexpected {payload_name!r} event"
                    )
        except asyncio.CancelledError:
            await self._stop_task(send_task)
            terminal = await self._cancel_context(context_id)
            raise
        except BaseException:
            if not terminal:
                self._healthy = False
            raise
        finally:
            await self._stop_task(send_task)
            self._active_context_id = None
            if not terminal:
                self._healthy = False

    async def _send_context(
        self,
        *,
        context_id: str,
        start: RimeV1StartOptions,
        inputs: AsyncIterable[RimeV1Input],
    ) -> None:
        await self._send(start.envelope(context_id))
        async for item in inputs:
            if isinstance(item, RimeV1Text):
                if item.text:
                    await self._send({"contextId": context_id, "text": item.text})
            else:
                await self._send({"contextId": context_id, "flush": {}})
        await self._send({"contextId": context_id, "end": {}})

    async def _cancel_context(self, context_id: str) -> bool:
        if self._ws.closed:
            self._healthy = False
            return False

        try:
            await self._send({"contextId": context_id, "cancel": {}})
            while True:
                message = await asyncio.wait_for(self._receive_envelope(), _CANCEL_TIMEOUT)
                payload_name, payload = self._payload(message)
                response_context = message.get("contextId", "")
                if payload_name == "error" and response_context == "":
                    self._healthy = False
                    return False
                if response_context != context_id:
                    self._healthy = False
                    return False
                if payload_name in ("cancelled", "done", "error"):
                    return True
                if payload_name not in ("started", "audio"):
                    self._healthy = False
                    return False
        except BaseException:
            self._healthy = False
            return False

    async def _receive_or_sender_failure(self, send_task: asyncio.Task[None]) -> dict[str, Any]:
        if send_task.done():
            send_task.result()
            return await self._receive_envelope()

        receive_task = asyncio.create_task(self._receive_envelope(), name="rime_v1_receive_context")
        try:
            done, _ = await asyncio.wait(
                (send_task, receive_task), return_when=asyncio.FIRST_COMPLETED
            )
            if send_task in done:
                send_task.result()
            return await receive_task
        except BaseException:
            if not receive_task.done():
                await self._stop_task(receive_task)
            raise

    async def _receive_envelope(self) -> dict[str, Any]:
        try:
            message = await self._ws.receive()
        except Exception as exc:
            self._healthy = False
            raise APIConnectionError("Rime v1 WebSocket receive failed") from exc

        if message.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSING,
        ):
            self._healthy = False
            raise APIConnectionError("Rime v1 WebSocket closed unexpectedly")
        if message.type == aiohttp.WSMsgType.ERROR:
            self._healthy = False
            raise APIConnectionError("Rime v1 WebSocket failed") from self._ws.exception()
        if message.type != aiohttp.WSMsgType.TEXT:
            raise self._protocol_error("Rime v1 JSON mode received a non-text frame")

        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            raise self._protocol_error("Rime v1 returned malformed JSON") from None
        if not isinstance(value, dict):
            raise self._protocol_error("Rime v1 returned a non-object envelope")
        return value

    def _payload(self, message: dict[str, Any]) -> tuple[str, object]:
        names = [
            name
            for name in ("ready", "started", "audio", "done", "cancelled", "error")
            if name in message
        ]
        if len(names) != 1:
            raise self._protocol_error("Rime v1 envelope must contain exactly one event")
        name = names[0]
        return name, message[name]

    async def _send(self, value: dict[str, object]) -> None:
        if self._ws.closed:
            self._healthy = False
            raise APIConnectionError("Rime v1 WebSocket is closed")
        try:
            await self._ws.send_str(json.dumps(value, separators=(",", ":")))
        except Exception as exc:
            self._healthy = False
            raise APIConnectionError("Rime v1 WebSocket send failed") from exc

    def _protocol_error(self, message: str) -> APIConnectionError:
        self._healthy = False
        return APIConnectionError(message, retryable=False)

    @staticmethod
    async def _stop_task(task: asyncio.Task[object]) -> None:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
