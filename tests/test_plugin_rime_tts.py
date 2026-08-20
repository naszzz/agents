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
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import aiohttp
import pytest

from livekit.agents import APIConnectionError, APIStatusError
from livekit.plugins.rime.tts import TTS, _rime_v1_ws_url
from livekit.plugins.rime.websocket_v1 import (
    RIME_V1_SUBPROTOCOL,
    RimeV1Audio,
    RimeV1Connection,
    RimeV1Flush,
    RimeV1Input,
    RimeV1Started,
    RimeV1StartOptions,
    RimeV1Text,
)

pytestmark = pytest.mark.plugin("rime")


class _FakeWebSocket:
    def __init__(self, *, protocol: str | None = RIME_V1_SUBPROTOCOL) -> None:
        self.protocol = protocol
        self.closed = False
        self.sent: list[dict[str, Any]] = []
        self._incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self._sent_event = asyncio.Event()

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))
        self._sent_event.set()

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        if timeout is None:
            return await self._incoming.get()
        return await asyncio.wait_for(self._incoming.get(), timeout)

    async def close(self) -> bool:
        self.closed = True
        return True

    def exception(self) -> BaseException | None:
        return None

    def push(self, value: dict[str, Any]) -> None:
        self._incoming.put_nowait(
            aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(value), None)
        )

    async def wait_for_sent(self, count: int) -> None:
        while len(self.sent) < count:
            self._sent_event.clear()
            if len(self.sent) < count:
                await asyncio.wait_for(self._sent_event.wait(), 1.0)


class _FakeSession:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket
        self.calls = 0
        self.url: str | None = None
        self.headers: dict[str, str] | None = None
        self.protocols: tuple[str, ...] | None = None

    async def ws_connect(
        self,
        url: str,
        *,
        headers: dict[str, str],
        protocols: tuple[str, ...],
        autoping: bool,
    ) -> _FakeWebSocket:
        self.calls += 1
        self.url = url
        self.headers = headers
        self.protocols = protocols
        assert autoping is True
        return self.websocket


async def _connect(websocket: _FakeWebSocket) -> tuple[RimeV1Connection, _FakeSession]:
    websocket.push({"ready": {"protocol": 1, "languages": ["eng"]}})
    session = _FakeSession(websocket)
    connection = await RimeV1Connection.connect(
        session=cast(aiohttp.ClientSession, session),
        url="wss://engine.test/ws",
        api_key="test-key",
        timeout=1.0,
    )
    return connection, session


def _start_options() -> RimeV1StartOptions:
    return RimeV1StartOptions(
        speaker="lyra",
        language="eng",
        sample_rate=24000,
        time_scale_factor=1.1,
        max_tokens=128,
        text_lookahead_tokens=4,
    )


def test_rime_v1_url_builder() -> None:
    assert _rime_v1_ws_url("https://engine.test") == "wss://engine.test/ws"
    assert _rime_v1_ws_url("wss://engine.test/base/") == "wss://engine.test/base/ws"
    assert _rime_v1_ws_url("ws://engine.test/ws") == "ws://engine.test/ws"
    with pytest.raises(ValueError):
        _rime_v1_ws_url("engine.test")


def test_rime_v1_options_are_opt_in() -> None:
    legacy = TTS(api_key="test-key", model="coda", use_websocket=True)
    assert legacy.sample_rate == 22050
    assert legacy.capabilities.aligned_transcript is True

    v1 = TTS(
        api_key="test-key",
        base_url="https://engine.test",
        model="coda",
        use_websocket=True,
        websocket_api="rime.v1",
    )
    assert v1.sample_rate == 24000
    assert v1.capabilities.aligned_transcript is False

    explicit_rate = TTS(
        api_key="test-key",
        base_url="https://engine.test",
        model="coda",
        sample_rate=22050,
        use_websocket=True,
        websocket_api="rime.v1",
    )
    assert explicit_rate.sample_rate == 22050


def test_rime_v1_rejects_unsupported_configuration() -> None:
    with pytest.raises(ValueError, match="base_url"):
        TTS(
            api_key="test-key",
            model="coda",
            use_websocket=True,
            websocket_api="rime.v1",
        )
    with pytest.raises(ValueError, match="only model='coda'"):
        TTS(
            api_key="test-key",
            base_url="https://engine.test",
            model="arcana",
            use_websocket=True,
            websocket_api="rime.v1",
        )
    with pytest.raises(ValueError, match="speed_alpha"):
        TTS(
            api_key="test-key",
            base_url="https://engine.test",
            model="coda",
            speed_alpha=1.1,
            use_websocket=True,
            websocket_api="rime.v1",
        )


@pytest.mark.asyncio
async def test_connection_waits_for_ready_and_streams_flush_resume() -> None:
    websocket = _FakeWebSocket()
    connection, session = await _connect(websocket)

    assert session.headers == {"Authorization": "Api-Key test-key"}
    assert session.protocols == (RIME_V1_SUBPROTOCOL,)

    async def inputs() -> AsyncIterator[RimeV1Input]:
        yield RimeV1Text("Hello ")
        yield RimeV1Flush()
        yield RimeV1Text("again.")

    outputs: list[RimeV1Started | RimeV1Audio] = []

    async def consume() -> None:
        async for output in connection.synthesize(start=_start_options(), inputs=inputs()):
            outputs.append(output)

    task = asyncio.create_task(consume())
    await websocket.wait_for_sent(5)
    context_id = websocket.sent[0]["contextId"]
    websocket.push({"contextId": context_id, "started": {"requestId": "request-1"}})
    websocket.push(
        {
            "contextId": context_id,
            "audio": base64.b64encode(b"\x01\x00\x02\x00").decode(),
        }
    )
    websocket.push({"contextId": context_id, "done": {}})
    await task

    assert [next(key for key in frame if key != "contextId") for frame in websocket.sent] == [
        "start",
        "text",
        "flush",
        "text",
        "end",
    ]
    assert websocket.sent[0]["start"] == {
        "speaker": "lyra",
        "language": "eng",
        "text": "",
        "audioParameters": {
            "audioFormat": "audio/pcm",
            "samplingRate": 24000,
            "timeScaleFactor": 1.1,
        },
        "codaParameters": {"maxTokens": 128, "textLookaheadTokens": 4},
    }
    assert outputs == [
        RimeV1Started(context_id=context_id, request_id="request-1"),
        RimeV1Audio(context_id=context_id, data=b"\x01\x00\x02\x00"),
    ]
    assert connection.reusable is True


@pytest.mark.asyncio
async def test_context_error_is_mapped_and_connection_remains_reusable() -> None:
    websocket = _FakeWebSocket()
    connection, _ = await _connect(websocket)

    async def inputs() -> AsyncIterator[RimeV1Input]:
        yield RimeV1Text("Hello")

    async def consume() -> None:
        async for _ in connection.synthesize(start=_start_options(), inputs=inputs()):
            pass

    task = asyncio.create_task(consume())
    await websocket.wait_for_sent(3)
    context_id = websocket.sent[0]["contextId"]
    websocket.push(
        {
            "contextId": context_id,
            "error": {
                "kind": "invalid_input",
                "message": "speaker not found",
                "requestId": "request-2",
            },
        }
    )

    with pytest.raises(APIStatusError) as exc_info:
        await task
    assert exc_info.value.status_code == 400
    assert exc_info.value.request_id == "request-2"
    assert exc_info.value.retryable is False
    assert connection.reusable is True


@pytest.mark.asyncio
async def test_protocol_failure_discards_connection() -> None:
    websocket = _FakeWebSocket()
    connection, _ = await _connect(websocket)

    async def inputs() -> AsyncIterator[RimeV1Input]:
        yield RimeV1Text("Hello")

    async def consume() -> None:
        async for _ in connection.synthesize(start=_start_options(), inputs=inputs()):
            pass

    task = asyncio.create_task(consume())
    await websocket.wait_for_sent(3)
    context_id = websocket.sent[0]["contextId"]
    websocket.push({"contextId": context_id, "started": {"requestId": "request-3"}})
    websocket.push({"contextId": context_id, "audio": "not-base64"})

    with pytest.raises(APIConnectionError, match="invalid base64"):
        await task
    assert connection.reusable is False


@pytest.mark.asyncio
async def test_cancelled_context_can_reuse_connection() -> None:
    websocket = _FakeWebSocket()
    connection, _ = await _connect(websocket)
    hold_input = asyncio.Event()

    async def inputs() -> AsyncIterator[RimeV1Input]:
        yield RimeV1Text("Hello")
        await hold_input.wait()

    started = asyncio.Event()

    async def consume() -> None:
        async for output in connection.synthesize(start=_start_options(), inputs=inputs()):
            if isinstance(output, RimeV1Started):
                started.set()

    task = asyncio.create_task(consume())
    await websocket.wait_for_sent(2)
    context_id = websocket.sent[0]["contextId"]
    websocket.push({"contextId": context_id, "started": {"requestId": "request-4"}})
    await asyncio.wait_for(started.wait(), 1.0)

    task.cancel()
    await websocket.wait_for_sent(3)
    assert websocket.sent[-1] == {"contextId": context_id, "cancel": {}}
    websocket.push({"contextId": context_id, "cancelled": {}})

    with pytest.raises(asyncio.CancelledError):
        await task
    assert connection.reusable is True


@pytest.mark.asyncio
async def test_wrong_selected_subprotocol_fails_connection() -> None:
    websocket = _FakeWebSocket(protocol=None)
    websocket.push({"ready": {"protocol": 1}})
    session = _FakeSession(websocket)

    with pytest.raises(APIConnectionError, match="selected WebSocket subprotocol"):
        await RimeV1Connection.connect(
            session=cast(aiohttp.ClientSession, session),
            url="wss://engine.test/ws",
            api_key="test-key",
            timeout=1.0,
        )
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_tts_adapter_emits_engine_audio_and_request_id() -> None:
    websocket = _FakeWebSocket()
    websocket.push({"ready": {"protocol": 1, "languages": ["eng"]}})
    session = _FakeSession(websocket)
    rime = TTS(
        api_key="test-key",
        base_url="https://engine.test",
        model="coda",
        http_session=cast(aiohttp.ClientSession, session),
        use_websocket=True,
        websocket_api="rime.v1",
    )
    stream = rime.stream()
    stream.push_text("Hello")
    stream.end_input()

    await websocket.wait_for_sent(4)
    context_id = websocket.sent[0]["contextId"]
    websocket.push({"contextId": context_id, "started": {"requestId": "engine-request"}})
    websocket.push(
        {
            "contextId": context_id,
            "audio": base64.b64encode(b"\x01\x00" * 4800).decode(),
        }
    )
    websocket.push({"contextId": context_id, "done": {}})

    audio = [event async for event in stream]
    assert audio
    assert audio[0].request_id == "engine-request"
    assert audio[0].segment_id == context_id
    assert audio[-1].is_final is True
    assert audio[0].frame.sample_rate == 24000

    second_stream = rime.stream()
    second_stream.push_text("Again")
    second_stream.end_input()
    await websocket.wait_for_sent(8)
    second_context_id = websocket.sent[4]["contextId"]
    websocket.push({"contextId": second_context_id, "started": {"requestId": "engine-request-2"}})
    websocket.push(
        {
            "contextId": second_context_id,
            "audio": base64.b64encode(b"\x02\x00" * 4800).decode(),
        }
    )
    websocket.push({"contextId": second_context_id, "done": {}})
    second_audio = [event async for event in second_stream]

    assert second_audio[0].request_id == "engine-request-2"
    assert second_context_id != context_id
    assert session.calls == 1
    await rime.aclose()
