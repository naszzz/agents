# Copyright 202 LiveKit, Inc.
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
import copy
import json
import os
import weakref
from collections.abc import AsyncIterable
from dataclasses import dataclass, replace
from typing import Literal, TypeVar
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from . import _websocket_v1
from .langs import TTSLangs
from .log import logger
from .models import DefaultCodaVoice, DefaultMistVoice, TTSModels

CODA_MODEL_TIMEOUT = 60 * 4
MIST_MODEL_TIMEOUT = 30
RIME_BASE_URL = "https://users.rime.ai/v1/rime-tts"
RIME_WS_BASE_URL = "wss://users-ws.rime.ai"
NUM_CHANNELS = 1
_OptionT = TypeVar("_OptionT")


@dataclass
class _TTSOptions:
    model: TTSModels | str
    speaker: str
    coda_options: _CodaOptions | None = None
    mist_options: _MistOptions | None = None


@dataclass
class _CodaOptions:
    repetition_penalty: NotGivenOr[float] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    top_p: NotGivenOr[float] = NOT_GIVEN
    max_tokens: NotGivenOr[int] = NOT_GIVEN
    lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN
    sample_rate: NotGivenOr[int] = NOT_GIVEN
    speed_alpha: NotGivenOr[float] = NOT_GIVEN
    time_scale_factor: NotGivenOr[float] = NOT_GIVEN


@dataclass
class _MistOptions:
    lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN
    sample_rate: NotGivenOr[int] = NOT_GIVEN
    speed_alpha: NotGivenOr[float] = NOT_GIVEN
    reduce_latency: NotGivenOr[bool] = NOT_GIVEN
    pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    time_scale_factor: NotGivenOr[float] = NOT_GIVEN


def _is_mist_model(model: TTSModels | str) -> bool:
    return "mist" in model


def _warn_if_arcana(model: NotGivenOr[TTSModels | str]) -> None:
    if is_given(model) and model == "arcana":
        logger.warning('Rime Arcana is no longer supported. Use model="coda" instead.')


def _timeout_for_model(model: TTSModels | str) -> int:
    if model == "coda":
        return CODA_MODEL_TIMEOUT
    return MIST_MODEL_TIMEOUT


def _model_params(opts: _TTSOptions) -> dict[str, object]:
    """Per-model option fields shared between the HTTP body and the WS query string."""
    params: dict[str, object] = {}
    if opts.model == "coda" and opts.coda_options is not None:
        co = opts.coda_options
        if is_given(co.lang):
            params["lang"] = co.lang
        if is_given(co.repetition_penalty):
            params["repetition_penalty"] = co.repetition_penalty
        if is_given(co.temperature):
            params["temperature"] = co.temperature
        if is_given(co.top_p):
            params["top_p"] = co.top_p
        if is_given(co.max_tokens):
            params["max_tokens"] = co.max_tokens
        if is_given(co.speed_alpha):
            params["speedAlpha"] = co.speed_alpha
        if is_given(co.time_scale_factor):
            params["timeScaleFactor"] = co.time_scale_factor
    elif _is_mist_model(opts.model) and opts.mist_options is not None:
        mo = opts.mist_options
        if is_given(mo.lang):
            params["lang"] = mo.lang
        if is_given(mo.speed_alpha):
            params["speedAlpha"] = mo.speed_alpha
        if is_given(mo.pause_between_brackets):
            params["pauseBetweenBrackets"] = mo.pause_between_brackets
        if is_given(mo.phonemize_between_brackets):
            params["phonemizeBetweenBrackets"] = mo.phonemize_between_brackets
        # time_scale_factor is supported by mistv3 but not mistv2.
        if is_given(mo.time_scale_factor) and opts.model != "mistv2":
            params["timeScaleFactor"] = mo.time_scale_factor
    return params


def _check_time_scale_factor_supported(
    model: TTSModels | str, time_scale_factor: NotGivenOr[float]
) -> None:
    if is_given(time_scale_factor) and model == "mistv2":
        raise ValueError(
            "time_scale_factor is not supported by the mistv2 model; use mistv3 or coda."
        )


class _TTSBase(tts.TTS[Literal["rime_tts_event"]]):
    def __init__(
        self,
        *,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        # Coda options
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        # Shared by mistv3 and coda (HTTP and v1 WebSocket)
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        # Supported by all models; the only speed param that works over WebSocket
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        # Mistv2 options
        sample_rate: int = 22050,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        use_websocket: bool = False,
        websocket_protocol: Literal["ws3", "v1"] = "ws3",
        segment: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        if websocket_protocol not in ("ws3", "v1"):
            raise ValueError('websocket_protocol must be either "ws3" or "v1"')
        if websocket_protocol == "v1" and not use_websocket:
            raise ValueError('websocket_protocol="v1" requires use_websocket=True')
        if websocket_protocol == "v1" and not is_given(base_url):
            raise ValueError('websocket_protocol="v1" requires an explicit model base_url')

        if is_given(base_url):
            # Infer streaming mode from URL prefix; an explicit use_websocket=True still wins.
            use_websocket = use_websocket or base_url.startswith(("ws://", "wss://"))
            resolved_base_url = base_url
        else:
            resolved_base_url = RIME_WS_BASE_URL if use_websocket else RIME_BASE_URL

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=use_websocket,
                aligned_transcript=use_websocket and websocket_protocol == "ws3",
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        resolved_api_key = api_key if is_given(api_key) else os.environ.get("RIME_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Rime API key is required, either as argument or set RIME_API_KEY environmental variable"  # noqa: E501
            )
        self._api_key = resolved_api_key

        _warn_if_arcana(model)
        if is_given(model):
            resolved_model = model
            model_is_explicit = True
        else:
            resolved_model = "coda"
            model_is_explicit = False

        _check_time_scale_factor_supported(resolved_model, time_scale_factor)
        if websocket_protocol == "v1":
            if resolved_model != "coda":
                raise ValueError('websocket_protocol="v1" only supports model="coda"')
            if is_given(speed_alpha):
                raise ValueError("speed_alpha is not supported by the Rime v1 WebSocket protocol")

        if not is_given(speaker):
            if not model_is_explicit:
                speaker = "astra"
            elif _is_mist_model(resolved_model):
                speaker = DefaultMistVoice
            elif resolved_model == "coda":
                speaker = DefaultCodaVoice
            else:
                speaker = "astra"

        self._opts = _TTSOptions(
            model=resolved_model,
            speaker=speaker,
        )
        if resolved_model == "coda":
            self._opts.coda_options = _CodaOptions(
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                lang=lang,
                sample_rate=sample_rate,
                speed_alpha=speed_alpha,
                time_scale_factor=time_scale_factor,
            )
        elif _is_mist_model(resolved_model):
            self._opts.mist_options = _MistOptions(
                lang=lang,
                sample_rate=sample_rate,
                speed_alpha=speed_alpha,
                reduce_latency=reduce_latency,
                pause_between_brackets=pause_between_brackets,
                phonemize_between_brackets=phonemize_between_brackets,
                time_scale_factor=time_scale_factor,
            )
        self._session = http_session
        self._base_url = resolved_base_url
        self._use_websocket = use_websocket
        self._websocket_protocol = websocket_protocol
        self._segment = segment if is_given(segment) else "bySentence"

        self._total_timeout = _timeout_for_model(resolved_model)

        self._streams: weakref.WeakSet[SynthesizeStream] = weakref.WeakSet()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._retired_pools: set[utils.ConnectionPool[aiohttp.ClientWebSocketResponse]] = set()
        self._pool_stream_counts: dict[
            utils.ConnectionPool[aiohttp.ClientWebSocketResponse], int
        ] = {}
        self._pool_close_tasks: set[asyncio.Task[None]] = set()
        self._pool = self._new_pool()

    def _new_pool(self) -> utils.ConnectionPool[aiohttp.ClientWebSocketResponse]:
        if self._websocket_protocol == "v1":
            websocket_url = self._ws_url()

            async def _connect(timeout: float) -> aiohttp.ClientWebSocketResponse:
                return await _websocket_v1.connect(
                    self._ensure_session(),
                    websocket_url=websocket_url,
                    api_key=self._api_key,
                    timeout=timeout,
                )

            connect_cb = _connect
        else:
            connect_cb = self._connect_ws

        return utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=connect_cb,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )

    def _retain_pool(self, pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse]) -> None:
        self._pool_stream_counts[pool] = self._pool_stream_counts.get(pool, 0) + 1

    def _release_pool(self, pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse]) -> None:
        stream_count = self._pool_stream_counts.get(pool, 0)
        if stream_count > 1:
            self._pool_stream_counts[pool] = stream_count - 1
            return

        self._pool_stream_counts.pop(pool, None)
        self._schedule_retired_pool_close(pool)

    def _retire_pool(self, pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse]) -> None:
        self._retired_pools.add(pool)
        if self._pool_stream_counts.get(pool, 0) == 0:
            self._schedule_retired_pool_close(pool)

    def _schedule_retired_pool_close(
        self, pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse]
    ) -> None:
        if pool not in self._retired_pools:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._retired_pools.remove(pool)
        task = loop.create_task(pool.aclose())
        self._pool_close_tasks.add(task)
        task.add_done_callback(self._on_retired_pool_closed)

    def _on_retired_pool_closed(self, task: asyncio.Task[None]) -> None:
        self._pool_close_tasks.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            logger.warning(
                "failed to close a retired Rime WebSocket pool",
                extra={"exception_type": type(error).__name__},
            )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Rime"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def _ws_url(self) -> str:
        if self._websocket_protocol == "v1":
            return _websocket_v1.websocket_url(self._base_url)
        params: dict[str, object] = {
            "speaker": self._opts.speaker,
            "modelId": self._opts.model,
            "audioFormat": "pcm",
            "samplingRate": self._sample_rate,
            "segment": self._segment,
            **_model_params(self._opts),
        }
        encoded = {
            k: ("true" if v else "false") if isinstance(v, bool) else v for k, v in params.items()
        }
        return f"{self._base_url}/ws3?{urlencode(encoded)}"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(
            session.ws_connect(
                self._ws_url(), headers={"Authorization": f"Bearer {self._api_key}"}
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        if self._websocket_protocol == "v1":
            await _websocket_v1.close(ws)
            return
        try:
            await ws.send_str(json.dumps({"operation": "eos"}))
            try:
                await asyncio.wait_for(ws.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            logger.warning(f"Error during Rime WS close sequence: {e}")
        finally:
            await ws.close()

    def prewarm(self) -> None:
        if self._use_websocket:
            self._pool.prewarm()

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        if not self._use_websocket:
            raise RuntimeError(
                "Rime TTS streaming requires use_websocket=True at construction time"
            )
        s = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(s)
        if self._websocket_protocol == "v1":
            pool = s._pool
            self._retain_pool(pool)

            def _release_pool(_: asyncio.Task[None]) -> None:
                self._release_pool(pool)

            s._task.add_done_callback(_release_pool)
        return s

    async def aclose(self) -> None:
        for s in list(self._streams):
            await s.aclose()
        self._streams.clear()
        await self._pool.aclose()
        for pool in list(self._retired_pools):
            await pool.aclose()
        self._retired_pools.clear()
        if self._pool_close_tasks:
            await asyncio.gather(*list(self._pool_close_tasks), return_exceptions=True)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        if self._use_websocket:
            raise RuntimeError(
                "Rime TTS one-shot synthesize requires use_websocket=False at construction time"
            )
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def _update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN,
        # Coda parameters
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        # Mistv2 parameters
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        _warn_if_arcana(model)
        effective_model = model if is_given(model) else self._opts.model
        _check_time_scale_factor_supported(effective_model, time_scale_factor)
        if self._websocket_protocol == "v1":
            if effective_model != "coda":
                raise ValueError('websocket_protocol="v1" only supports model="coda"')
            if is_given(speed_alpha):
                raise ValueError("speed_alpha is not supported by the Rime v1 WebSocket protocol")

        # The WS URL is bound when its pool connects. Refresh the pool when that URL changes.
        prev_ws_url = self._ws_url() if self._use_websocket else None
        if is_given(base_url):
            self._base_url = base_url
        if is_given(model):
            self._opts.model = model
            self._total_timeout = _timeout_for_model(model)

            if model == "coda" and self._opts.coda_options is None:
                self._opts.coda_options = _CodaOptions()
            elif _is_mist_model(model) and self._opts.mist_options is None:
                self._opts.mist_options = _MistOptions()

        if is_given(speaker):
            self._opts.speaker = speaker
        if is_given(sample_rate):
            self._sample_rate = sample_rate

        if self._opts.model == "coda" and self._opts.coda_options is not None:
            if is_given(repetition_penalty):
                self._opts.coda_options.repetition_penalty = repetition_penalty
            if is_given(temperature):
                self._opts.coda_options.temperature = temperature
            if is_given(top_p):
                self._opts.coda_options.top_p = top_p
            if is_given(max_tokens):
                self._opts.coda_options.max_tokens = max_tokens
            if is_given(lang):
                self._opts.coda_options.lang = lang
            if is_given(sample_rate):
                self._opts.coda_options.sample_rate = sample_rate
            if is_given(speed_alpha):
                self._opts.coda_options.speed_alpha = speed_alpha
            if is_given(time_scale_factor):
                self._opts.coda_options.time_scale_factor = time_scale_factor

        elif _is_mist_model(self._opts.model) and self._opts.mist_options is not None:
            if is_given(lang):
                self._opts.mist_options.lang = lang
            if is_given(sample_rate):
                self._opts.mist_options.sample_rate = sample_rate
            if is_given(speed_alpha):
                self._opts.mist_options.speed_alpha = speed_alpha
            if is_given(reduce_latency):
                self._opts.mist_options.reduce_latency = reduce_latency
            if is_given(pause_between_brackets):
                self._opts.mist_options.pause_between_brackets = pause_between_brackets
            if is_given(phonemize_between_brackets):
                self._opts.mist_options.phonemize_between_brackets = phonemize_between_brackets
            if is_given(time_scale_factor):
                self._opts.mist_options.time_scale_factor = time_scale_factor

        if prev_ws_url is not None and self._ws_url() != prev_ws_url:
            if self._websocket_protocol == "v1":
                old_pool = self._pool
                self._pool = self._new_pool()
                self._retire_pool(old_pool)
            else:
                self._pool.invalidate()


class TTS(_TTSBase):
    """Rime TTS adapter for the HTTP and legacy ws3 transports.

    The v1 Coda constructor arguments remain available for compatibility. New
    Coda WebSocket integrations should use :class:`CodaTTS`.
    """

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        speed_alpha: NotGivenOr[float] = NOT_GIVEN,
        reduce_latency: NotGivenOr[bool] = NOT_GIVEN,
        pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        self._update_options(
            model=model,
            speaker=speaker,
            lang=lang,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            sample_rate=sample_rate,
            time_scale_factor=time_scale_factor,
            speed_alpha=speed_alpha,
            reduce_latency=reduce_latency,
            pause_between_brackets=pause_between_brackets,
            phonemize_between_brackets=phonemize_between_brackets,
            base_url=base_url,
        )


class CodaTTS(_TTSBase):
    """Rime Coda adapter for the v1 streaming WebSocket protocol."""

    def __init__(
        self,
        *,
        websocket_url: str,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: TTSLangs | str = "eng",
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: int = 22050,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        _websocket_v1.validate_websocket_url(websocket_url)
        super().__init__(
            base_url=websocket_url,
            model="coda",
            speaker=speaker,
            lang=lang,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            time_scale_factor=time_scale_factor,
            sample_rate=sample_rate,
            api_key=api_key,
            http_session=http_session,
            use_websocket=True,
            websocket_protocol="v1",
        )

    @property
    def websocket_url(self) -> str:
        return self._base_url

    def _ws_url(self) -> str:
        return self._base_url

    def update_options(
        self,
        *,
        websocket_url: NotGivenOr[str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        lang: NotGivenOr[TTSLangs | str] = NOT_GIVEN,
        repetition_penalty: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        time_scale_factor: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(websocket_url):
            _websocket_v1.validate_websocket_url(websocket_url)
        self._update_options(
            speaker=speaker,
            lang=lang,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            sample_rate=sample_rate,
            time_scale_factor=time_scale_factor,
            base_url=websocket_url,
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(self, tts: _TTSBase, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: _TTSBase = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict[str, object] = {
            "speaker": self._opts.speaker,
            "text": self._input_text,
            "modelId": self._opts.model,
            **_model_params(self._opts),
        }
        format = "audio/pcm"
        if self._opts.model == "coda" and self._opts.coda_options is not None:
            if is_given(self._opts.coda_options.sample_rate):
                payload["samplingRate"] = self._opts.coda_options.sample_rate
        elif _is_mist_model(self._opts.model) and self._opts.mist_options is not None:
            mist_opts = self._opts.mist_options
            if is_given(mist_opts.sample_rate):
                payload["samplingRate"] = mist_opts.sample_rate
            if self._opts.model == "mistv2" and is_given(mist_opts.reduce_latency):
                payload["reduceLatency"] = mist_opts.reduce_latency

        try:
            async with self._tts._ensure_session().post(
                self._tts._base_url,
                headers={
                    "accept": format,
                    "Authorization": f"Bearer {self._tts._api_key}",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self._tts._total_timeout, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio"):
                    content = await resp.text()
                    logger.error("Rime returned non-audio data", extra={"lk.pii.data": content})
                    return

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=format,
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """One stream = one utterance. Server-side bySentence segmentation by default;
    pass segment="immediate" on the TTS to disable server buffering when the agent
    is already feeding sentence-tokenized text."""

    def __init__(self, *, tts: _TTSBase, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: _TTSBase = tts
        self._opts = copy.deepcopy(tts._opts)
        self._pool = tts._pool
        self._end_flush_sentinel: object | None = None

    def _enqueue_flush_sentinel(self) -> tts.SynthesizeStream._FlushSentinel:
        sentinel = self._FlushSentinel()
        self._input_ch.send_nowait(sentinel)
        self._input_buffer.append(sentinel)
        return sentinel

    def flush(self) -> None:
        if self._tts._websocket_protocol != "v1":
            super().flush()
            return
        if self._input_ch.closed:
            return

        # A Rime v1 flush speaks pending text but keeps the same synthesis context open.
        # Keep _mtc_text intact so later text remains part of this one LiveKit segment.
        self._enqueue_flush_sentinel()

    def end_input(self) -> None:
        if self._input_ch.closed:
            return
        if self._tts._websocket_protocol != "v1":
            super().end_input()
            return

        if self._mtc_text:
            self._mtc_pending_texts.append(self._mtc_text)
            self._mtc_text = ""

        self._end_flush_sentinel = self._enqueue_flush_sentinel()
        self._input_ch.close()
        self._input_ended = True

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if self._tts._websocket_protocol == "v1":
            await self._run_v1(output_emitter)
            return
        await self._run_ws3(output_emitter)

    async def _run_v1(self, output_emitter: tts.AudioEmitter) -> None:
        coda = self._opts.coda_options
        if coda is None:
            raise APIError("Rime v1 requires Coda options", retryable=False)
        if not is_given(coda.lang) or not is_given(coda.sample_rate):
            raise APIError("Rime v1 requires language and sample_rate", retryable=False)

        def _optional(value: NotGivenOr[_OptionT]) -> _OptionT | None:
            return value if is_given(value) else None

        options = _websocket_v1.SynthesisOptions(
            speaker=self._opts.speaker,
            language=str(coda.lang),
            sampling_rate=coda.sample_rate,
            repetition_penalty=_optional(coda.repetition_penalty),
            temperature=_optional(coda.temperature),
            top_p=_optional(coda.top_p),
            max_tokens=_optional(coda.max_tokens),
            time_scale_factor=_optional(coda.time_scale_factor),
        )

        async def _input_events() -> AsyncIterable[str | _websocket_v1.Flush]:
            async for event in self._input_ch:
                if isinstance(event, self._FlushSentinel):
                    if event is not self._end_flush_sentinel:
                        yield _websocket_v1.Flush()
                else:
                    yield event

        ws = await self._pool.get(timeout=self._conn_options.timeout)
        self._acquire_time = self._pool.last_acquire_time
        self._connection_reused = self._pool.last_connection_reused
        reusable = False
        try:
            result = await _websocket_v1.run_context(
                ws,
                context_id=utils.shortuuid(),
                options=options,
                input_events=_input_events(),
                output_emitter=output_emitter,
                timeout=self._conn_options.timeout,
                mark_started=self._mark_started,
            )
            reusable = result.reusable
        except _websocket_v1._ContextCancelled as e:
            reusable = e.reusable
            raise
        finally:
            if reusable:
                self._pool.put(ws)
            else:
                self._pool.remove(ws)
                await self._tts._close_ws(ws)

    async def _run_ws3(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        context_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=context_id)

        sent_stream = self._tts._sentence_tokenizer.stream()
        input_sent_event = asyncio.Event()
        empty_input = False

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal empty_input
            sent_count = 0
            async for ev in sent_stream:
                pkt = {"text": ev.token + " ", "contextId": context_id}
                self._mark_started()
                await ws.send_str(json.dumps(pkt))
                input_sent_event.set()
                sent_count += 1
            if sent_count == 0:
                empty_input = True
                input_sent_event.set()
                output_emitter.end_input()
                return
            await ws.send_str(json.dumps({"operation": "flush", "contextId": context_id}))

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent_event.wait()
            if empty_input:
                return
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Rime ws closed unexpectedly",
                        request_id=request_id,
                    )
                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"Rime ws error: {ws.exception()}")
                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Rime ws message type %s", msg.type)
                    continue
                data = json.loads(msg.data)
                t = data.get("type")
                if t == "chunk":
                    output_emitter.push(base64.b64decode(data["data"]))
                elif t == "timestamps":
                    wt = data.get("word_timestamps") or {}
                    words = wt.get("words") or []
                    starts = wt.get("start") or []
                    ends = wt.get("end") or []
                    for w, s, e in zip(words, starts, ends, strict=False):
                        output_emitter.push_timed_transcript(
                            TimedString(text=w + " ", start_time=s, end_time=e)
                        )
                elif t == "done":
                    output_emitter.end_input()
                    break
                elif t == "error":
                    msg_text = data.get("message", "(no message)")
                    raise APIError(f"Rime ws error: {msg_text}")

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    input_sent_event.set()
                    await sent_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(f"Rime WS error: {e}") from e
