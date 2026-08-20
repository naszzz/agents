# Rime plugin for LiveKit Agents

Support for voice synthesis with the [Rime](https://rime.ai/) API.

See [https://docs.livekit.io/agents/integrations/tts/rime/](https://docs.livekit.io/agents/integrations/tts/rime/) for more information.

## Installation

```bash
pip install livekit-plugins-rime
```

## Pre-requisites

You'll need an API key from Rime. It can be set as an environment variable: `RIME_API_KEY`

## Streaming WebSocket interfaces

The plugin keeps the public Rime `/ws3` interface as its default streaming path. Set
`websocket_api="rime.v1"` to use the Rime engine streaming interface during its preview:

```python
from livekit.plugins import rime

tts = rime.TTS(
    model="coda",
    speaker="lyra",
    use_websocket=True,
    websocket_api="rime.v1",
    base_url="https://your-rime-engine.example.com",
    text_lookahead_tokens=4,
)
```

The preview interface requires an explicit `base_url` and supports Coda only. It uses
`rime.v1.json` at `GET /ws`, sends the API key with the `Api-Key` authorization scheme, and
defaults to 24 kHz PCM when `sample_rate` is not set. It does not provide aligned transcripts or
support `speed_alpha`.
