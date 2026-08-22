# Rime plugin for LiveKit Agents

Support for voice synthesis with the [Rime](https://rime.ai/) API.

See [https://docs.livekit.io/agents/integrations/tts/rime/](https://docs.livekit.io/agents/integrations/tts/rime/) for more information.

## Installation

```bash
pip install livekit-plugins-rime
```

## Pre-requisites

You'll need an API key from Rime. It can be set as an environment variable: `RIME_API_KEY`

## Streaming Coda WebSocket API

The Rime v1 WebSocket protocol accepts text fragments and returns audio before the input turn
is complete. Use the Coda adapter with the final WebSocket endpoint.

```python
import os

from livekit.plugins import rime

tts = rime.CodaTTS(
    websocket_url="wss://api.rimetts.com/coda/v1/coda/ws",
    speaker="astra",
    api_key=os.environ["RIME_API_KEY"],
)
```

Pass the active Coda WebSocket endpoint explicitly. `CodaTTS` always uses Coda, WebSocket
streaming, and the v1 JSON protocol. The older `TTS` v1 constructor arguments remain available
for compatibility.

The first v1 implementation has these limits:

- It supports the `coda` model only.
- It requests raw `audio/pcm` data.
- It does not provide aligned word timestamps.
- It does not support `speed_alpha`. Use `time_scale_factor` for supported speed changes.
- It uses the JSON `rime.v1.json` WebSocket subprotocol.
