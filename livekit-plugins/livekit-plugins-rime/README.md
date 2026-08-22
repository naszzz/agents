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
is complete. It is opt-in. The existing `ws3` protocol remains the default.

```python
import os

from livekit.plugins import rime

tts = rime.TTS(
    model="coda",
    speaker="astra",
    api_key=os.environ["RIME_API_KEY"],
    use_websocket=True,
    websocket_protocol="v1",
    base_url="https://api.rimetts.com/coda/v1/coda",
)
```

The example route is a staging route. Pass the active Coda model route explicitly. The client
changes `https` to `wss` and adds `/ws`.

The first v1 implementation has these limits:

- It supports the `coda` model only.
- It requests raw `audio/pcm` data.
- It does not provide aligned word timestamps.
- It does not support `speed_alpha`. Use `time_scale_factor` for supported speed changes.
- It uses the JSON `rime.v1.json` WebSocket subprotocol.
