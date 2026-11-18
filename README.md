# ZONOS2

<p align="center">
  <img src="./assets/ZONOS2BlogThumbnail.png" alt="ZONOS2 title card" width="750" />
</p>

<div align="center">
  <a href="https://discord.gg/gTW9JwST8q" target="_blank">
    <img src="https://img.shields.io/badge/Join%20Our%20Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

---


ZONOS2 is our latest text-to-speech model trained on more than 6 million hours of varied multilingual speech, delivering expressiveness and quality on par with—or even surpassing—top TTS providers at low latency with MoE. ZONOS2 excels at high-fidelity and naturalistic voice cloning.


During inference we use nemo TN normalized UTF-8 bytes and an ECAPA-TDNN embedding to generate DAC tokens with our MoE backbone. An inference overview can be seen below.
<p align="center">
  <img src="./assets/zonos2_arlooop_animated.gif" alt="ZONOS2 title card" width="750" />
</p>

Language support is as follows.
| Tier   | Languages                                                                                                                                                                                      |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tier 1 | English, Mandarin Chinese, Japanese                                                                                                                                                            |
| Tier 2 | Korean, Russian, Italian, Portuguese, French, Spanish, Vietnamese, German, Hebrew, Dutch                                                                                                       |
| Tier 3 | Swedish, Hindi, Tamil, Telugu, Thai, Norwegian, Bengali, Tagalog, Arabic, Danish, Indonesian, Polish, Ukrainian, Romanian, Finnish, Hungarian, Lithuanian, Estonian, Slovak, Croatian, Latvian |


For high-performance local inference we provide a TTS inference server built on [Mini-SGLang](https://github.com/sgl-project/mini-sglang).

For cpu inference and cross platform support we provide a ggml implementation for ZONOS2 in this [repo](https://github.com/Zyphra/zonos2.cpp).

**For more details and speech samples, check out our [blog](https://www.zyphra.com/our-work/zonos2).**

**We also have a hosted version available at [cloud.zyphra.com/audio-playground](https://cloud.zyphra.com/audio-playground).**

---

## Quick Start

> **Platform Support**: Linux only (x86_64). Requires NVIDIA GPU with CUDA toolkit matching your driver version (`nvidia-smi` to check).

### 1. Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Zyphra/Zonos2.git
cd Zonos2
uv sync
```

### 2. Launch the TTS Server

```bash
uv run python -m zonos2 --model-path Zyphra/ZONOS2 --tts-default-voices-dir ./default_voices/
```

`uv run` always uses the project environment, so no venv activation is needed.

The server starts on `http://localhost:1919` by default. TTS mode is auto-detected for zonos2 models.
`--tts-default-voices-dir <folder>` pre-populates the web UI with voice-clone
speakers from disk; the folder is scanned recursively for speaker audio
(`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`, `.aac`, `.webm`) and saved
embeddings (`.npy`, `.npz`). The newest voice is selected automatically on
startup.

### 3. Generate Speech

**curl:**

```bash
curl -X POST http://localhost:1919/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "stream": true}' \
  --output output.pcm

# Convert to WAV
ffmpeg -f f32le -ar 44100 -ac 1 -i output.pcm output.wav
```

**Web UI:** Open `http://localhost:1919/` in your browser.

## Python API (offline inference)

You can also run the engine directly in a Python script, without starting a
server, via `TTSLLM`. The offline path is at parity with the server: it applies
the same text normalization, supports voice cloning, and exposes the same
conditioning controls.

```python
from zonos2.message import TTSSamplingParams
from zonos2.tts import TTSLLM

tts = TTSLLM(model_path="Zyphra/ZONOS2")

results = tts.generate(
    ["Hello from the offline Python API.", "Batched prompts work too."],
    TTSSamplingParams(seed=42),
)

for i, result in enumerate(results):
    print(f"frames={len(result['audio_tokens'])}, eos_frame={result['eos_frame']}")
    tts.save_audio(result["audio"], f"output_{i}.wav")
```

### Voice cloning

Compute a speaker embedding from a reference audio file (decoded with the same
ffmpeg path the server uses) and pass it to `generate()`:

```python
emb = tts.embed_speaker_file("default_voices/AmericanFemale.mp3")

result = tts.generate_one(
    "This is spoken in the cloned voice.",
    TTSSamplingParams(seed=42),
    speaker_embedding=emb,         # also: clean_speaker_background=..., accurate_mode=...
)
tts.save_audio(result["audio"], "cloned.wav")
```

### Conditioning controls (server parity)

`generate()` / `generate_one()` accept the same high-level controls as the
server and resolve them with the server's exact bucketing logic:

```python
tts.generate_one(
    "Numbers like 123 are normalized to words.",
    TTSSamplingParams(),
    language="en_us",              # text normalization is on by default
    speed=1.1,                     # or speaking_rate=<bytes/sec>, or speaking_rate_bucket=<int>
    quality_values={"trailing_silence_s": 0.4},   # or quality_buckets=...
    max_tokens=600,                # clamped to the model limit
)
```

Set `text_normalization=False` to feed text through verbatim. The standalone
resolvers (`resolve_speaking_rate_bucket`, `resolve_quality_buckets`,
`resolve_max_tokens`) are also available if you want to precompute buckets.

For long passages, use `generate_long_one(text, TTSSamplingParams(prefix_cfg_scale=1.2), speaker_embedding=emb)`; it keeps whole prior chunks as acoustic context, optionally pins the first chunk, and applies the same speaker conditioning to every chunk. `chunk_chars`, `window_chunks`, `pin_anchor`, and `split_mode` control chunking and context.


## API Reference

### `POST /tts/generate`

Full-featured TTS endpoint with streaming support.

Long text is automatically split into overlapping chunks, each synthesized as a
teacher-forced continuation of prior audio. This avoids boundary discontinuities and
voice drift over long passages; see the `long_form_*` parameters below to tune or
disable it.

**Request body:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `language` | string | `en_us` | Text-normalization language: `en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`, `pt_br`, `ja`, `cmn`, `ko` |
| `text_normalization` | bool | `true` | Verbalize numbers, dates and currency before synthesis (`false` = raw byte tokenization) |
| `temperature` | float | `1.15` | Sampling temperature |
| `topk` | int | `106` | Top-k sampling |
| `top_p` | float | `0.0` | Nucleus (top-p) sampling threshold; `0` disables |
| `min_p` | float | `0.18` | Min-p probability filter; `0` disables |
| `max_tokens` | int \| null | model max | Maximum audio tokens. Omit or set `null` to use the model context limit; long prompts are clamped to remaining context. |
| `fade_out_ms` | float | `0.0` | Cosine fade-out applied to the audio tail; `0` disables |
| `repetition_window` | int | `50` | Recent generated frames to check per codebook; `0` disables |
| `repetition_penalty` | float | `1.2` | Per-codebook repetition penalty strength; `1.0` disables |
| `repetition_codebooks` | int | `8` | Number of codebooks from CB0 upward to penalize; negative means all |
| `seed` | int \| null | `null` | Random seed for reproducibility |
| `speaking_rate_enabled` | bool | `false` | Set `true` to use model speaking-rate conditioning when another speaking-rate field is present |
| `speaking_rate_bucket` | int \| null | `null` | Exact model speaking-rate bucket to prepend before text |
| `speaking_rate` | float \| null | `null` | Target speaking rate in cleaned UTF-8 bytes per second; mapped to a bucket |
| `speed` | float \| null | `null` | OpenAI-style multiplier; `1.0` maps to the model's neutral speaking-rate bucket |
| `quality_enabled` | bool | `true` | Enable quality-bin conditioning on supported models |
| `quality_buckets` | object \| list \| null | `{"trailing_silence_s": 3}` | Per-feature quality bucket indices (keyed by feature name, or a list in feature order) |
| `quality_values` | object \| list \| null | `null` | Raw quality metric values, mapped to buckets server-side (alternative to `quality_buckets`) |
| `clean_speaker_background` | bool | `false` | Mark the reference voice as having a clean background (supported models) |
| `accurate_mode` | bool | `true` | `true` = accurate mode (closer voice match), `false` = expressive mode |
| `emotion_enabled` | bool | `false` | Enable emotion-control conditioning (requires loaded emotion directions) |
| `emotion_sliders` | object \| null | `null` | Per-emotion weights, e.g. `{"happy": 1.0, "sad": 0.5}` (available names from `/tts/capabilities`) |
| `emotion_valence` | float | `0.0` | Valence axis (−1 negative … +1 positive) |
| `emotion_arousal` | float | `0.0` | Arousal axis (−1 calm … +1 excited) |
| `emotion_strength` | float | `1.0` | Multiplier on the calibrated strength; `1.0` = calibrated, higher exaggerates |
| `emotion_cfg_scale` | float | `1.0` | Emotion guidance; `1.0` = off, ~1.5 strongly amplifies emotion (best with expressive mode), ~2× compute |
| `cfg_scale` | float | `1.0` | Speaker-embedding classifier-free guidance; `1.0` disables, `>1` pushes generation toward the target speaker |
| `prefix_cfg_scale` | float | `1.0` | Acoustic-prefix classifier-free guidance for long-form continuation chunks; `1.0` disables, `>1` sharpens continuity across chunk joins, `<1`/negative downweights the prefix. No effect on the first chunk |
| `long_form` | bool \| null | `null` | `null` = auto (engage when text exceeds the chunk size), `true`/`false` force long-form on/off |
| `long_form_chunk_chars` | int | `150` | New text generated per step, in characters |
| `long_form_window_chunks` | int | `2` | Total chunks fed per step; `window-1` prior chunks are teacher-forced context. `1` disables teacher forcing |
| `long_form_pin_anchor` | bool | `true` | Pin the whole first chunk into every continuation prefix to prevent timbre drift over long passages |
| `long_form_split_mode` | string | `"word"` | Chunking strategy: `"word"` (greedy word packing) or `"sentence"` (sentence boundaries, falling back to words for over-long sentences) |
| `stream` | bool | `true` | Stream audio chunks |

**Response:** Raw PCM audio (`audio/pcm`, float32, 44.1 kHz, mono). Headers include `X-Audio-Sample-Rate`, `X-Audio-Channels`, `X-Audio-Format`.

### Continue speech across HTTP requests

The Python server can save the audio tokens and text from a request.
The next request uses this context to continue the speech without playing the old audio again.

#### Write a chapter to a WAV file

Start the Python server with `--num-tokenizer 0` (the default).
In `chapter.txt`, put a blank line between paragraphs.
Run the [Python client](scripts/tts_chapter.py) from this repository:

```bash
python scripts/tts_chapter.py chapter.txt chapter.wav --speaker default_voices/AmericanFemale.mp3
```

The client uses streaming HTTP. Add `--buffered` to receive all the audio for a request at once.
Use `--base-url` to point it at another server.
The client checks the final status of each paragraph. If it can retry, it starts from the last usable ID.
It adds only accepted audio to the WAV. If the chapter fails, it leaves the output file unchanged.
If a paragraph takes too many frames, try shorter paragraphs or set `--retry-max-tokens`.
Use `--help` to see the other options.

#### Use the API directly

Send the same `X-TTS-Session-ID` in generation requests, status checks, and speaker uploads.
Set `retain_continuation: true` on the first request.
The server returns an ID in `X-TTS-Continuation-ID`.
Read the full response before you send the next request.
Make sure that the context is [usable](#completion-and-retry).
If the context is usable, send its ID as `continuation_of` in the next request.

By default, each linked request saves its own context and returns a new ID.
Use the new ID to continue from that response.
You can also use one usable ID in more than one request.
Each request starts from the same saved context without changing it.
If you do not need another ID, set `retain_continuation: false`.
The WAV client saves an ID even for the last paragraph so it can check the final status in streaming mode.

A chunk pairs a piece of speech with its text.
Conditioning controls how the model speaks.
Each linked request uses the same speaker, language, conditioning, and chunk configuration as the previous request.
Do not send those fields again on linked requests.
You can change sampling controls such as `seed`, `temperature`, and `prefix_cfg_scale`.

With `long_form_pin_anchor: true`, the server keeps the first chunk in context.
With the default `long_form_window_chunks: 2`, it also keeps the most recent chunk.
If `long_form_window_chunks` is `1`, enable `long_form_pin_anchor` to save context.
If you ask to save context with a one-chunk window and pinning off, the server returns HTTP 400.

If a request saves context, the server turns off the control for trailing silence.
The server trims three frames from the end of a completed chunk before it saves that chunk.
It does the same for chunks in the middle of a long-form request.

#### Play separate clips with a pause

A fixed reference is saved text and audio tokens.
It helps keep the voice consistent across clips.
Speaking pace and tone can still vary.
The server does not send the reference audio again.

To make separate clips with the same reference:

1. Generate a short passage in the voice that you want with `retain_continuation: true`.
2. Make sure that the saved reference is [usable](#completion-and-retry).
3. Use the same `X-TTS-Session-ID` that you used to save the reference.
4. Use the original reference ID as `continuation_of` for every clip.
5. Send the following body to `POST /tts/generate`:

   ```json
   {
     "text": "Please have your identity card ready.",
     "continuation_of": "<reference-id>",
     "retain_continuation": false,
     "defer_audio_tail": false,
     "stream": false
   }
   ```

6. Make sure that `X-TTS-Generation-Status` is `complete`.
7. Save the returned PCM as a WAV file.

Setting `retain_continuation: false` does not remove the original reference from the cache.

Leave out `audio_tail_of` for these clips.
With a linked tail, a response can start with the ending of the previous clip.
A pause between responses can interrupt a sentence.

Play each WAV to the end.
Add the pause in your player.
If you play the WAV files without a pause, the join can still be audible.

#### Play separate responses as one track

The decoder turns audio tokens into sound.
An audio tail is a saved ending.
The model uses `continuation_of` for its reference.
The decoder uses `audio_tail_of` to finish the previous clip with the start of the next clip.
The two IDs can point to different responses.

If you have all the text, use one `long_form: true` request.
If the text arrives in parts, use the same `X-TTS-Session-ID` for generation, status, finish, and delete requests.
Set `defer_audio_tail: true` on the first request in the track.
The server holds the ending and returns `X-TTS-Audio-Tail-ID` and `X-TTS-Audio-Status-URL`.

For each response:

1. Read all the PCM.
2. Make sure that the status at `X-TTS-Audio-Status-URL` reports `finish_reason: "complete"`.
3. If more text follows, make sure that `audio_tail_pending` is `true`.
4. Append the PCM to the track.

On the next request, set `audio_tail_of` to the previous `X-TTS-Audio-Tail-ID`.
If you use a fixed reference, keep `continuation_of` the same for every request.
If more text follows that request, set `defer_audio_tail: true`.
If the request is the last one, set `defer_audio_tail: false`.
If you kept the last tail, call `POST /tts/audio-tails/{id}/finish`.
Append the PCM from that call to the track.

`usable` describes the model context, not the audio tail.
You can keep an audio tail with `retain_continuation: false`.
`/tts/capabilities` reports this feature as `audio_tail_continuation`.

If you switch languages, start a new model context without `continuation_of`.
Use `audio_tail_of` to keep the same track.
You can still hear a change in voice or language.

The server does not change a tail when you use it.
If an attempt fails, discard its audio before you retry with the same tail.
Do not retry audio that already played.

After you accept the next response or finish the track, delete the old tail with `DELETE /tts/audio-tails/{id}`.
If playback stops early, close the active stream.
Delete the tail without finishing it.
Deleting a tail does not remove the saved model context.

The same cache limits apply to model context and audio tails.
The server needs a cache entry for audio status even if you do not save model context.
Do not use `fade_out_ms` on these requests.
You can send the next request before the previous audio ends.
If the next audio arrives late, playback can pause.

#### Save separate WAV files without adding a pause

[Make one continuous track](#play-separate-responses-as-one-track) before you split it into files.
Do not cut at HTTP response boundaries.
The next response can start with part of an earlier sentence.
Cut at the end of each sentence instead.

If each linked request generates one sentence, find its cut before you delete its tail:

1. Count all PCM samples from the start of the track through that response.
2. Call `POST /tts/audio-tails/{id}/finish` with the tail ID from that response.
3. Make sure that `X-TTS-Generation-Status` is `complete`.
4. Add the number of samples from `/finish` to the count from step 1.

For example, the first response can send 6,144 samples.
The server can hold 13,824 more samples in the tail.
The first WAV then contains 19,968 samples, not 6,144.
`/finish` does not remove the tail.
If another sentence follows, do not append the PCM from `/finish`.
The next linked response supplies that ending.

Use the end of the track as the last cut.
Save the sections as WAV files without extra silence or fades.
Keep the sample rate and format of the original track.
If you join the samples again in order, they match the original track.
A player can still add a gap when it opens the next file.

If short clips sound stiff, try generating several sentences at a time with the fixed reference.
Text alignment matches words to times in audio.
If one request generates several sentences, use an audio editor or text alignment to find the cuts between them.
The tail gives only the end of the request.

#### Cache limits

The cache stores context between requests.
The frame and text limits count the first chunk and the recent chunks.
To change a limit, set its environment variable before you start the server:

| Limit | Default | Environment variable |
|-------|---------|----------------------|
| Entries | 32 | `ZONOS2_CONTINUATION_MAX_ENTRIES` |
| Seconds before an entry expires | 300 | `ZONOS2_CONTINUATION_TTL_SECONDS` |
| Audio frames per entry | 4096 | `ZONOS2_CONTINUATION_MAX_FRAMES` |
| Text bytes per entry | 16384 | `ZONOS2_CONTINUATION_MAX_TEXT_BYTES` |

When the cache fills, the server removes older entries that are not pending or in use.
An entry can still expire while a request uses it.
The server returns HTTP 404 for an unknown ID, an expired ID, or an ID from another session.
If you try to reuse pending or unusable context, the server returns HTTP 409.
If a full cache has no entry that the server can remove, a request that saves context returns HTTP 503.
A linked request with `retain_continuation: false` does not need a new entry unless it uses the audio track.

#### Completion and retry

With `stream: false`, check `X-TTS-Generation-Status` for `complete`, `frame_limit`, or `failed`.
If generation reaches the frame limit, the server can return partial audio with HTTP 200.
The server returns audio only for frames with all their tokens.
If no frame is complete, the response body is empty.
You cannot reuse this context.
The model can detect the end of speech before it produces the last audio tokens.
The context stays unusable if any of these tokens are missing.

If the server returns `X-TTS-Continuation-Status-URL`, read all the audio first.
Send a GET request to that URL with the same `X-TTS-Session-ID` value.
The reply gives the `finish_reason` and tells you whether the context is `usable`.
For `stream: true`, read the final status from this URL.
The PCM stream does not include the final status.
If the worker fails after streaming starts, the status URL can report `failed` even though the stream began with HTTP 200.
Use `retain_continuation: true` for a final status URL, or use buffered mode.

The server returns an HTTP error if the worker rejects a request before sending audio.
If the worker stops replying, the server fails the request after 60 seconds without a reply.
Set `ZONOS2_TTS_WORKER_TIMEOUT_SECONDS` before startup to change this limit.
Before streaming starts, a timeout returns HTTP 504. After streaming starts, check the status URL.
If a client disconnects, the shared worker drops its audio buffers. The scheduler stops at a safe point.

If generation reaches the frame limit, retry from the last usable ID that is still in the cache.
Use shorter text or increase `max_tokens` within the model limit.
If the context exceeds a cache limit, use smaller chunks. The context can be unusable even when generation completes.
In a single long-form request, the server stops at the first incomplete chunk.
It does not use the text from an unfinished chunk to generate more audio.

#### Deploy one or more Python servers

- **One server, one GPU:** Use `--num-tokenizer 0`.
- **Two replicas:** Run a separate server on each GPU. Use a router that sends the same `X-TTS-Session-ID` to the same replica. Route generation, status checks, and speaker uploads this way. Make sure the router passes audio through as it arrives and does not send a failed request to another replica. On one host, use different API ports and leave each API port plus one free for its worker.
- **One server, two GPUs:** Use `--tp-size 2` and `--num-tokenizer 0`. Both GPUs work on the same requests. There is still one API and one cache. Add `--disable-pynccl` to use PyTorch's NCCL if the system does not have the optional PyNCCL build libraries.

The [CUDA integration test](tests/misc/test_tts_continuation_gpu.py) checks these setups with a real checkpoint.
Each chain still runs in order. Requests from different chains can run together.

A continuation ID or uploaded speaker ID belongs to one API process.
A restart clears its IDs. If a session goes to another replica, that replica cannot use those IDs.
Start a new chain after a restart or a move. The server cannot recover the old chain.
The session header does not prove who the caller is. Use a trusted network or add access control in front of the server.

Without an audio track, the server decodes each response separately.
Where two responses meet, this can cause an audible click.

A cold start is the first render after startup.
Fixed seeds control which tokens the model selects.
They do not guarantee identical PCM after a cold start or across CUDA backends, hardware, and library versions.
Before deployment, listen to audio from the target GPU for voice changes and audible joins.

### `POST /v1/audio/speech`

OpenAI-compatible endpoint.

**Request body:**

```json
{
  "model": "zonos2",
  "input": "Hello world",
  "voice": "alloy",
  "response_format": "pcm"
}
```

For speaking-rate-enabled checkpoints, set `speaking_rate_enabled` to `true`
and use `speaking_rate_bucket` for exact bucket control, `speaking_rate` for
bytes-per-second control, or `speed` for OpenAI-style multiplier control.

## Emotion Control

You can nudge a voice toward an emotion (happy, sad, angry, surprised) or along
the valence/arousal axes **without changing speaker identity**. Emotion is
applied as additive *direction* vectors on the speaker conditioning — no model
or checkpoint changes — so the timbre is preserved while prosody shifts.

Direction vectors ship in `./emotion_directions/`, and the server **auto-loads
them on startup when that folder is present** — emotion sliders appear in the
web UI automatically, no extra flags. Point elsewhere (or disable) with
`--tts-emotion-directions-dir <dir>` (pass an empty string to turn it off).

```bash
curl -X POST http://localhost:1919/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "I cannot believe you did that!",
        "emotion_enabled": true,
        "emotion_sliders": {"happy": 1.0},
        "accurate_mode": false,
        "emotion_cfg_scale": 1.5,
        "stream": true
      }' \
  --output happy.pcm
```

`GET /tts/capabilities` reports what the loaded directions expose:
`emotion_enabled`, `emotion_names`, `emotion_axes`, and `emotion_calibrated`
(whether a per-speaker strength calibration is loaded). The shipped directions
include a `calibration.json` so `emotion_strength: 1.0` is already a sensible
per-emotion default; raise it to exaggerate. For strong, reliable effects use
**expressive mode** (`accurate_mode: false`) together with
`emotion_cfg_scale` around `1.5`.

**Building your own directions.** Use `scripts/build_emotion_directions.py` to
encode an emotion-labelled corpus (e.g. ESD) into a new `emotion_directions/`
set, and `scripts/calibrate_emotion_strength.py` to auto-tune per-speaker,
per-emotion strength against the emotion2vec recognizer. See each script's
`--help`.

## Citation
If you find this model useful in an academic context please cite as:
```
@misc{zyphra2025zonos,
  title     = {Zonos V2 Technical Report},
  author    = {Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge},
  year      = {2026},
}
```

## License

ZONOS2 is released under the [MIT License](LICENSE).

It incorporates third-party components under their own licenses — see
[`NOTICE`](NOTICE) and [`licenses/`](licenses/):

- The TTS inference server and runtime are derived from
  [Mini-SGLang](https://github.com/sgl-project/mini-sglang) (MIT).
- `python/zonos2/vendor/nemo_text_processing/` is vendored from
  [NVIDIA NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)
  (Apache-2.0).
