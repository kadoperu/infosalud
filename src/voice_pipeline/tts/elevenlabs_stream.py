"""
TTS: ElevenLabs streaming PCM (crudo a ffplay).
stream_tts_elevenlabs, tts_worker que reproduce vía playback.
"""
import asyncio
from typing import AsyncIterator

import httpx

from ..config import (
    TTS_OUTPUT_FORMAT,
    TTS_SAMPLE_RATE,
    get_eleven_api_key,
    get_eleven_voice_id,
)
from ..audio.playback import play_pcm_stream


async def stream_tts_elevenlabs(text: str) -> AsyncIterator[bytes]:
    """
    Stream de PCM 16-bit desde ElevenLabs (crudo, sin archivos).
    output_format va como query parameter.
    """
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{get_eleven_voice_id()}/stream"
        f"?output_format={TTS_OUTPUT_FORMAT}"
    )
    headers = {
        "xi-api-key": get_eleven_api_key(),
        "Content-Type": "application/json",
        "Accept": "audio/pcm",
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


async def tts_worker(state, tts_text_queue: asyncio.Queue) -> None:
    """
    Worker que recibe texto de la cola, lo convierte a audio vía ElevenLabs
    y lo reproduce con ffplay (playback). Respeta barge-in.
    """
    print("[TTS] Worker iniciado")

    while state.running:
        text = await tts_text_queue.get()
        if text is None:
            break

        state.barge_in_event.clear()
        state.tts_playing = True

        print("[TTS][TTS-Start] Enviando texto a ElevenLabs…")
        print(f"[TTS] Texto: {text}")

        try:
            stream = stream_tts_elevenlabs(text)

            def on_barge_in():
                print("[CTRL-Barge-in] Deteniendo TTS por barge-in")

            await play_pcm_stream(
                state,
                stream,
                on_barge_in=on_barge_in,
            )
        except RuntimeError as e:
            if "ffplay" in str(e):
                print("[TTS] ERROR:", e)
        except Exception as e:
            print(f"[TTS] Error durante streaming TTS: {e}")
        finally:
            state.tts_playing = False
            state.barge_in_event.clear()
            print("[TTS][TTS-End] Fin de reproducción")
