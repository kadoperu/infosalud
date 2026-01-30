"""
Entrada de audio: micrófono vía sounddevice.
RawInputStream, list_audio_devices, create_input_stream, mic_sender.
"""
import asyncio

import sounddevice as sd

from ..config import MIC_CHUNK_FRAMES, MIC_SAMPLE_RATE


def list_audio_devices() -> None:
    """Imprime los dispositivos de audio disponibles."""
    print("=== Dispositivos de audio disponibles ===")
    print(sd.query_devices())


def create_input_stream(state, mic_queue, device: int | None):
    """
    Crea un RawInputStream que pone cada chunk de audio en mic_queue.
    state debe tener .loop (asyncio loop) para put_nowait thread-safe.
    """
    def callback(indata, frames, time, status):
        if status:
            print("[MIC] Status:", status)
        state.loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(indata))

    return sd.RawInputStream(
        samplerate=MIC_SAMPLE_RATE,
        blocksize=MIC_CHUNK_FRAMES,
        dtype="int16",
        channels=1,
        callback=callback,
        device=device,
    )


async def mic_sender(state, mic_queue: asyncio.Queue, dg_conn) -> None:
    """
    Reenvía chunks del micrófono a Deepgram sin pasar por VAD.
    Usado cuando --pipeline_vad 0 (VAD desactivado).
    """
    print("[STT] Sender iniciado (sin VAD)")
    while state.running:
        chunk = await mic_queue.get()
        if chunk is None:
            break
        dg_conn.send(chunk)
