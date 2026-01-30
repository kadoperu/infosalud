"""
VAD con Silero (snakers4/silero-vad).
Carga del modelo, int2float, procesamiento por chunk y pipeline mic → VAD → STT.
"""
import asyncio
from typing import Any

import numpy as np
import torch

from ..config import MIC_SAMPLE_RATE, SILERO_WINDOW_SAMPLES, SileroVADConfig


def load_silero_vad(config: SileroVADConfig | None = None) -> Any:
    """
    Carga el modelo Silero VAD y devuelve un VADIterator para streaming
    (chunks de 512 muestras @ 16 kHz).
    """
    if config is None:
        config = SileroVADConfig()
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (_, _, _, VADIterator, _) = utils
    vad_iterator = VADIterator(
        model,
        threshold=config.threshold,
        sampling_rate=MIC_SAMPLE_RATE,
        min_silence_duration_ms=config.min_silence_duration_ms,
        speech_pad_ms=config.speech_pad_ms,
    )
    return vad_iterator


def int2float(audio_int16: np.ndarray) -> np.ndarray:
    """Convierte int16 a float32 [-1, 1] para Silero."""
    audio = audio_int16.astype(np.float32)
    abs_max = np.abs(audio).max()
    if abs_max > 0:
        audio /= 32768.0
    return audio


def process_vad_chunk(vad_iterator: Any, chunk: bytes) -> dict | None:
    """
    Ejecuta Silero VAD sobre un chunk de 512 muestras (sync).
    Devuelve None, {'start': ...} o {'end': ...}.
    """
    arr = np.frombuffer(chunk, dtype=np.int16)
    if len(arr) != SILERO_WINDOW_SAMPLES:
        return None
    audio_float = int2float(arr)
    tensor = torch.from_numpy(audio_float).unsqueeze(0)
    with torch.no_grad():
        return vad_iterator(tensor, return_seconds=False)


async def vad_and_send(
    state,
    mic_queue: asyncio.Queue,
    utterance_queue: asyncio.Queue,
    dg_conn,
    vad_iterator: Any,
) -> None:
    """
    Pipeline: mic → Silero VAD → Deepgram.
    - Silero start → barge-in si TTS está sonando.
    - Silero end → enviar state.last_final_transcript a utterance_queue.
    - Siempre reenvía el chunk a Deepgram.
    """
    print("[VAD] Silero + STT sender iniciado")
    while state.running:
        chunk = await mic_queue.get()
        if chunk is None:
            break
        try:
            vad_result = await state.loop.run_in_executor(
                None, process_vad_chunk, vad_iterator, chunk
            )
        except Exception as e:
            print(f"[VAD] Error Silero: {e}")
            vad_result = None
        if vad_result is not None:
            if "start" in vad_result:
                print("[VAD][Silero] Speech started")
                if state.tts_playing:
                    print("[VAD][VAD-Barge-in] Usuario habló (Silero) → pedir corte TTS")
                    state.barge_in_event.set()
            if "end" in vad_result:
                print("[VAD][Silero] Speech ended (fin de enunciado)")
                text = (state.last_final_transcript or "").strip()
                if text:
                    utterance_queue.put_nowait(text)
                state.last_final_transcript = ""
        dg_conn.send(chunk)
