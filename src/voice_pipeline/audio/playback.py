"""
Reproducción de audio: PCM crudo a ffplay por stdin.
Usado por el TTS worker para salida en tiempo real.
"""
import asyncio
import subprocess
from typing import AsyncIterator

from ..config import TTS_SAMPLE_RATE


def create_ffplay_process():
    """
    Crea un proceso ffplay que lee PCM s16le mono desde stdin.
    Devuelve el Popen; el caller debe escribir chunks y luego cerrar stdin.
    """
    try:
        proc = subprocess.Popen(
            [
                "ffplay",
                "-f", "s16le",
                "-ar", str(TTS_SAMPLE_RATE),
                "-autoexit",
                "-nodisp",
                "-",
            ],
            stdin=subprocess.PIPE,
        )
        return proc
    except FileNotFoundError:
        raise RuntimeError("ffplay no encontrado en PATH. Instala ffmpeg.")


async def play_pcm_stream(
    state,
    stream: AsyncIterator[bytes],
    *,
    on_barge_in=None,
) -> None:
    """
    Reproduce un stream de chunks PCM escribiéndolos al stdin de ffplay.
    Si state.barge_in_event se activa, se detiene la reproducción.
    on_barge_in es opcional (ej. mensaje de log).
    """
    proc = create_ffplay_process()
    try:
        async for chunk in stream:
            if state.barge_in_event.is_set():
                if on_barge_in:
                    on_barge_in()
                break
            if proc.stdin:
                try:
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
                except BrokenPipeError:
                    break
            await asyncio.sleep(0)
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
