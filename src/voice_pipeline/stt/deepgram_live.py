"""
STT: Deepgram live streaming.
Con pipeline_vad=1: solo transcripción (VAD = Silero).
Con pipeline_vad=0: transcripción y envío a utterance_queue en is_final (sin Silero).
"""
import asyncio
from typing import Optional

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

from ..config import MIC_SAMPLE_RATE


def setup_deepgram(state, utterance_queue: Optional[asyncio.Queue] = None):
    """
    Configura la conexión WebSocket a Deepgram para transcripción en vivo.
    - Si utterance_queue es None (VAD Silero activo): solo actualiza state.last_final_transcript en is_final.
    - Si utterance_queue se pasa (VAD desactivado): en is_final también pone el texto en utterance_queue.
    Devuelve la conexión (conn); el caller envía chunks con conn.send(chunk).
    """
    dg = DeepgramClient()
    conn = dg.listen.websocket.v("1")

    def handle_transcript(result):
        if not hasattr(result, "channel"):
            return
        if not result.channel.alternatives:
            return

        alt = result.channel.alternatives[0]
        text = (alt.transcript or "").strip()
        if not text:
            return

        level = getattr(alt, "confidence", None)
        if not result.is_final:
            if level is not None:
                print(f"[VAD][VAD-TAKLING] level={level:.2f}")
            else:
                print("[VAD][VAD-TAKLING] level=?")
            print(f"\t[STT][STT-Interim] {text}")
        else:
            print(f"\t[STT][STT-Final] {text}")

        if state.tts_playing and not result.is_final and len(text.split()) >= 2:
            print("[VAD][VAD-Barge-in] Usuario habló (texto interim) → pedir corte TTS")
            state.barge_in_event.set()

        if result.is_final:
            state.last_final_transcript = text
            if utterance_queue is not None:
                utterance_queue.put_nowait(text)

    def transcript_handler(*args, **kwargs):
        result = kwargs.get("result")
        if result is None:
            if len(args) >= 2:
                result = args[1]
            elif len(args) == 1:
                result = args[0]
        if result is None:
            return
        state.loop.call_soon_threadsafe(handle_transcript, result)

    conn.on(LiveTranscriptionEvents.Transcript, transcript_handler)

    def on_error(error, **kwargs):
        print(f"[STT] Error en WebSocket de Deepgram: {error}")

    def error_handler(*args, **kwargs):
        err = kwargs.get("error")
        if err is None and args:
            err = args[-1]
        state.loop.call_soon_threadsafe(on_error, err)

    conn.on(LiveTranscriptionEvents.Error, error_handler)

    def on_close(close, **kwargs):
        print(f"[STT] WebSocket cerrado: {close}")

    def close_handler(*args, **kwargs):
        cl = kwargs.get("close")
        if cl is None and args:
            cl = args[-1]
        state.loop.call_soon_threadsafe(on_close, cl)

    conn.on(LiveTranscriptionEvents.Close, close_handler)

    conn.start(
        LiveOptions(
            model="nova-3",
            language="es",
            encoding="linear16",
            sample_rate=MIC_SAMPLE_RATE,
            channels=1,
            interim_results=True,
            vad_events=False,
        )
    )

    print("[STT] Conectado a Deepgram")
    return conn
