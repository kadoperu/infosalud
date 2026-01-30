"""
CLI: arranca el pipeline voz ↔ LLM.
  voz input → [VAD Silero opcional] → STT Deepgram → LLM OpenAI → TTS ElevenLabs → voz output

Uso:
  python -m voice_pipeline
  python -m voice_pipeline --list-devices
  python -m voice_pipeline --input-device 1 --output-device 2
  python -m voice_pipeline --input-device 1 --output-device 2 --pipeline_vad 1   # VAD activado (default)
  python -m voice_pipeline --input-device 1 --output-device 2 --pipeline_vad 0   # VAD desactivado
"""
import argparse
import asyncio
import signal

import sounddevice as sd

from .config import (
    MIC_SAMPLE_RATE,
    SILERO_WINDOW_SAMPLES,
    get_deepgram_api_key,
    get_openai_api_key,
)
from .state import PipelineState
from .audio.mic import create_input_stream, list_audio_devices, mic_sender
from .vad.silero import load_silero_vad, vad_and_send
from .stt.deepgram_live import setup_deepgram
from .tts.elevenlabs_stream import tts_worker
from .llm.langgraph_client import llm_worker
from openai import OpenAI


async def run_pipeline(args: argparse.Namespace) -> None:
    """Arranca el pipeline: mic → [VAD opcional] → STT → LLM → TTS → playback."""
    get_deepgram_api_key()
    get_openai_api_key()

    pipeline_vad = getattr(args, "pipeline_vad", 1)
    if pipeline_vad:
        print(f"[AUDIO] Mic: {MIC_SAMPLE_RATE} Hz (chunks {SILERO_WINDOW_SAMPLES} muestras para Silero VAD)")
    else:
        print(f"[AUDIO] Mic: {MIC_SAMPLE_RATE} Hz (pipeline VAD desactivado)")

    state = PipelineState()
    openai_client = OpenAI()

    mic_queue: asyncio.Queue = asyncio.Queue()
    utterance_queue: asyncio.Queue = asyncio.Queue()
    tts_queue: asyncio.Queue = asyncio.Queue()

    if pipeline_vad:
        print("[VAD] Cargando Silero VAD…")
        vad_iterator = load_silero_vad()
        print("[VAD] Silero VAD listo")
        dg_conn = setup_deepgram(state)
        mic_stream = create_input_stream(state, mic_queue, args.input_device)
        sender_task = asyncio.create_task(
            vad_and_send(state, mic_queue, utterance_queue, dg_conn, vad_iterator)
        )
    else:
        dg_conn = setup_deepgram(state, utterance_queue)
        mic_stream = create_input_stream(state, mic_queue, args.input_device)
        sender_task = asyncio.create_task(mic_sender(state, mic_queue, dg_conn))

    tasks = [
        sender_task,
        asyncio.create_task(llm_worker(state, openai_client, utterance_queue, tts_queue)),
        asyncio.create_task(tts_worker(state, tts_queue)),
    ]

    mic_stream.start()
    print("🎧 LISTO — Habla cuando quieras (Ctrl+C para salir)")

    stop_event = asyncio.Event()

    def on_stop():
        stop_event.set()

    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_stop)
    except NotImplementedError:
        pass

    await stop_event.wait()

    state.running = False
    mic_stream.stop()
    dg_conn.finish()

    for q in (mic_queue, utterance_queue, tts_queue):
        q.put_nowait(None)

    for t in tasks:
        t.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    print("👋 Salida limpia")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline conversacional de voz en tiempo real"
    )
    parser.add_argument("--list-devices", action="store_true", help="Listar dispositivos de audio")
    parser.add_argument("--input-device", type=int, default=None, help="ID dispositivo de entrada (mic)")
    parser.add_argument("--output-device", type=int, default=None, help="ID dispositivo de salida")
    parser.add_argument(
        "--pipeline_vad",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 = VAD Silero activado (default), 0 = VAD desactivado (mic directo a Deepgram)",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.input_device is not None or args.output_device is not None:
        sd.default.device = (args.input_device, args.output_device)

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
