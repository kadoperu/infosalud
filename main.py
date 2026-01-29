#!/usr/bin/env python3
"""
Asistente de voz en consola con:
- Deepgram STT (streaming + VAD)
- OpenAI LLM
- ElevenLabs TTS (streaming real vía ffplay)
- Barge-in robusto (VAD + texto intermedio)

Notas importantes:
- El micrófono se captura con sounddevice a 16 kHz (Deepgram).
- El TTS se reproduce con ffplay, que decodifica el mp3 de ElevenLabs.
- Necesitas tener ffmpeg/ffplay instalado en el sistema.
"""

import argparse
import asyncio
import os
import signal
import subprocess
from typing import List, Optional

import httpx
import numpy as np
import sounddevice as sd
from openai import OpenAI
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from dotenv import load_dotenv

# =========================================================
# 🔧 CONFIGURACIÓN GLOBAL
# =========================================================

load_dotenv()

# 🔊 SAMPLE RATES
MIC_SAMPLE_RATE = 16000     # Deepgram STT requiere 16 kHz
CHANNELS = 1
CHUNK_MS = 40               # Tamaño de bloque (~latencia)
MIC_CHUNK_FRAMES = int(MIC_SAMPLE_RATE * CHUNK_MS / 1000)

SYSTEM_PROMPT = (
    "Eres un asistente de voz breve, claro y conversacional. "
    "Responde en español, frases cortas, sin emojis."
)

# =========================================================
# 🔐 CREDENCIALES
# =========================================================

ELEVEN_API_KEY = os.environ.get("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID")

if not ELEVEN_API_KEY:
    raise RuntimeError("ELEVEN_API_KEY no definido")

if not ELEVEN_VOICE_ID:
    raise RuntimeError("ELEVEN_VOICE_ID no definido")


# =========================================================
# 🧠 ESTADO GLOBAL COMPARTIDO
# =========================================================

class State:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.barge_in_event = asyncio.Event()
        self.tts_playing = False
        self.history: List[dict] = []
        self.running = True


# =========================================================
# 🔊 ELEVENLABS — STREAM TTS (MP3) → ffplay
# =========================================================

async def stream_tts_elevenlabs(text: str):
    """
    Genera audio en streaming desde ElevenLabs (mp3).
    No lo decodificamos aquí: se lo pasamos tal cual a ffplay.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",  # dejamos claro que queremos mp3
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "output_format": "mp3_44100_128",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


# =========================================================
# 🧠 LLM WORKER (OpenAI)
# =========================================================

async def llm_worker(
    state: State,
    openai_client: OpenAI,
    utterance_queue: asyncio.Queue,
    tts_text_queue: asyncio.Queue,
):
    print("[LLM] Worker iniciado")

    while state.running:
        text = await utterance_queue.get()
        if text is None:
            break

        print(f"[LLM] Usuario: {text}")

        def call_llm():
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(state.history[-6:])
            messages.append({"role": "user", "content": text})

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return response.choices[0].message.content.strip()

        response_text = await state.loop.run_in_executor(None, call_llm)
        print(f"[LLM] Respuesta: {response_text}")

        state.history.append({"role": "user", "content": text})
        state.history.append({"role": "assistant", "content": response_text})

        await tts_text_queue.put(response_text)


# =========================================================
# 🔊 TTS WORKER (ffplay, sin chisquidos)
# =========================================================

async def tts_worker(state: State, tts_text_queue: asyncio.Queue):
    """
    Toma texto, lo manda a ElevenLabs en streaming, y los bytes
    los envía a ffplay, que se encarga de decodificar y reproducir.
    Con barge-in: si VAD detecta voz o STT capta texto, se corta ffplay.
    """
    print("[TTS] Worker iniciado")

    while state.running:
        text = await tts_text_queue.get()
        if text is None:
            break

        print("[TTS] Sintetizando con ElevenLabs…")

        # Marcamos que TTS está sonando
        state.barge_in_event.clear()
        state.tts_playing = True

        # Lanzamos ffplay para reproducir desde stdin
        proc = subprocess.Popen(
            [
                "ffplay",
                "-autoexit",
                "-nodisp",
                "-loglevel",
                "quiet",
                "-",  # stdin
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            async for chunk in stream_tts_elevenlabs(text):
                # Si el usuario habla → barge-in
                if state.barge_in_event.is_set():
                    print("[CTRL] Barge-in: deteniendo TTS (ffplay)")
                    break

                if proc.stdin:
                    proc.stdin.write(chunk)
                    proc.stdin.flush()

                await asyncio.sleep(0)  # ceder al loop

            # Cerramos stdin para que ffplay termine (si no hubo barge-in)
            if proc.stdin and not state.barge_in_event.is_set():
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            # Si hubo barge-in y el proceso sigue vivo → lo matamos
            if state.barge_in_event.is_set() and proc.poll() is None:
                proc.terminate()

        finally:
            state.tts_playing = False
            state.barge_in_event.clear()
            try:
                proc.wait(timeout=1)
            except Exception:
                pass


# =========================================================
# 🎤 MICRÓFONO → DEEPGRAM
# =========================================================

def create_input_stream(state: State, mic_queue: asyncio.Queue, device: Optional[int]):

    def callback(indata, frames, time, status):
        if status:
            print("[MIC] Status:", status)
        # Mandamos los bytes crudos al mic_queue
        state.loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(indata))

    return sd.RawInputStream(
        samplerate=MIC_SAMPLE_RATE,
        blocksize=MIC_CHUNK_FRAMES,
        dtype="int16",
        channels=CHANNELS,
        callback=callback,
        device=device,
    )


async def mic_sender(state: State, mic_queue: asyncio.Queue, dg_conn):
    print("[STT] Sender iniciado")

    while state.running:
        chunk = await mic_queue.get()
        if chunk is None:
            break
        dg_conn.send(chunk)


# =========================================================
# 🧠 DEEPGRAM STT + VAD (con logs)
# =========================================================

def setup_deepgram(state: State, utterance_queue: asyncio.Queue):

    dg = DeepgramClient()
    conn = dg.listen.websocket.v("1")

    # ---- Manejo de TRANSCRIPCIONES ----
    def handle_transcript(result):
        """
        Maneja eventos de transcripción de Deepgram.
        """
        if not hasattr(result, "channel") or not result.channel.alternatives:
            return

        alt = result.channel.alternatives[0]
        transcript = (alt.transcript or "").strip()
        if not transcript:
            return

        is_final = getattr(result, "is_final", False)

        if is_final:
            print(f"[STT] Final: {transcript}")
            utterance_queue.put_nowait(transcript)
        else:
            print(f"[STT] Interim: {transcript}")

        # Barge-in basado en texto intermedio "suficiente"
        if state.tts_playing and not is_final:
            if len(transcript.split()) >= 2:
                print("[CTRL] Barge-in por STT interim")
                state.barge_in_event.set()

    def transcript_handler(*args, **kwargs):
        """
        Wrapper para adaptarse a las distintas formas en que el SDK
        pasa el objeto de resultado. Solo pasa 'result' como posicional
        a handle_transcript (sin kwargs, para no romper asyncio).
        """
        result = None
        if "result" in kwargs:
            result = kwargs["result"]
        elif len(args) >= 2:
            result = args[1]
        elif len(args) == 1:
            result = args[0]

        if result is None:
            return

        state.loop.call_soon_threadsafe(handle_transcript, result)

    # ---- Manejo de VAD ----
    def handle_speech_started():
        print("[VAD] SpeechStarted")
        # Barge-in inmediato si TTS está sonando
        if state.tts_playing:
            print("[CTRL] Barge-in por VAD")
            state.barge_in_event.set()

    def handle_utterance_end():
        print("[VAD] UtteranceEnd")

    # Registro de eventos
    conn.on(
        LiveTranscriptionEvents.Transcript,
        transcript_handler,
    )

    conn.on(
        LiveTranscriptionEvents.SpeechStarted,
        lambda *a, **k: state.loop.call_soon_threadsafe(handle_speech_started),
    )

    conn.on(
        LiveTranscriptionEvents.UtteranceEnd,
        lambda *a, **k: state.loop.call_soon_threadsafe(handle_utterance_end),
    )

    # Configuración del streaming con VAD
    options = LiveOptions(
        model="nova-3",
        language="es",
        encoding="linear16",
        sample_rate=MIC_SAMPLE_RATE,
        channels=1,
        interim_results=True,
        vad_events=True,
        utterance_end_ms=1000,
    )

    conn.start(options)
    print("[STT] Conectado a Deepgram")
    return conn


# =========================================================
# 🚀 MAIN
# =========================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args()

    # Configurar dispositivos de audio por defecto si se pasan índices
    if args.input_device is not None or args.output_device is not None:
        sd.default.device = (args.input_device, args.output_device)

    print(f"[AUDIO] Mic: {MIC_SAMPLE_RATE} Hz")

    # Validar API keys de Deepgram y OpenAI
    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("ERROR: DEEPGRAM_API_KEY no está definido.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY no está definido.")
        return

    state = State()
    openai_client = OpenAI()

    mic_queue: asyncio.Queue = asyncio.Queue()
    utterance_queue: asyncio.Queue = asyncio.Queue()
    tts_queue: asyncio.Queue = asyncio.Queue()

    # STT Deepgram
    dg_conn = setup_deepgram(state, utterance_queue)

    # Mic
    mic_stream = create_input_stream(state, mic_queue, args.input_device)

    # Tareas
    tasks = [
        asyncio.create_task(mic_sender(state, mic_queue, dg_conn)),
        asyncio.create_task(llm_worker(state, openai_client, utterance_queue, tts_queue)),
        asyncio.create_task(tts_worker(state, tts_queue)),
    ]

    mic_stream.start()
    print("🎧 LISTO — Habla cuando quieras (Ctrl+C para salir)")

    stop_event = asyncio.Event()

    def stop():
        print("\n[CTRL] Señal de parada recibida. Cerrando…")
        stop_event.set()

    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop)
    except NotImplementedError:
        # En Windows puede no estar disponible
        pass

    await stop_event.wait()

    # Apagado limpio
    state.running = False

    try:
        mic_stream.stop()
        mic_stream.close()
    except Exception:
        pass

    try:
        dg_conn.finish()
    except Exception:
        pass

    for q in (mic_queue, utterance_queue, tts_queue):
        try:
            q.put_nowait(None)
        except Exception:
            pass

    for t in tasks:
        t.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    print("👋 Salida limpia")


if __name__ == "__main__":
    asyncio.run(main())