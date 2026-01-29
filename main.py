#!/usr/bin/env python3
"""
Asistente de voz en consola con:
- Deepgram STT (streaming + VAD)
- OpenAI LLM
- ElevenLabs TTS (streaming real vía ffplay)
- Barge-in robusto (VAD + texto intermedio)
- Etiquetas de diagnóstico:
  [VAD-TAKLING], [VAD-TALKING/STOP], [STT-Interim], [STT-Final],
  [LLM][HUMAN], [LLM][AI], [TTS-Start], [TTS-Chunk], [TTS-End],
  [VAD][VAD-Barge-in], [CTRL-Barge-in], [VAD-SpeechStarted], [VAD-UtteranceEnd]
"""

# python main.py --list-devices
# python main.py --input-device 1 --output-device 2

import argparse
import asyncio
import os
import signal
import subprocess
from typing import List

import httpx
import sounddevice as sd
from openai import OpenAI
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from dotenv import load_dotenv

# =========================================================
# 🔧 CONFIGURACIÓN GLOBAL
# =========================================================

load_dotenv()

MIC_SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 40
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
# 🧠 ESTADO GLOBAL
# =========================================================

class State:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.barge_in_event = asyncio.Event()
        self.tts_playing = False
        self.history: List[dict] = []
        self.running = True

# =========================================================
# 🔊 UTILIDAD: LISTAR DISPOSITIVOS
# =========================================================

def list_audio_devices():
    print("=== Dispositivos de audio disponibles ===")
    print(sd.query_devices())

# =========================================================
# 🔊 ELEVENLABS TTS
# =========================================================

# pcm_44100 requiere plan Pro en ElevenLabs (403 sin Pro). Usamos 16 kHz para todos los planes.
TTS_SAMPLE_RATE = 16000
TTS_OUTPUT_FORMAT = f"pcm_{TTS_SAMPLE_RATE}"

async def stream_tts_elevenlabs(text: str):
    """
    Streaming PCM 16-bit desde ElevenLabs (crudo, sin archivos).
    output_format va como query parameter (la API lo ignora si está en el body).
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}/stream?output_format={TTS_OUTPUT_FORMAT}"
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
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




# =========================================================
# 🧠 LLM WORKER
# =========================================================

async def llm_worker(state, openai_client, utterance_queue, tts_text_queue):
    print("[LLM] Worker iniciado")
    while state.running:
        text = await utterance_queue.get()
        if text is None:
            break

        print(f"[LLM][HUMAN] {text}")

        def call_llm():
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(state.history[-6:])
            messages.append({"role": "user", "content": text})
            r = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return r.choices[0].message.content.strip()

        response = await state.loop.run_in_executor(None, call_llm)
        print(f"[LLM][AI] {response}")

        state.history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": response},
        ]
        await tts_text_queue.put(response)

# =========================================================
# 🔊 TTS WORKER (stream crudo a ffplay, sin archivos)
# =========================================================

async def tts_worker(state, tts_text_queue):
    print("[TTS] Worker iniciado")

    while state.running:
        text = await tts_text_queue.get()
        if text is None:
            break

        state.barge_in_event.clear()
        state.tts_playing = True

        print("[TTS][TTS-Start] Enviando texto a ElevenLabs…")
        print(f"[TTS] Texto: {text}")

        # ffplay: PCM crudo s16le mono desde stdin (-ac no está disponible en este contexto en ffplay)
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
        except FileNotFoundError:
            print("[TTS] ERROR: ffplay no encontrado en PATH. Instala ffmpeg.")
            state.tts_playing = False
            state.barge_in_event.clear()
            continue

        try:
            async for chunk in stream_tts_elevenlabs(text):
                if state.barge_in_event.is_set():
                    print("[CTRL-Barge-in] Deteniendo TTS por barge-in")
                    break

                if proc.stdin:
                    try:
                        proc.stdin.write(chunk)
                        proc.stdin.flush()
                    except BrokenPipeError:
                        print("[TTS] ffplay cerró stdin (Broken pipe)")
                        break

                print(f"[TTS][TTS-Chunk] {len(chunk)} bytes")
                await asyncio.sleep(0)
        except Exception as e:
            print(f"[TTS] Error durante streaming TTS: {e}")
        finally:
            state.tts_playing = False
            state.barge_in_event.clear()
            print("[TTS][TTS-End] Fin de reproducción")

            # Cerramos stdin y esperamos a que ffplay termine
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


# =========================================================
# 🎤 MICRÓFONO
# =========================================================

def create_input_stream(state, mic_queue, device):
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

async def mic_sender(state, mic_queue, dg_conn):
    print("[STT] Sender iniciado")
    while state.running:
        chunk = await mic_queue.get()
        if chunk is None:
            break
        dg_conn.send(chunk)

# =========================================================
# 🧠 DEEPGRAM STT + VAD
# =========================================================

def setup_deepgram(state, utterance_queue):
    dg = DeepgramClient()
    conn = dg.listen.websocket.v("1")

    # ---------- TRANSCRIPCIÓN ----------
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

        # 🔥 Barge-in por texto interim
        if state.tts_playing and not result.is_final and len(text.split()) >= 2:
            print("[VAD][VAD-Barge-in] Usuario habló (texto interim) → pedir corte TTS")
            state.barge_in_event.set()

        if result.is_final:
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

    conn.on(
        LiveTranscriptionEvents.Transcript,
        transcript_handler,
    )

    # ---------- VAD: SpeechStarted ----------
    def on_speech_started(event):
        print("[VAD][VAD-SpeechStarted] Evento inicio: empezó a hablar")

        if state.tts_playing:
            print("[VAD][VAD-Barge-in] Usuario habló (VAD SpeechStarted) → pedir corte TTS")
            state.barge_in_event.set()

    def speech_started_handler(*args, **kwargs):
        ev = kwargs.get("speech_started")
        if ev is None:
            if len(args) >= 2:
                ev = args[1]
            elif len(args) == 1:
                ev = args[0]
        state.loop.call_soon_threadsafe(on_speech_started, ev)

    conn.on(
        LiveTranscriptionEvents.SpeechStarted,
        speech_started_handler,
    )

    # ---------- VAD: UtteranceEnd ----------
    def on_utterance_end(event):
        print("[VAD][VAD-TALKING/STOP] Detenido por silencio (>=1000 ms)")
        print("[VAD][VAD-UtteranceEnd] Fin de enunciado (evento VAD)")

    def utterance_end_handler(*args, **kwargs):
        ev = kwargs.get("utterance_end")
        if ev is None:
            if len(args) >= 2:
                ev = args[1]
            elif len(args) == 1:
                ev = args[0]
        state.loop.call_soon_threadsafe(on_utterance_end, ev)

    conn.on(
        LiveTranscriptionEvents.UtteranceEnd,
        utterance_end_handler,
    )

    # ---------- ERRORES / CIERRE ----------
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

    # ---------- INICIO DEL STREAM ----------
    conn.start(
        LiveOptions(
            model="nova-3",
            language="es",
            encoding="linear16",
            sample_rate=MIC_SAMPLE_RATE,
            channels=1,
            interim_results=True,
            vad_events=True,
            utterance_end_ms=1000,
        )
    )

    print("[STT] Conectado a Deepgram")
    return conn

# =========================================================
# 🚀 MAIN
# =========================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    if args.input_device is not None or args.output_device is not None:
        sd.default.device = (args.input_device, args.output_device)

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("ERROR: DEEPGRAM_API_KEY no definido")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY no definido")
        return

    print(f"[AUDIO] Mic: {MIC_SAMPLE_RATE} Hz")

    state = State()
    openai_client = OpenAI()

    mic_queue = asyncio.Queue()
    utterance_queue = asyncio.Queue()
    tts_queue = asyncio.Queue()

    dg_conn = setup_deepgram(state, utterance_queue)
    mic_stream = create_input_stream(state, mic_queue, args.input_device)

    tasks = [
        asyncio.create_task(mic_sender(state, mic_queue, dg_conn)),
        asyncio.create_task(llm_worker(state, openai_client, utterance_queue, tts_queue)),
        asyncio.create_task(tts_worker(state, tts_queue)),
    ]

    mic_stream.start()
    print("🎧 LISTO — Habla cuando quieras (Ctrl+C para salir)")

    stop_event = asyncio.Event()

    def stop():
        stop_event.set()

    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop)
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

if __name__ == "__main__":
    asyncio.run(main())