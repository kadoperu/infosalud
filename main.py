#!/usr/bin/env python3
import argparse
import asyncio
import os
import signal
from typing import List, Optional

import httpx
import numpy as np
import sounddevice as sd
from openai import OpenAI
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

from dotenv import load_dotenv
load_dotenv()

SAMPLE_RATE = 16000 # 16000 es el valor por defecto, 48000 es el valor más alto que soporta el micrófono
CHANNELS = 1
CHUNK_MS = 40  # tamaño de bloque de micrófono (ms). ¡Puedes afinarlo!
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)

# python main.py --list-devices
# python main.py --input-device 1 --output-device 2

SYSTEM_PROMPT = (
    "Eres un asistente de voz breve, claro y conversacional. "
    "Contesta en frases cortas, en español, sin emojis."
)


class State:
    """Estado compartido entre tareas y callbacks."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self.barge_in_event: asyncio.Event = asyncio.Event()
        self.tts_playing: bool = False
        self.history: List[dict] = []
        self.utterance_buffer: List[str] = []
        self.running: bool = True


async def synth_tts_deepgram(text: str) -> bytes:
    """
    Llama al TTS REST de Deepgram y devuelve audio PCM16 mono 16k.
    Usa modelo Aura-2 (ajustable).
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY no está definido.")

    params = {
        "model": "aura-2-thalia-en",  # puedes cambiar de voz/modelo
        "encoding": "linear16",
        "sample_rate": str(SAMPLE_RATE),
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            "https://api.deepgram.com/v1/speak",
            params=params,
            headers=headers,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            chunks = [chunk async for chunk in resp.aiter_bytes()]
    return b"".join(chunks)


async def llm_worker(
    state: State,
    openai_client: OpenAI,
    utterance_queue: asyncio.Queue,
    tts_text_queue: asyncio.Queue,
) -> None:
    """
    Consume textos reconocidos (utterance_queue),
    llama al LLM y manda la respuesta a TTS.
    """
    print("[LLM] Worker iniciado.")
    while state.running:
        try:
            text = await utterance_queue.get()
            if text is None:
                break

            print(f"[LLM] Usuario dijo: {text}")
            print("[LLM] Generando respuesta...")

            def _call_llm() -> str:
                # Mantener últimos N turnos de contexto
                N = 6
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(state.history[-N:])
                messages.append({"role": "user", "content": text})

                completion = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                return completion.choices[0].message.content.strip()

            # Ejecutar LLM en thread para no bloquear asyncio
            response_text = await state.loop.run_in_executor(None, _call_llm)

            print(f"[LLM] Respuesta: {response_text}")

            # Actualizar historial
            state.history.append({"role": "user", "content": text})
            state.history.append({"role": "assistant", "content": response_text})

            await tts_text_queue.put(response_text)

        except Exception as e:
            print(f"[LLM] Error: {e}")


async def tts_worker(
    state: State,
    tts_text_queue: asyncio.Queue,
) -> None:
    """
    Consume textos para TTS, sintetiza con Deepgram y reproduce con sounddevice.
    Soporta barge-in usando state.barge_in_event + sounddevice.stop().
    """
    print("[TTS] Worker iniciado.")
    while state.running:
        try:
            text = await tts_text_queue.get()
            if text is None:
                break

            print("[TTS] Sintetizando con Deepgram...")
            audio_bytes = await synth_tts_deepgram(text)

            # PCM16 mono 16k
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            duration = len(audio_np) / float(SAMPLE_RATE)
            if duration <= 0:
                print("[TTS] Audio vacío, saltando.")
                continue

            # Resetear evento de barge-in para esta locución
            state.barge_in_event.clear()
            state.tts_playing = True

            print(f"[AUDIO] Reproduciendo TTS (duración ~{duration:.2f}s)...")
            sd.play(audio_np, samplerate=SAMPLE_RATE, blocking=False)

            try:
                # Espera hasta que:
                #   - termine el audio (timeout)
                #   - o llegue un barge-in (evento)
                await asyncio.wait_for(
                    state.barge_in_event.wait(), timeout=duration + 0.5
                )
                # Si entra aquí por barge-in:
                if state.barge_in_event.is_set():
                    print("[CTRL] Barge-in: stopping TTS (evento detectado).")
            except asyncio.TimeoutError:
                # No hubo barge-in; playback terminó (o casi)
                pass
            finally:
                sd.stop()
                state.tts_playing = False
                state.barge_in_event.clear()

        except Exception as e:
            print(f"[TTS] Error: {e}")
            state.tts_playing = False
            state.barge_in_event.clear()


def create_input_stream(
    state: State,
    mic_queue: asyncio.Queue,
    input_device: Optional[int],
) -> sd.RawInputStream:
    """
    Crea un stream de entrada desde el micrófono y manda bloques a mic_queue.
    """
    def callback(in_data, frames, time_info, status):
        if status:
            print(f"[AUDIO] Input status: {status}")
        # in_data ya viene en bytes (RawInputStream, dtype=int16)
        # Pasamos el bloque al loop de asyncio de forma thread-safe
        state.loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(in_data))

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_FRAMES,
        dtype="int16",
        channels=CHANNELS,
        callback=callback,
        device=input_device,
    )
    return stream


async def mic_to_stt_sender(
    state: State,
    mic_queue: asyncio.Queue,
    dg_connection,
) -> None:
    """
    Consume bloques de audio del micrófono y los envía al WebSocket de Deepgram.
    """
    print("[STT] Sender iniciado.")
    while state.running:
        chunk = await mic_queue.get()
        if chunk is None:
            break
        try:
            dg_connection.send(chunk)
        except Exception as e:
            print(f"[STT] Error al enviar audio a Deepgram: {e}")
            # Aquí podrías implementar reconexión más avanzada
            await asyncio.sleep(1.0)



def setup_deepgram_stt(
    state: State,
    utterance_queue: asyncio.Queue,
):
    """
    Configura el cliente de Deepgram STT en streaming con VAD / UtteranceEnd.
    Devuelve el objeto de conexión (websocket).
    """
    dg_client = DeepgramClient()  # usa DEEPGRAM_API_KEY del entorno

    dg_connection = dg_client.listen.websocket.v("1")  # WebSocket v1

    def handle_transcript(result):
        try:
            alt = result.channel.alternatives[0]
            transcript = (alt.transcript or "").strip()
        except Exception as e:
            print(f"[STT] Error parseando Transcript interno: {e}")
            return

        if not transcript:
            return

        is_final = getattr(result, "is_final", False)

        # --- LÓGICA DE BARGE-IN MÁS TOLERANTE ---
        # Si TTS está sonando y llega texto interim con cierta longitud,
        # lo interpretamos como que el usuario realmente está hablando
        # y cortamos el TTS.
        if state.tts_playing and not is_final:
            # Umbrales de tolerancia:
            MIN_CHARS = 5      # mínimo de caracteres
            MIN_WORDS = 2      # mínimo de palabras

            num_chars = len(transcript)
            num_words = len(transcript.split())

            if num_chars >= MIN_CHARS and num_words >= MIN_WORDS:
                print(f"[CTRL] Barge-in por STT interim: '{transcript}'")
                state.barge_in_event.set()
        # --- FIN BARGE-IN ---

        if is_final:
            print(f"[STT] Final: {transcript}")
            state.utterance_buffer.append(transcript)
        else:
            print(f"[STT] Interim: {transcript}")

    # Originalmente, se cortaba TTS cuando se detectaba inicio de voz.
    # def handle_speech_started(_event):
    #     # Se dispara cuando Deepgram detecta inicio de voz
    #     print("[VAD] SpeechStarted (Deepgram).")
    #     # Si hay TTS sonando, hacemos barge-in
    #     if state.tts_playing:
    #         print("[CTRL] Barge-in: stopping TTS… (SpeechStarted)")
    #         state.barge_in_event.set()

    # Ahora solo logueamos, NO cortamos TTS aquí.
    def handle_speech_started(_event):
        # Se dispara cuando Deepgram detecta inicio de voz
        # Ahora solo logueamos, NO cortamos TTS aquí.
        print("[VAD] SpeechStarted (Deepgram).")



    def handle_utterance_end(_event):
        # Se dispara cuando Deepgram detecta el final de un enunciado
        if not state.utterance_buffer:
            return
        text = " ".join(state.utterance_buffer).strip()
        state.utterance_buffer.clear()
        if not text:
            return
        print(f"[STT] UtteranceEnd: {text}")
        utterance_queue.put_nowait(text)

    def handle_error(error, **kwargs):
        print(f"[STT] Error en WebSocket de Deepgram: {error}")

    def handle_close(close, **kwargs):
        print(f"[STT] WebSocket cerrado: {close}")

    def _extract_event(arg_name: str, args, kwargs):
        """
        Deepgram 3.x puede llamar así:
          handler(self, event, **kwargs)
        o
          handler(event, **kwargs)
        o pasar el evento en kwargs.
        """
        if arg_name in kwargs:
            return kwargs[arg_name]
        if len(args) >= 2:
            return args[1]
        if len(args) == 1:
            return args[0]
        return None

    # Wrappers que se ejecutan en el hilo del SDK
    def on_transcript(*args, **kwargs):
        event = _extract_event("result", args, kwargs)
        if event is None:
            print("[STT] on_transcript sin evento válido, args:", args, "kwargs:", kwargs)
            return
        state.loop.call_soon_threadsafe(handle_transcript, event)

    def on_speech_started(*args, **kwargs):
        event = _extract_event("speech_started", args, kwargs)
        state.loop.call_soon_threadsafe(handle_speech_started, event)

    def on_utterance_end(*args, **kwargs):
        event = _extract_event("utterance_end", args, kwargs)
        state.loop.call_soon_threadsafe(handle_utterance_end, event)

    def on_error(*args, **kwargs):
        err = _extract_event("error", args, kwargs)
        state.loop.call_soon_threadsafe(handle_error, err)

    def on_close(*args, **kwargs):
        cl = _extract_event("close", args, kwargs)
        state.loop.call_soon_threadsafe(handle_close, cl)

    # Registro de eventos (forma correcta para deepgram-sdk 3.x)
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
    dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    dg_connection.on(LiveTranscriptionEvents.Close, on_close)

    # Configuración del streaming con VAD/UtteranceEnd
    options = LiveOptions(
        model="nova-3",          # modelo STT
        language="es",           # español
        smart_format=True,
        encoding="linear16",
        channels=CHANNELS,
        sample_rate=SAMPLE_RATE,
        # Para UtteranceEnd + VAD:
        interim_results=True,
        utterance_end_ms="1000",  # 1s de silencio
        vad_events=True,
    )

    print("[STT] Conectando a Deepgram (streaming)…")
    dg_connection.start(options)
    print("[STT] Conexión Deepgram lista.")
    return dg_connection




def list_audio_devices() -> None:
    """Imprime dispositivos de audio disponibles."""
    print("=== Dispositivos de audio ===")
    print(sd.query_devices())


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asistente de voz por consola (Deepgram + OpenAI)."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Listar dispositivos de audio y salir.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Índice de dispositivo de entrada (micrófono).",
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Índice de dispositivo de salida (parlantes/auriculares).",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Configurar dispositivos por defecto si se especifican
    if args.input_device is not None or args.output_device is not None:
        sd.default.device = (args.input_device, args.output_device)
        print(f"[AUDIO] Usando input_device={args.input_device}, output_device={args.output_device}")


    # Ajustar SAMPLE_RATE automáticamente al del dispositivo de entrada
    global SAMPLE_RATE, CHUNK_FRAMES
    try:
        # Si no se especifica dispositivo, usar el input por defecto
        dev_index = args.input_device if args.input_device is not None else None
        dev_info = sd.query_devices(dev_index, "input")
        SAMPLE_RATE = int(dev_info["default_samplerate"])
        CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)
        print(f"[AUDIO] Usando sample rate {SAMPLE_RATE} Hz para mic, STT y TTS")
    except Exception as e:
        print(f"[AUDIO] No se pudo obtener sample rate del dispositivo: {e}")
        print("[AUDIO] Usando valor por defecto 16000 Hz (puede fallar si el dispositivo no lo soporta).")

    # Validar API keys


    # Validar API keys
    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("ERROR: DEEPGRAM_API_KEY no está definido.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY no está definido.")
        return

    # Crear estado y clientes
    state = State()
    openai_client = OpenAI()

    # Colas
    mic_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    utterance_queue: asyncio.Queue = asyncio.Queue()
    tts_text_queue: asyncio.Queue = asyncio.Queue()

    # STT Deepgram
    dg_connection = setup_deepgram_stt(state, utterance_queue)

    # Mic stream
    try:
        in_stream = create_input_stream(
            state=state,
            mic_queue=mic_queue,
            input_device=args.input_device,
        )
    except Exception as e:
        print(f"[AUDIO] Error creando stream de micrófono: {e}")
        dg_connection.finish()
        return

    # Tareas asyncio
    tasks = [
        asyncio.create_task(mic_to_stt_sender(state, mic_queue, dg_connection)),
        asyncio.create_task(llm_worker(state, openai_client, utterance_queue, tts_text_queue)),
        asyncio.create_task(tts_worker(state, tts_text_queue)),
    ]

    # Iniciar captura de audio
    in_stream.start()
    print("=======================================")
    print("[CTRL] Ready / Listening… (Ctrl+C para salir)")
    print("=======================================")

    # Manejo de Ctrl+C limpio
    loop = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def _handle_sigint():
        print("\n[CTRL] Señal de parada recibida. Cerrando…")
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
    except NotImplementedError:
        # Windows o entornos que no soportan signals
        pass

    # Bucle principal en espera hasta Ctrl+C
    await stop_event.wait()

    # Parar estado
    state.running = False

    # Cerrar streams y conexión
    try:
        in_stream.stop()
        in_stream.close()
    except Exception:
        pass

    sd.stop()

    try:
        dg_connection.finish()
    except Exception:
        pass

    # Vaciar colas para desbloquear tareas
    for q in (mic_queue, utterance_queue, tts_text_queue):
        try:
            q.put_nowait(None)
        except Exception:
            pass

    # Cancelar tareas restantes
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print("[CTRL] Salida limpia. Bye.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Por si acaso
        print("\n[CTRL] Interrumpido por teclado.")
