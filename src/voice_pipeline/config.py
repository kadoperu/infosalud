"""
Configuración central del pipeline de voz.
Sample rates, tamaños de chunk, prompts y variables de entorno.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------------------------------
# Audio / Mic
# -----------------------------------------------------------------------------
MIC_SAMPLE_RATE = 16000
CHANNELS = 1

# Silero VAD espera chunks de 512 muestras a 16 kHz (32 ms)
SILERO_WINDOW_SAMPLES = 512
CHUNK_MS = int(1000 * SILERO_WINDOW_SAMPLES / MIC_SAMPLE_RATE)
MIC_CHUNK_FRAMES = SILERO_WINDOW_SAMPLES

# -----------------------------------------------------------------------------
# TTS (ElevenLabs)
# -----------------------------------------------------------------------------
TTS_SAMPLE_RATE = 16000
TTS_OUTPUT_FORMAT = f"pcm_{TTS_SAMPLE_RATE}"

# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Eres un asistente de voz breve, claro y conversacional. "
    "Responde en español, frases cortas, no mas de 15 palabras"
)


# -----------------------------------------------------------------------------
# Credenciales (validación bajo demanda)
# -----------------------------------------------------------------------------
def get_eleven_api_key() -> str:
    key = os.environ.get("ELEVEN_API_KEY")
    if not key:
        raise RuntimeError("ELEVEN_API_KEY no definido")
    return key


def get_eleven_voice_id() -> str:
    vid = os.environ.get("ELEVEN_VOICE_ID")
    if not vid:
        raise RuntimeError("ELEVEN_VOICE_ID no definido")
    return vid


def get_deepgram_api_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY no definido")
    return key


def get_openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY no definido")
    return key


# -----------------------------------------------------------------------------
# VAD (Silero) - parámetros opcionales
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SileroVADConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 400
    speech_pad_ms: int = 30
