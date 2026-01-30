"""
Utilidades de logging para el pipeline de voz.
Helpers opcionales para mensajes estructurados o colores.
"""
import logging

# Por ahora solo definimos un logger del pipeline; se puede extender con colores o formato.
PIPELINE_LOGGER = logging.getLogger("voice_pipeline")


def log_vad(tag: str, msg: str) -> None:
    """Log de eventos VAD."""
    print(f"[VAD][{tag}] {msg}")


def log_stt(tag: str, msg: str) -> None:
    """Log de eventos STT."""
    print(f"[STT][{tag}] {msg}")


def log_tts(tag: str, msg: str) -> None:
    """Log de eventos TTS."""
    print(f"[TTS][{tag}] {msg}")


def log_llm(tag: str, msg: str) -> None:
    """Log de eventos LLM."""
    print(f"[LLM][{tag}] {msg}")
