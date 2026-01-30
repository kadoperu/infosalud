"""
Plataforma Pipeline Conversacional de AI en tiempo real.
  voz input → VAD Silero → STT Deepgram → LLM OpenAI → TTS ElevenLabs → voz output
"""
from .main import main
from .state import PipelineState
from .config import (
    MIC_SAMPLE_RATE,
    MIC_CHUNK_FRAMES,
    SYSTEM_PROMPT,
    SileroVADConfig,
)

__all__ = [
    "main",
    "PipelineState",
    "MIC_SAMPLE_RATE",
    "MIC_CHUNK_FRAMES",
    "SYSTEM_PROMPT",
    "SileroVADConfig",
]
