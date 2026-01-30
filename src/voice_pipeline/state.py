"""
Estado en tiempo de ejecución del pipeline de voz.
History, barge-in, flags de ejecución y último transcript final.
"""
import asyncio
from typing import Any, List


class PipelineState:
    """Estado compartido entre VAD, STT, LLM y TTS."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.barge_in_event = asyncio.Event()
        self.tts_playing = False
        self.history: List[dict[str, Any]] = []
        self.running = True
        # Último transcript final de Deepgram; se envía al LLM cuando Silero marca fin de voz
        self.last_final_transcript = ""
