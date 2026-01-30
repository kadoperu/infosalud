"""
Conector entre pipeline de voz y LLM (OpenAI).
llm_worker: recibe texto de utterance_queue → llama al modelo → pone respuesta en tts_text_queue.
"""
import asyncio
from typing import Any

from openai import OpenAI

from ..config import SYSTEM_PROMPT


async def llm_worker(
    state,
    openai_client: OpenAI,
    utterance_queue: asyncio.Queue,
    tts_text_queue: asyncio.Queue,
) -> None:
    """
    Worker que recibe transcripciones finales, llama al LLM (OpenAI) y pone
    la respuesta en la cola de TTS. Mantiene state.history para contexto.
    """
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
