from __future__ import annotations

import os
from typing import List, Optional

from .contracts import ChatMessage

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


SYSTEM_PROMPT = """You are a graph RAG assistant.
Answer using the provided context first.
If the context is not enough, explicitly say what is missing.
Keep the answer concise and factual."""


class OpenAIChatGenerator:
    def __init__(
        self,
        *,
        default_model: str = "gpt-5-nano",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is not installed. Install `openai` first.") from exc

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.default_model = default_model
        self._client = OpenAI(api_key=key)

    def generate(
        self,
        *,
        question: str,
        history: List[ChatMessage],
        context_chunks: List[str],
        model: Optional[str] = None,
    ) -> str:
        target_model = str(model or self.default_model)
        context_text = "\n".join(context_chunks).strip()
        if not context_text:
            context_text = "No retrieval context was found."

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n{question.strip()}\n\n"
                    f"Context:\n{context_text}\n\n"
                    "Please answer based on the context."
                ),
            }
        )

        response = self._client.chat.completions.create(
            model=target_model,
            messages=messages
        )
        print(response)  # For debugging purposes
        output = (response.choices[0].message.content or "").strip()
        if not output:
            return "I could not generate a response."
        return output
