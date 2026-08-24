from __future__ import annotations

from typing import Any


def content_to_text(content: Any) -> str:
    """Normalize OpenAI-style text parts to the string expected by chat templates."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    raise ValueError("message content must be a string or a list of text parts")


def format_example(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    messages = [
        {"role": message["role"], "content": content_to_text(message["content"])}
        for message in example["messages"]
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
