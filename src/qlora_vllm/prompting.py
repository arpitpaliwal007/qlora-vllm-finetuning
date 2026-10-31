"""The single prompt template shared by data preparation, training and evaluation."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a question answering assistant. Answer the question using only the provided context. "
    "Reply with the shortest span of the context that answers the question, and nothing else."
)
USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


def build_messages(context: str, question: str, answer: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def prompt_messages(record: dict) -> list[dict[str, str]]:
    """Return the prompt (everything before the assistant turn) of a prepared record."""
    messages = record["messages"]
    if messages[-1]["role"] == "assistant":
        return messages[:-1]
    return messages
