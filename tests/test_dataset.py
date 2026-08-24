from qlora_vllm.dataset import content_to_text


def test_content_to_text_handles_openai_text_parts():
    assert content_to_text([{"type": "text", "text": "hello"}, {"type": "image_url"}]) == "hello"
