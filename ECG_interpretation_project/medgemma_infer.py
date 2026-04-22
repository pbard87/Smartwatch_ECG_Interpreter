from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from threading import Lock

# Module-level cache for model and processor to avoid reloading weights on every call
_MODEL = None
_PROCESSOR = None
_LOAD_LOCK = Lock()
_MODEL_NAME = "google/medgemma-1.5-4b-it"


def _load_model_once():
    global _MODEL, _PROCESSOR
    if _MODEL is None or _PROCESSOR is None:
        with _LOAD_LOCK:
            if _MODEL is None or _PROCESSOR is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"loading model on device: {device}")
                _PROCESSOR = AutoProcessor.from_pretrained(_MODEL_NAME, device_map="auto")
                _MODEL = AutoModelForImageTextToText.from_pretrained(_MODEL_NAME, device_map="auto")
                _MODEL.eval()


def infer_local(prompt, image_path, generation_kwargs=None):
    """Infer using a cached model and processor. Returns (response_text, raw_outputs)."""
    _load_model_once()
    # build messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt}
            ]
        },
    ]
    inputs = _PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_MODEL.device)

    # Use inference mode to reduce memory and disable grad
    gen_kwargs = {} if generation_kwargs is None else dict(generation_kwargs)
    # ensure max_new_tokens is set
    gen_kwargs.setdefault("max_new_tokens", 512 + 256)
    with torch.inference_mode():
        outputs = _MODEL.generate(**inputs, **gen_kwargs)

    response = _PROCESSOR.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    return response, outputs
