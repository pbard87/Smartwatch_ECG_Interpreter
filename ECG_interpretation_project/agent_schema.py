from typing import Optional, Tuple, Any
import json
import jsonschema
from jsonschema import ValidationError
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError, field_validator

# JSON Schema for model output
ECG_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "heart_rate_bpm": {"type": ["number", "null"]},
        "rhythm": {"type": ["string", "null"]},
        "axis": {"type": ["string", "null"]},
        "intervals": {
            "type": ["object", "null"],
            "properties": {
                "PR_ms": {"type": ["number", "null"]},
                "QRS_ms": {"type": ["number", "null"]},
                "QT_ms": {"type": ["number", "null"]},
                "QTc_ms": {"type": ["number", "null"]},
                "method": {"type": ["string", "null"]}
            },
            "additionalProperties": False
        },
        "conduction": {"type": ["string", "null"]},
        "hypertrophy": {"type": ["string", "null"]},
        "ischemia_or_injury": {"type": ["string", "null"]},
        "ectopy": {"type": ["string", "null"]},
        "artifact": {"type": ["string", "null"]},
        "overall_impression": {"type": ["string", "null"]},
        "confidence_0to1": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["confidence_0to1"],
    "additionalProperties": False
}


class InterpIntervals(BaseModel):
    PR_ms: Optional[float] = None
    QRS_ms: Optional[float] = None
    QT_ms: Optional[float] = None
    QTc_ms: Optional[float] = None
    method: Optional[str] = None


class ECGInterpretation(BaseModel):
    heart_rate_bpm: Optional[float] = None
    rhythm: Optional[str] = None
    axis: Optional[str] = None
    intervals: Optional[InterpIntervals] = None
    conduction: Optional[str] = None
    hypertrophy: Optional[str] = None
    ischemia_or_injury: Optional[str] = None
    ectopy: Optional[str] = None
    artifact: Optional[str] = None
    overall_impression: Optional[str] = None
    confidence_0to1: float = Field(..., ge=0.0, le=1.0)

    @field_validator("rhythm", "axis", "conduction", "hypertrophy", "ischemia_or_injury",
                    "ectopy", "artifact", "overall_impression", mode="before")
    def empty_to_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v


def parse_json_safe(raw: Any) -> Tuple[Optional[dict], Optional[str]]:
    """Try to coerce raw model output into a JSON dict. Return (dict, error_str)."""
    if isinstance(raw, dict):
        return raw, None
    try:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
            # first try parsing the whole text
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return obj, None
                return None, "parsed JSON is not an object"
            except Exception:
                # attempt to extract a JSON object substring from the text
                # find the first balanced JSON object using brace counting
                start_idx = None
                depth = 0
                for i, ch in enumerate(text):
                    if ch == '{':
                        if start_idx is None:
                            start_idx = i
                        depth += 1
                    elif ch == '}':
                        if start_idx is not None:
                            depth -= 1
                            if depth == 0:
                                candidate = text[start_idx:i+1]
                                try:
                                    obj = json.loads(candidate)
                                    if isinstance(obj, dict):
                                        return obj, None
                                except Exception:
                                    # continue searching
                                    start_idx = None
                                    depth = 0
                return None, "parsed JSON not found in text"
        return None, "unsupported raw type"
    except Exception as e:
        return None, f"json parse error: {e}"


def validate_with_schema(obj: dict) -> Tuple[bool, Optional[str]]:
    try:
        jsonschema.validate(instance=obj, schema=ECG_OUTPUT_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e)


def validate_with_pydantic(obj: dict) -> Tuple[Optional[ECGInterpretation], Optional[str]]:
    try:
        model = ECGInterpretation.model_validate(obj)
        return model, None
    except PydanticValidationError as e:
        return None, str(e)


def validate_model_output(raw: Any) -> Tuple[bool, Optional[ECGInterpretation], str]:
    """
    Full validation pipeline:
    - coerce to JSON
    - jsonschema structural validation
    - pydantic parsing for typed validation
    Returns (is_valid, parsed_model_or_None, message)
    """
    obj, parse_err = parse_json_safe(raw)
    if obj is None:
        return False, None, f"parse_error: {parse_err}"
    ok, schema_err = validate_with_schema(obj)
    if not ok:
        return False, None, f"schema_error: {schema_err}"
    model, p_err = validate_with_pydantic(obj)
    if model is None:
        return False, None, f"pydantic_error: {p_err}"
    return True, model, "ok"


RE_PROMPT_TEMPLATE = (
    "Your previous answer failed schema validation for the expected ECG JSON.\n"
    "Validation error: {error}\n"
    "Return only a JSON object matching this schema exactly (no commentary):\n"
    "{schema}\n"
    "If a value is unknown, use null.\n"
)

RE_PROMPT_WITH_EXAMPLE = (
    "The JSON must exactly follow this example structure (use null for unknowns):\n"
    "{example}\n"
    "Return only the JSON object and nothing else.\n"
)
def requery_with_schema(infer_fn, prompt_context: str, image_path: str, validation_error: str, schema: dict, attempts: int = 3):
    """Call infer_fn to re-request valid JSON until attempts exhausted.
    Strategy:
      - Attempt 0: ask for JSON matching schema
      - Attempt 1: provide a concrete example JSON and ask for only JSON
      - Attempt 2+: provide example + enforce deterministic generation kwargs
    infer_fn is expected to accept (prompt=..., image_path=..., generation_kwargs=...) and return (raw_out, raw_meta)
    """
    last_raw = None
    last_meta = None

    # small canonical example with nulls for optional fields
    example_obj = {k: None for k in schema.get("properties", {}).keys()}
    # set a plausible example for confidence
    if "confidence_0to1" in example_obj:
        example_obj["confidence_0to1"] = 0.9

    for attempt in range(attempts):
        if attempt == 0:
            extra = RE_PROMPT_TEMPLATE.format(error=validation_error, schema=json.dumps(schema, indent=2))
            gen_kwargs = None
        elif attempt == 1:
            extra = RE_PROMPT_TEMPLATE.format(error=validation_error, schema=json.dumps(schema, indent=2)) + "\n" + RE_PROMPT_WITH_EXAMPLE.format(example=json.dumps(example_obj, indent=2))
            gen_kwargs = None
        else:
            extra = RE_PROMPT_TEMPLATE.format(error=validation_error, schema=json.dumps(schema, indent=2)) + "\n" + RE_PROMPT_WITH_EXAMPLE.format(example=json.dumps(example_obj, indent=2))
            # force deterministic generation (beam search, no sampling)
            gen_kwargs = {"num_beams": 4, "do_sample": False, "temperature": 0.0}

        new_prompt = prompt_context + "\n\n" + extra
        try:
            if gen_kwargs is not None:
                raw_out, raw_meta = infer_fn(prompt=new_prompt, image_path=image_path, generation_kwargs=gen_kwargs)
            else:
                raw_out, raw_meta = infer_fn(prompt=new_prompt, image_path=image_path)
        except TypeError:
            # fallback if infer_fn doesn't accept generation_kwargs
            raw_out, raw_meta = infer_fn(prompt=new_prompt, image_path=image_path)

        last_raw = raw_out
        last_meta = raw_meta
        ok, model, message = validate_model_output(raw_out)
        if ok:
            return True, model, raw_out, raw_meta
    return False, None, last_raw, last_meta
