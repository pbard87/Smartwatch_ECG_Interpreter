import time
import json
import uuid
from typing import Optional, Tuple, Any
from medgemma_infer import infer_local
from agent_schema import validate_model_output, requery_with_schema, ECG_OUTPUT_SCHEMA
from agent_schema import parse_json_safe
from ecg_tools import extract_ecg_strip_from_pdf, extract_text_from_pdf


class ECGAgent:
    """Orchestrates ECG interpretation steps: prompt creation, model call, validation, re-query, and postprocessing."""

    def __init__(self, infer_fn=infer_local):
        self.infer_fn = infer_fn

    def build_prompt(self, age: int, sex: str, diagnoses: Any, symptoms: Any, pdf_text: Optional[str] = None) -> str:
        base = (
            "You are a helpful medical assistant specialized in ECG interpretation. "
            "Please interpret the included one-lead ECG recorded by the patient provided this context:\n\n"
            f"A {age} year-old, {sex} presents with {symptoms}.\n"
            f"Medical history: {diagnoses}.\n"
        )
        schema_hint = (
            "Return a JSON object with the following keys: \n"
            f"{json.dumps(list(ECG_OUTPUT_SCHEMA['properties'].keys()))}\n"
            "Use null for unknown values and do not invent numbers."
        )
        if pdf_text:
            base += "\nAdditional extracted text from the PDF:\n" + pdf_text + "\n\n"
        return base + "\n" + schema_hint

    def call_model(self, prompt: str, image_path: Optional[str] = None, generation_kwargs: Optional[dict] = None) -> Tuple[Any, Any, float]:
        start = time.perf_counter()
        # infer_fn may accept generation_kwargs
        try:
            raw_out, raw_meta = self.infer_fn(prompt=prompt, image_path=image_path, generation_kwargs=generation_kwargs)
        except TypeError:
            raw_out, raw_meta = self.infer_fn(prompt=prompt, image_path=image_path)
        elapsed = time.perf_counter() - start
        return raw_out, raw_meta, elapsed

    def validate_and_requery(self, raw_out: Any, prompt: str, image_path: Optional[str] = None, attempts: int = 2):
        ok, model, msg = validate_model_output(raw_out)
        if ok:
            return True, model, raw_out, None
        success, model2, raw_out2, meta2 = requery_with_schema(self.infer_fn, prompt, image_path, msg, ECG_OUTPUT_SCHEMA, attempts=attempts)
        if success:
            return True, model2, raw_out2, meta2
        return False, None, raw_out, None

    def postprocess_for_display(self, model_obj) -> str:
        if model_obj is None:
            return {"error msg:", "No valid JSON format interpretation available."}
        # pydantic v2 model_dump for serialization
        try:
            return json.dumps(model_obj.model_dump(), indent=2)
        except Exception:
            return str(model_obj)

    def run(self, age: int, sex: str, diagnoses: Any, symptoms: Any, image_path: Optional[str], image=None, pdf_text: Optional[str] = None, file_name: Optional[str] = None):
        run_id = str(uuid.uuid4())
        prompt = self.build_prompt(age, sex, diagnoses, symptoms, pdf_text=pdf_text)
        raw_out, raw_meta, inf_time = self.call_model(prompt, image_path=image_path)
        valid, model_obj, final_raw, raw_meta2 = self.validate_and_requery(raw_out, prompt, image_path=image_path)

        # If validated but confidence is low, attempt to re-extract PDF and re-run
        reextract_attempts = []
        confidence_threshold = 0.6
        max_reextracts = 2
        attempts_done = 0
        current_raw = final_raw if final_raw is not None else raw_out
        current_model = model_obj
        current_prompt = prompt
        current_pdf_text = pdf_text
        while current_model is not None and hasattr(current_model, 'confidence_0to1') and current_model.confidence_0to1 < confidence_threshold and attempts_done < max_reextracts:
            attempts_done += 1
            # re-extract image and text
            try:
                new_image, new_image_file = extract_ecg_strip_from_pdf(image_path, threshold_factor=2.0+attempts_done)
                new_pdf_text = extract_text_from_pdf(image_path)
            except Exception as e:
                reextract_attempts.append({"attempt": attempts_done, "error": str(e)})
                break

            # rebuild prompt with new text
            current_pdf_text = new_pdf_text
            current_prompt = self.build_prompt(age, sex, diagnoses, symptoms, pdf_text=current_pdf_text)
            # call model again with deterministic generation to reduce variability
            gen_kwargs = {"num_beams": 4, "do_sample": False, "temperature": 0.0}
            new_raw_out, new_raw_meta, new_inf_time = self.call_model(current_prompt, image_path=new_image_file, generation_kwargs=gen_kwargs)
            ok, new_model_obj, new_final_raw, _ = self.validate_and_requery(new_raw_out, current_prompt, image_path=new_image_file)
            reextract_attempts.append({"attempt": attempts_done, "validated": ok, "confidence": getattr(new_model_obj, 'confidence_0to1', None) if new_model_obj else None})
            # update current
            current_raw = new_final_raw if new_final_raw is not None else new_raw_out
            current_model = new_model_obj

        # if we updated via re-extraction, use those values
        if attempts_done > 0:
            final_raw = current_raw
            model_obj = current_model
            # include reextract attempts in logs
            log_reextract = reextract_attempts
        else:
            log_reextract = []

        # prepare logs
        log = {
            "run_id": run_id,
            "file_name": file_name,
            "inference_time": inf_time,
            "raw_model_output": final_raw,
            "validated": valid,
            "reextract_attempts": log_reextract,
        }

        # build display json response
        display_json = self.postprocess_for_display(model_obj)

        # return raw text too for UI inspection
        # raw_text = raw_out if isinstance(raw_out, str) else str(raw_out)
        raw_text = final_raw if isinstance(final_raw, str) else str(final_raw)


        # attempt to remove JSON substring from model response for a cleaned text
        def _remove_json_substring(text: str) -> str:
            if not isinstance(text, str):
                return str(text)
            # try to parse JSON first
            obj, _ = parse_json_safe(text)
            if obj is None:
                return text
            # find first balanced JSON object and remove it
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
                            # remove the JSON slice and any surrounding backticks or whitespace
                            before = text[:start_idx]
                            after = text[i+1:]
                            cleaned = (before + after).strip()
                            # remove surrounding code fences if left
                            if cleaned.startswith('```') and cleaned.endswith('```'):
                                cleaned = cleaned.strip('`')
                            return cleaned.strip()
            return text

        cleaned_text = _remove_json_substring(raw_text)

        return image, display_json, raw_text, cleaned_text, log
