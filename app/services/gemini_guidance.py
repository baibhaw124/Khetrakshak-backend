import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_FILE)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
REQUEST_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "90000"))
MAX_ATTEMPTS = 2


def _is_transient_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    return status in {429, 500, 502, 503, 504} or any(
        marker in message
        for marker in ("deadline_exceeded", "deadline expired", "timed out", "temporarily unavailable", "429", "500", "502", "503", "504")
    )


def _validate_guidance(result: object) -> dict:
    required = {"treatment_steps", "prevention", "product_suggestions", "caution"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise ValueError("Gemini returned an unexpected response structure.")
    for key in ("treatment_steps", "prevention", "product_suggestions"):
        if not isinstance(result[key], list):
            raise ValueError(f"Gemini returned invalid {key}.")
    if not isinstance(result["caution"], str):
        raise ValueError("Gemini returned an invalid caution field.")
    for item in result["product_suggestions"]:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str)
            for key in ("category", "purpose", "safety_note")
        ):
            raise ValueError("Gemini returned an invalid product suggestion.")
    return result


def generate_treatment_guidance(
    crop: str, disease: str, confidence: float, severity: str,
    symptoms: list, recommendations: list, prevention: list,
) -> dict:
    """Generate cautious crop guidance; local treatment remains the fallback."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
    context = {
        "crop": crop,
        "detected_disease": disease,
        "model_confidence_percent": round(confidence * 100, 2) if confidence <= 1 else round(confidence, 2),
        "severity": severity,
        "known_symptoms": symptoms,
        "existing_recommendations": recommendations,
        "existing_prevention": prevention,
    }
    prompt = f"""
You are an agricultural crop-care assistant for farmers in India.
The supplied ML result is a prediction, not a confirmed diagnosis.
Assessment:
{json.dumps(context, ensure_ascii=False)}

Return ONLY valid JSON with:
- treatment_steps: array of concise strings
- prevention: array of concise strings
- product_suggestions: array of objects with category, purpose, safety_note
- caution: string

Rules:
1. Respect the crop, suspected disease, symptoms and supplied local guidance.
2. Start with integrated disease management, monitoring, sanitation and cultural practices.
3. Suggest NON-CHEMICAL tools or supplies only. Where genuinely relevant, use one or more
   of these exact category names: pruning shears, garden gloves, mulching film,
   drip irrigation kit, plant support clips, crop netting, leaf moisture meter,
   debris collection bags. Do not invent other product categories.
4. Do not suggest pesticides, fungicides, herbicides, chemical active ingredients, doses or brands.
5. Never invent prices, stock, sellers or purchase URLs.
6. If chemical crop protection may be necessary, advise consulting a local agricultural officer
   and using only products registered for the crop and disease, following the approved label.
7. Keep advice practical, cautious and understandable.
"""
    try:
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=1000,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini returned an empty response.")
                return _validate_guidance(json.loads(response.text))
            except Exception as error:
                last_error = error
                if attempt + 1 >= MAX_ATTEMPTS or not _is_transient_error(error):
                    raise
                time.sleep(2 ** attempt)
        raise last_error or RuntimeError("Gemini guidance generation failed.")
    finally:
        client.close()
