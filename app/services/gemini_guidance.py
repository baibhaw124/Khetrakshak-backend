import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Explicitly load .env from the backend root directory.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_FILE)

MODEL_NAME = "gemini-3.8-flash"

def generate_treatment_guidance(
    crop: str,
    disease: str,
    confidence: float,
    severity: str,
    symptoms: list,
    recommendations: list,
    prevention: list,
) -> dict:
    """
    Generate additional crop-treatment guidance using Gemini.
    The caller should handle API errors and retain existing guidance.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=15000),
    )

    context = {
        "crop": crop,
        "detected_disease": disease,
        "model_confidence_percent": round(confidence * 100, 2)
        if confidence <= 1
        else round(confidence, 2),
        "severity": severity,
        "known_symptoms": symptoms,
        "existing_recommendations": recommendations,
        "existing_prevention": prevention,
    }

    prompt = f"""
You are an agricultural crop-care assistant helping farmers in India.

Use the supplied crop-disease assessment to provide practical,
cautious treatment and prevention guidance.

Assessment:
{json.dumps(context, ensure_ascii=False)}

Return ONLY a valid JSON object with these keys:
- treatment_steps: array of strings
- prevention: array of strings
- product_suggestions: array of objects, each containing
  category, purpose, and safety_note
- caution: string

Rules:
1. Treat the detected disease as a prediction, not a confirmed diagnosis.
2. Respect the supplied recommendations and prevention information.
3. Give clear, practical steps in simple language.
4. Suggest product CATEGORIES, not invented brands, prices, or shopping links.
5. Do not invent pesticide doses or chemical application rates.
6. Recommend only products appropriate to the crop and suspected disease.
7. Tell the farmer to follow the product label and local agricultural
   expert guidance before applying any pesticide or fungicide.
8. If evidence is uncertain, recommend confirming the diagnosis first.
9. Do not recommend unnecessary chemical treatment.
10. Keep each array concise and useful for a farmer.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=1200,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        result = json.loads(response.text)

        required_keys = {
            "treatment_steps",
            "prevention",
            "product_suggestions",
            "caution",
        }
        if not isinstance(result, dict) or not required_keys.issubset(result):
            raise ValueError("Gemini returned an unexpected response structure.")

        if not all(
            isinstance(result[key], list)
            for key in (
                "treatment_steps",
                "prevention",
                "product_suggestions",
            )
        ):
            raise ValueError("Gemini returned invalid list fields.")

        if not isinstance(result["caution"], str):
            raise ValueError("Gemini returned an invalid caution field.")

        for item in result["product_suggestions"]:
            if not isinstance(item, dict) or not all(
                isinstance(item.get(key), str)
                for key in ("category", "purpose", "safety_note")
            ):
                raise ValueError("Gemini returned an invalid product suggestion.")

        return result

    finally:
        client.close()
