import logging
import asyncio

from app.services.gemini_guidance import generate_treatment_guidance
from app.services.product_links import enrich_product_suggestions
from pathlib import Path
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException

from ml.inference.leaf_validator import validate_leaf
from ml.inference.predictor import predict_image
from ml.inference.risk_assessment import assess_risk
from ml.inference.treatment_engine import get_treatment

router = APIRouter(
    prefix="/disease",
    tags=["Disease Detection"]
)
# ============================================================
# KHETRAKSHAK AI
# DISEASE ANALYSIS
# ============================================================

@router.post("/analyze")
async def analyze_disease(file: UploadFile = File(...)):
    """Analyze a plant-leaf image without retaining the uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    extension = Path(file.filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    # Read at most one byte over the limit so oversized uploads are rejected
    # before they are written to disk. Keep the existing 10 MB UI limit aligned.
    max_upload_bytes = 10 * 1024 * 1024
    image_data = await file.read(max_upload_bytes + 1)
    if not image_data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_data) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    # Never use the client-provided filename as a filesystem path. A random
    # server-generated name avoids path traversal and concurrent filename clashes.
    temp_dir = Path("ml/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    image_path = temp_dir / f"{uuid.uuid4().hex}{extension}"

    try:
        image_path.write_bytes(image_data)
        validation = validate_leaf(image_path)

        if not validation["is_leaf"]:
            return {
                "status": "rejected",
                "reason": "non_leaf",
                "classification": "non_leaf",
                "confidence": validation["confidence"],
                "message": "Please upload a clear image of a plant leaf.",
            }

        predictions = predict_image(image_path, top_k=1)
        prediction = predictions[0]
        crop = prediction["crop"]
        disease = prediction["disease"]
        confidence = prediction["confidence"]
        risk_result = assess_risk(confidence, disease)
        treatment = get_treatment(crop, disease)

        
        # Preserve the existing local treatment guidance as the fallback.
        treatment_steps = treatment["recommendations"]
        ai_prevention = treatment["prevention"]
        product_suggestions = []
        ai_caution = ""
        ai_guidance_status = "not_needed"

        # Healthy predictions should not trigger unnecessary treatment.
        normalized_disease = str(disease).strip().lower()
        is_healthy = normalized_disease in {
            "healthy",
            "healthy leaf",
            "healthy plant",
        }

        if not is_healthy:
            ai_guidance_status = "fallback"

            try:
                guidance = await asyncio.to_thread(
                    generate_treatment_guidance,
                    crop=crop,
                    disease=disease,
                    confidence=confidence,
                    severity=treatment["severity"],
                    symptoms=treatment["symptoms"],
                    recommendations=treatment["recommendations"],
                    prevention=treatment["prevention"],
                )

                treatment_steps = guidance["treatment_steps"]
                ai_prevention = guidance["prevention"]
                product_suggestions = guidance["product_suggestions"]
                ai_caution = guidance["caution"]
                ai_guidance_status = "available"

            
            except Exception:
                logging.exception("Gemini treatment guidance failed")

                # Detection remains successful if Gemini is unavailable.
                ai_caution = (
                    "AI treatment guidance is temporarily unavailable. "
                    "Use the existing recommendations and consult a local "
                    "agricultural expert before applying crop-protection products."
                )


        return {
            "status": "success",
            "classification": "leaf",
            "crop": crop,
            "disease": disease,
            "confidence": round(confidence, 2),
            "risk": risk_result["risk"],
            "risk_score": risk_result["risk_score"],
            "message": risk_result["message"],
            "severity": treatment["severity"],
            "symptoms": treatment["symptoms"],
            "recommendations": treatment["recommendations"],
            "prevention": treatment["prevention"],
            "treatment_steps": treatment_steps,
            "ai_prevention": ai_prevention,
            "product_suggestions": enrich_product_suggestions(product_suggestions),
            "ai_caution": ai_caution,
            "ai_guidance_status": ai_guidance_status,
        }

    finally:
        if image_path.exists():
            image_path.unlink()
