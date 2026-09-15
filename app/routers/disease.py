from pathlib import Path

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
async def analyze_disease(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # 1. BASIC FILE VALIDATION
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No image file provided."
        )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )


    # --------------------------------------------------------
    # 2. SAVE TEMPORARY IMAGE
    # --------------------------------------------------------

    temp_dir = Path("ml/temp")

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        temp_dir /
        file.filename
    )

    image_data = await file.read()

    with open(
        image_path,
        "wb"
    ) as image_file:

        image_file.write(
            image_data
        )


    try:

        # ----------------------------------------------------
        # 3. LEAF VALIDATION
        # ----------------------------------------------------

        validation = validate_leaf(
            image_path
        )


        # ----------------------------------------------------
        # 4. REJECT NON-LEAF
        # ----------------------------------------------------

        if not validation["is_leaf"]:

            return {
                "status": "rejected",
                "reason": "non_leaf",
                "classification": "non_leaf",
                "confidence": validation["confidence"],
                "message": "Please upload a clear image of a plant leaf."
            }


        # ----------------------------------------------------
        # 5. DISEASE PREDICTION
        # ----------------------------------------------------

        predictions = predict_image(
            image_path,
            top_k=1
        )

        prediction = predictions[0]


        # ----------------------------------------------------
        # 6. EXTRACT RESULT
        # ----------------------------------------------------

        crop = prediction["crop"]

        disease = prediction["disease"]

        confidence = prediction["confidence"]


        # ----------------------------------------------------
        # 7. RISK ASSESSMENT
        # ----------------------------------------------------

        risk_result = assess_risk(
            confidence,
            disease
        )
        treatment = get_treatment(
            crop,
            disease
        )

        # ----------------------------------------------------
        # 8. FINAL RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "classification": "leaf",

            "crop": crop,

            "disease": disease,

            "confidence": round(
                confidence,
                2
            ),

            "risk": risk_result["risk"],

            "risk_score": risk_result["risk_score"],

            "message": risk_result["message"],

            "severity": treatment["severity"],
            "symptoms": treatment["symptoms"],

            "recommendations": treatment["recommendations"],

            "prevention": treatment["prevention"],
        }

    finally:

        # ----------------------------------------------------
        # 9. DELETE TEMPORARY IMAGE
        # ----------------------------------------------------

        if image_path.exists():

            image_path.unlink() 