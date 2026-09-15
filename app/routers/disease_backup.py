from pathlib import Path
import tempfile

from fastapi import APIRouter, File, UploadFile, HTTPException

from ml.inference.predictor import predict_image


router = APIRouter(
    prefix="/disease",
    tags=["Disease Detection"]
)


@router.post("/analyze")
async def analyze_disease(
    file: UploadFile = File(...)
):
    """
    Analyze an uploaded plant leaf image
    using the trained Khetrakshak EfficientNetB0 model.
    """

    # --------------------------------------------------------
    # Validate file
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
            detail="Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded image temporarily
        # ----------------------------------------------------

        file_bytes = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(file_bytes)

            temp_path = Path(
                temp_file.name
            )
        # ----------------------------------------------------
        # REAL ML PREDICTION
        # ----------------------------------------------------

        predictions = predict_image(
            temp_path,
            top_k=3
        )

        if not predictions:
            raise HTTPException(
                status_code=500,
                detail="Model did not return a prediction."
            )

        # Best prediction
        best_prediction = predictions[0]

        # ----------------------------------------------------
        # Basic risk calculation
        # ----------------------------------------------------

        confidence = best_prediction[
            "confidence"
        ]

        if confidence >= 80:
            risk = "High"

        elif confidence >= 50:
            risk = "Medium"

        else:
            risk = "Low"

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "status": "success",
            "filename": file.filename,

            "crop": best_prediction["crop"],

            "disease": best_prediction["disease"],

            "confidence": round(
                confidence,
                2
            ),

            "risk": risk,

            "top_predictions": predictions,

            "message": "Real prediction generated successfully."
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            f"Prediction error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the uploaded image."
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary image
        # ----------------------------------------------------

        if temp_path and temp_path.exists():

            temp_path.unlink()