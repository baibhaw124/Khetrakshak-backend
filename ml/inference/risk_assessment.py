# ============================================================
# KHETRAKSHAK AI
# RISK ASSESSMENT
# ============================================================

def assess_risk(confidence, disease):

    confidence = float(confidence)

    # Healthy crop
    if disease.lower() in [
        "healthy",
        "healthy leaf",
        "healthy plant"
    ]:
        return {
            "risk": "Low",
            "risk_score": 10,
            "message": "The leaf appears healthy."
        }

    # Disease confidence based risk
    if confidence >= 90:
        risk = "High"
        risk_score = 90
        message = "High confidence disease detection. Immediate attention recommended."

    elif confidence >= 70:
        risk = "Medium"
        risk_score = 70
        message = "Moderate disease risk detected. Monitoring and treatment recommended."

    else:
        risk = "Low"
        risk_score = 40
        message = "Low confidence disease detection. Further observation recommended."

    return {
        "risk": risk,
        "risk_score": risk_score,
        "message": message
    }