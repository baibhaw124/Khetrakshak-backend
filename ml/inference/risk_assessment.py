# ============================================================
# KHETRAKSHAK AI
# WEATHER-AWARE RISK ASSESSMENT
# ============================================================


def assess_risk(confidence, disease, weather=None):

    confidence = float(confidence)

    # --------------------------------------------------------
    # 1. HEALTHY CROP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 2. BASE RISK FROM AI CONFIDENCE
    # --------------------------------------------------------

    if confidence >= 90:

        risk_score = 90

    elif confidence >= 70:

        risk_score = 70

    else:

        risk_score = 40


    # --------------------------------------------------------
    # 3. WEATHER-BASED RISK MODIFIERS
    # --------------------------------------------------------

    weather_adjustment = 0

    if weather:

        temperature = weather.get("temperature")
        humidity = weather.get("humidity")
        rain = weather.get("rain")
        precipitation = weather.get("precipitation")

        # --------------------------------------------
        # High humidity can support many fungal diseases
        # --------------------------------------------

        if humidity is not None:

            humidity = float(humidity)

            if humidity >= 85:

                weather_adjustment += 10

            elif humidity >= 70:

                weather_adjustment += 5


        # --------------------------------------------
        # Rain / precipitation increases leaf wetness
        # --------------------------------------------

        if rain is not None:

            rain = float(rain)

            if rain > 5:

                weather_adjustment += 10

            elif rain > 0:

                weather_adjustment += 5

        elif precipitation is not None:

            precipitation = float(precipitation)

            if precipitation > 5:

                weather_adjustment += 10

            elif precipitation > 0:

                weather_adjustment += 5


        # --------------------------------------------
        # Temperature range
        # --------------------------------------------

        if temperature is not None:

            temperature = float(temperature)

            if 20 <= temperature <= 30:

                weather_adjustment += 5


    # --------------------------------------------------------
    # 4. FINAL RISK SCORE
    # --------------------------------------------------------

    risk_score += weather_adjustment

    # Maximum score = 100

    risk_score = min(
        risk_score,
        100
    )


    # --------------------------------------------------------
    # 5. FINAL RISK LEVEL
    # --------------------------------------------------------

    if risk_score >= 80:

        risk = "High"

        message = (
            "High disease risk detected. "
            "Current weather conditions may support disease development. "
            "Immediate attention recommended."
        )

    elif risk_score >= 50:

        risk = "Medium"

        message = (
            "Moderate disease risk detected. "
            "Current weather conditions may increase disease pressure. "
            "Monitoring and treatment recommended."
        )

    else:

        risk = "Low"

        message = (
            "Low disease risk detected. "
            "Continue monitoring crop health."
        )


    # --------------------------------------------------------
    # 6. RETURN RESULT
    # --------------------------------------------------------

    return {

        "risk": risk,

        "risk_score": risk_score,

        "message": message,

        "weather_adjustment": weather_adjustment
    }