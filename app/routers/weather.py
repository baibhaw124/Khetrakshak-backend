from fastapi import APIRouter, HTTPException, Query
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import json


router = APIRouter(
    prefix="/weather",
    tags=["Weather"]
)


@router.get("/current")
def get_current_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180)
):
    """
    Get current weather conditions for a geographic location.
    """

    params = urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
        ),
        "timezone": "auto",
    })

    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

    except HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Weather service returned HTTP {error.code}."
        )

    except URLError:
        raise HTTPException(
            status_code=502,
            detail="Unable to connect to weather service."
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve weather data."
        )

    current = data.get("current")

    if not current:
        raise HTTPException(
            status_code=502,
            detail="Weather service returned no current weather data."
        )

    return {
        "status": "success",

        "location": {
            "latitude": latitude,
            "longitude": longitude
        },

        "weather": {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "rain": current.get("rain"),
            "wind_speed": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "time": current.get("time")
        },

        "source": "Open-Meteo"
    }