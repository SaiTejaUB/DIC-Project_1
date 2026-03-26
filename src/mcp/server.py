# MCP server for accident severity prediction
# wraps the trained random forest pipeline so Claude Desktop can call it
# run with: python src/mcp/server.py

import os
import sys
import pickle
import json
import asyncio
import pandas as pd

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")

THRESHOLD = 0.603  # best threshold found during training

FEATURE_COLS = [
    "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)", "Distance(mi)",
    "Hour", "DayOfWeek", "Month", "Duration_min",
    "Sunrise_Sunset_enc", "Weather_enc",
    "Start_Lat", "Start_Lng",
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit",
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming",
    "Traffic_Signal", "Turning_Loop",
]

WEATHER_MAP = {
    "clear": 0, "cloudy": 1, "rain": 2, "rain_drizzle": 2,
    "snow": 3, "fog": 4, "thunderstorm": 5, "other": 6,
}

if not os.path.exists(MODEL_PATH):
    print(f"model not found at {MODEL_PATH} - run train_random_forest.py first", file=sys.stderr)
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("model loaded", file=sys.stderr)


def parse_inputs(params):
    try:
        temp = float(params["temperature_f"])
        visibility = float(params["visibility_mi"])
        wind_speed = float(params["wind_speed_mph"])
        hour = int(params["hour"])
        dow = int(params["day_of_week"])
        month = int(params["month"])
        lat = float(params["lat"])
        lng = float(params["lng"])
    except KeyError as e:
        raise ValueError(f"missing required field: {e}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid input: {e}")

    if not (0 <= hour <= 23):
        raise ValueError("hour must be 0-23")
    if not (1 <= month <= 12):
        raise ValueError("month must be 1-12")
    if not (0 <= dow <= 6):
        raise ValueError("day_of_week must be 0-6")

    distance = float(params.get("distance_mi", 0.01))
    duration = float(params.get("duration_min", 30))
    is_night = int(bool(params.get("is_night", False)))
    weather_enc = WEATHER_MAP.get(str(params.get("weather_condition", "other")).lower(), 6)

    def get_bool(key):
        val = params.get(key, False)
        if isinstance(val, str):
            return 1 if val.lower() in ("true", "1", "yes") else 0
        return int(bool(val))

    features = {
        "Temperature(F)": temp,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind_speed,
        "Distance(mi)": distance,
        "Hour": hour,
        "DayOfWeek": dow,
        "Month": month,
        "Duration_min": duration,
        "Sunrise_Sunset_enc": is_night,
        "Weather_enc": weather_enc,
        "Start_Lat": lat,
        "Start_Lng": lng,
        "Amenity": get_bool("amenity"),
        "Bump": get_bool("bump"),
        "Crossing": get_bool("crossing"),
        "Give_Way": get_bool("give_way"),
        "Junction": get_bool("junction"),
        "No_Exit": get_bool("no_exit"),
        "Railway": get_bool("railway"),
        "Roundabout": get_bool("roundabout"),
        "Station": get_bool("station"),
        "Stop": get_bool("stop"),
        "Traffic_Calming": get_bool("traffic_calming"),
        "Traffic_Signal": get_bool("traffic_signal"),
        "Turning_Loop": get_bool("turning_loop"),
    }
    return pd.DataFrame([features])[FEATURE_COLS]


def run_prediction(params):
    df = parse_inputs(params)
    prob = float(model.predict_proba(df)[0][1])
    severe = prob >= THRESHOLD
    return {
        "prediction": "Severe" if severe else "Minor",
        "probability": round(prob, 4),
        "threshold": THRESHOLD,
    }


server = Server("accident-severity")


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="predict_severity",
            description=(
                "Predict whether a US road accident will be severe (severity 3-4) or minor (1-2). "
                "Required: temperature_f, visibility_mi, wind_speed_mph, hour, day_of_week, month, lat, lng. "
                "weather_condition options: clear, cloudy, rain, snow, fog, thunderstorm."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "temperature_f":     {"type": "number"},
                    "visibility_mi":     {"type": "number"},
                    "wind_speed_mph":    {"type": "number"},
                    "hour":              {"type": "integer", "minimum": 0, "maximum": 23},
                    "day_of_week":       {"type": "integer", "minimum": 0, "maximum": 6},
                    "month":             {"type": "integer", "minimum": 1, "maximum": 12},
                    "lat":               {"type": "number"},
                    "lng":               {"type": "number"},
                    "distance_mi":       {"type": "number"},
                    "duration_min":      {"type": "number"},
                    "is_night":          {"type": "boolean"},
                    "weather_condition": {"type": "string"},
                    "amenity":           {"type": "boolean"},
                    "bump":              {"type": "boolean"},
                    "crossing":          {"type": "boolean"},
                    "give_way":          {"type": "boolean"},
                    "junction":          {"type": "boolean"},
                    "no_exit":           {"type": "boolean"},
                    "railway":           {"type": "boolean"},
                    "roundabout":        {"type": "boolean"},
                    "station":           {"type": "boolean"},
                    "stop":              {"type": "boolean"},
                    "traffic_calming":   {"type": "boolean"},
                    "traffic_signal":    {"type": "boolean"},
                    "turning_loop":      {"type": "boolean"},
                },
                "required": [
                    "temperature_f", "visibility_mi", "wind_speed_mph",
                    "hour", "day_of_week", "month", "lat", "lng",
                ],
            },
        )
    ]


@server.call_tool()
async def call_tool(name, arguments):
    if name != "predict_severity":
        raise ValueError(f"unknown tool: {name}")
    try:
        result = run_prediction(arguments)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except ValueError as e:
        return [types.TextContent(type="text", text=f"Error: {e}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
