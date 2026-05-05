# MCP server — wraps our trained Random Forest model so Claude Desktop can use it
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")

# threshold tuned for best F1 during training
THRESHOLD = 0.603

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
    weather_str = str(params.get("weather_condition", "other")).lower()
    weather_enc = WEATHER_MAP.get(weather_str, 6)

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

    # also return the raw values we need for risk factor checks
    raw = {
        "temperature_f": temp,
        "visibility_mi": visibility,
        "wind_speed_mph": wind_speed,
        "hour": hour,
        "is_night": is_night,
        "weather_condition": weather_str,
        "junction": get_bool("junction"),
        "crossing": get_bool("crossing"),
    }

    return pd.DataFrame([features])[FEATURE_COLS], raw


def get_risk_factors(raw):
    # flag conditions we know from EDA are associated with higher severity
    factors = []
    if raw["visibility_mi"] < 3.0:
        factors.append(f"Low visibility ({raw['visibility_mi']} mi)")
    if raw["wind_speed_mph"] > 20:
        factors.append(f"High wind speed ({raw['wind_speed_mph']} mph)")
    if raw["temperature_f"] < 20:
        factors.append(f"Freezing temperature ({raw['temperature_f']}°F)")
    elif raw["temperature_f"] > 100:
        factors.append(f"Extreme heat ({raw['temperature_f']}°F)")
    if raw["is_night"]:
        factors.append("Nighttime conditions")
    if raw["weather_condition"] in ("snow", "fog", "thunderstorm"):
        factors.append(f"Hazardous weather ({raw['weather_condition']})")
    if raw["hour"] in (0, 1, 2, 3, 4):
        factors.append("Late night / early morning hour")
    if raw["junction"]:
        factors.append("Junction present")
    if raw["crossing"]:
        factors.append("Pedestrian crossing present")
    return factors


def get_risk_level(prob):
    if prob < 0.4:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Very High"


def run_prediction(params):
    df, raw = parse_inputs(params)
    prob = float(model.predict_proba(df)[0][1])
    severe = prob >= THRESHOLD
    level = get_risk_level(prob)
    factors = get_risk_factors(raw)

    label = "Severe" if severe else "Minor"
    if factors:
        summary = f"{label} accident predicted ({level} risk, {prob:.1%} probability). Key risk factors: {', '.join(factors)}."
    else:
        summary = f"{label} accident predicted ({level} risk, {prob:.1%} probability). No major risk factors flagged."

    return {
        "prediction": label,
        "probability": round(prob, 4),
        "risk_level": level,
        "threshold": THRESHOLD,
        "risk_factors": factors,
        "summary": summary,
    }


server = Server("accident-severity")


@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="predict_severity",
            description=(
                "Predict whether a US road accident will be severe (severity 3-4) or minor (1-2). "
                "Returns the prediction, probability, risk level, flagged risk factors, and a plain-English summary. "
                "Required: temperature_f, visibility_mi, wind_speed_mph, hour, day_of_week, month, lat, lng. "
                "weather_condition options: clear, cloudy, rain, snow, fog, thunderstorm."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "temperature_f":     {"type": "number", "description": "Temperature in Fahrenheit"},
                    "visibility_mi":     {"type": "number", "description": "Visibility in miles"},
                    "wind_speed_mph":    {"type": "number", "description": "Wind speed in mph"},
                    "hour":              {"type": "integer", "minimum": 0, "maximum": 23},
                    "day_of_week":       {"type": "integer", "minimum": 0, "maximum": 6, "description": "0=Monday, 6=Sunday"},
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
                "required": ["temperature_f", "visibility_mi", "wind_speed_mph", "hour", "day_of_week", "month", "lat", "lng"],
            },
        ),
        types.Tool(
            name="compare_scenarios",
            description=(
                "Compare two accident scenarios side by side and show which is higher risk. "
                "Good for questions like: how much more dangerous is driving at night in a storm vs a clear afternoon?"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "label_a": {"type": "string", "description": "Name for scenario A"},
                    "label_b": {"type": "string", "description": "Name for scenario B"},
                    "scenario_a": {
                        "type": "object",
                        "description": "First scenario — same fields as predict_severity",
                        "properties": {
                            "temperature_f": {"type": "number"},
                            "visibility_mi": {"type": "number"},
                            "wind_speed_mph": {"type": "number"},
                            "hour": {"type": "integer"},
                            "day_of_week": {"type": "integer"},
                            "month": {"type": "integer"},
                            "lat": {"type": "number"},
                            "lng": {"type": "number"},
                            "distance_mi": {"type": "number"},
                            "duration_min": {"type": "number"},
                            "is_night": {"type": "boolean"},
                            "weather_condition": {"type": "string"},
                            "junction": {"type": "boolean"},
                            "crossing": {"type": "boolean"},
                            "traffic_signal": {"type": "boolean"},
                        },
                        "required": ["temperature_f", "visibility_mi", "wind_speed_mph", "hour", "day_of_week", "month", "lat", "lng"],
                    },
                    "scenario_b": {
                        "type": "object",
                        "description": "Second scenario — same fields as predict_severity",
                        "properties": {
                            "temperature_f": {"type": "number"},
                            "visibility_mi": {"type": "number"},
                            "wind_speed_mph": {"type": "number"},
                            "hour": {"type": "integer"},
                            "day_of_week": {"type": "integer"},
                            "month": {"type": "integer"},
                            "lat": {"type": "number"},
                            "lng": {"type": "number"},
                            "distance_mi": {"type": "number"},
                            "duration_min": {"type": "number"},
                            "is_night": {"type": "boolean"},
                            "weather_condition": {"type": "string"},
                            "junction": {"type": "boolean"},
                            "crossing": {"type": "boolean"},
                            "traffic_signal": {"type": "boolean"},
                        },
                        "required": ["temperature_f", "visibility_mi", "wind_speed_mph", "hour", "day_of_week", "month", "lat", "lng"],
                    },
                },
                "required": ["scenario_a", "scenario_b"],
            },
        ),
        types.Tool(
            name="get_model_insights",
            description=(
                "Return what the model learned — top feature importances, high-risk conditions, "
                "class distribution, and how Phase 3 Spark MLlib compared to Phase 2 sklearn."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name, arguments):
    if name == "predict_severity":
        try:
            result = run_prediction(arguments)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except ValueError as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    elif name == "compare_scenarios":
        try:
            label_a = arguments.get("label_a", "Scenario A")
            label_b = arguments.get("label_b", "Scenario B")
            result_a = run_prediction(arguments["scenario_a"])
            result_b = run_prediction(arguments["scenario_b"])

            diff = result_b["probability"] - result_a["probability"]
            higher = label_b if diff > 0 else label_a
            lower = label_a if diff > 0 else label_b

            return [types.TextContent(type="text", text=json.dumps({
                label_a: result_a,
                label_b: result_b,
                "comparison": {
                    "higher_risk": higher,
                    "probability_difference": round(abs(diff), 4),
                    "summary": f"{higher} has a {abs(diff):.1%} higher severe accident probability than {lower}.",
                },
            }, indent=2))]
        except (ValueError, KeyError) as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    elif name == "get_model_insights":
        insights = {
            "model": "Random Forest (scikit-learn, Phase 2)",
            "training_data": "US Accidents March 2023, ~2.8M records",
            "task": "Binary classification — Minor (Severity 1-2) vs Severe (Severity 3-4)",
            "class_distribution": {
                "Severity 1": "~1%",
                "Severity 2": "~80%",
                "Severity 3": "~16%",
                "Severity 4": "~3%",
                "note": "Heavy imbalance handled with SMOTE + class weighting",
            },
            "optimal_threshold": THRESHOLD,
            "top_features": [
                {"feature": "Distance(mi)", "importance": "32%", "note": "Longer road impact correlates with higher severity"},
                {"feature": "Start_Lat / Start_Lng", "importance": "~12% each", "note": "Location is a strong predictor"},
                {"feature": "Duration_min", "importance": "8%", "note": "Longer accidents tend to be more severe"},
                {"feature": "Temperature(F)", "importance": "6%", "note": "Extreme temps increase risk"},
                {"feature": "Visibility(mi)", "importance": "5%", "note": "Low visibility raises severity probability"},
            ],
            "high_risk_conditions": [
                "Visibility < 3 miles",
                "Wind speed > 20 mph",
                "Nighttime driving",
                "Snow, fog, or thunderstorm",
                "Late night hours (12AM-4AM)",
                "Junctions and crossings",
            ],
            "phase2_vs_phase3": {
                "Phase 2 sklearn RF (no HPMS)": "F1 = 0.59",
                "Phase 3 Spark MLlib RF (with HPMS)": "F1 = 0.84",
                "note": "Adding road infrastructure features (AADT, speed limit, lane count) gave a big boost",
            },
        }
        return [types.TextContent(type="text", text=json.dumps(insights, indent=2))]

    else:
        raise ValueError(f"unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
