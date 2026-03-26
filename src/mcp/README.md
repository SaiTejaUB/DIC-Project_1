# Accident Severity MCP Server

Wraps the trained Random Forest model as an MCP server so Claude Desktop can call it directly.

## Dependencies

```
pip install mcp pandas numpy scikit-learn imbalanced-learn
```

## Setup

1. Train the model first by running `python src/models/train_random_forest.py` — this saves `models/random_forest.pkl`
2. Add the server to Claude Desktop's config (see below)
3. Run Claude Desktop and ask it to predict accident severity

## Claude Desktop Config

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "accident-severity": {
      "command": "python",
      "args": ["C:/path/to/DIC-Project_1/src/mcp/server.py"]
    }
  }
}
```

## Example Usage

Once connected, ask Claude Desktop:

> "Predict severity for an accident at 7am on a Tuesday in January, temperature 32F, visibility 1 mile, wind 15mph, at a junction, rainy weather"

Claude will call `predict_severity` with the appropriate values and return something like:

```json
{
  "prediction": "Severe",
  "probability": 0.681,
  "threshold": 0.603
}
```

## Tool Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| temperature | float | yes | Temperature in Fahrenheit |
| visibility | float | yes | Visibility in miles |
| wind_speed | float | yes | Wind speed in mph |
| distance | float | yes | Length of road affected (miles) |
| hour | int | yes | Hour of day (0-23) |
| day_of_week | int | yes | 0=Monday, 6=Sunday |
| month | int | yes | 1-12 |
| duration_min | float | yes | Accident duration in minutes |
| is_night | int | no | 1=night, 0=day (default 0) |
| weather_enc | int | no | 0=Clear, 1=Cloudy, 2=Rain, 3=Snow, 4=Fog, 5=Thunderstorm (default 0) |
| lat | float | no | Latitude (default 39.0) |
| lng | float | no | Longitude (default -98.0) |
| junction | int | no | 1 if at a junction (default 0) |
| crossing | int | no | 1 if at a crossing (default 0) |
| traffic_signal | int | no | 1 if traffic signal present (default 0) |
