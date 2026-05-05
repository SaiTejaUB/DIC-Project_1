# Accident Severity MCP Server

Wraps the trained Random Forest model as an MCP server so Claude Desktop can predict accident severity, compare scenarios, and explain model insights — all in natural language.

## Setup

1. Train the model first (or use the existing pkl):
   ```
   python src/models/train_random_forest.py
   ```
   This saves `models/random_forest.pkl`.

2. Install dependencies:
   ```
   pip install mcp pandas numpy scikit-learn imbalanced-learn
   ```

3. Add to Claude Desktop's `claude_desktop_config.json`:
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

4. Restart Claude Desktop. The server prints `model loaded` on startup.

---

## Tools

### `predict_severity`
Predicts Minor vs Severe for a single accident scenario.

Returns: prediction, probability, risk level (Low / Medium / High / Very High), flagged risk factors, and a plain-English summary.

**Required inputs:** `temperature_f`, `visibility_mi`, `wind_speed_mph`, `hour`, `day_of_week`, `month`, `lat`, `lng`

**Optional inputs:** `distance_mi`, `duration_min`, `is_night`, `weather_condition` (clear / cloudy / rain / snow / fog / thunderstorm), and road feature flags: `junction`, `crossing`, `traffic_signal`, `amenity`, `bump`, `give_way`, `no_exit`, `railway`, `roundabout`, `station`, `stop`, `traffic_calming`, `turning_loop`

---

### `compare_scenarios`
Compares two accident scenarios side by side. Returns predictions for both and highlights which is higher risk and by how much.

**Required inputs:** `scenario_a`, `scenario_b` (each uses the same fields as `predict_severity`), optional `label_a` and `label_b` for naming.

---

### `get_model_insights`
Returns key findings from the model: top feature importances, known high-risk conditions, class distribution, and a Phase 2 vs Phase 3 performance comparison.

No inputs required.

---

## Demo Prompts

**Single prediction:**
> "Predict the severity of an accident at 2am on a Saturday in January, temperature 28°F, visibility 0.5 miles, wind 25mph, foggy weather, at a junction in Buffalo NY."

**Comparison:**
> "Compare a clear Monday afternoon commute at 5pm, 65°F, good visibility — versus a foggy Tuesday night at 2am, 30°F, visibility 0.5 miles. Which is more dangerous and by how much?"

**Model insights:**
> "What did the model learn about accident severity? What are the most important features and what conditions are highest risk?"

**Stakeholder story:**
> "I'm a highway safety engineer in California. What conditions should I prioritize for road improvements based on the model's findings?"
