# CarSense AI — Autonomous Car-Buying Decision Agent (India)

CarSense AI is an academic decision-support prototype. It combines a goal-based / utility-based recommendation agent with a learning-from-feedback loop.

**Agent cycle:** Perceive → Estimate → Filter / Re-plan → Score → Decide → Explain → Learn.

## Project files

| File | Purpose |
|---|---|
| `data.py` | Catalogue of 61 illustrative Indian-market car models and approximate attributes |
| `agent.py` | Buyer-profile parsing, TCO estimation, Random Forest resale model, TOPSIS, cosine similarity, simulated SVD peer model, hybrid ranking, explanations and feedback updates |
| `chatbot.py` | Interactive terminal chatbot that calls the same decision engine |
| `evaluate.py` | Eight buyer personas, baseline comparisons, ablation study and ranking / regression metrics |
| `app.py` | Streamlit dashboard with recommendations, reasoning, TCO, what-if analysis, comparison and evaluation |
| `test_agent.py` | Smoke tests for parsing, validation and agent execution |
| `requirements.txt` | Python dependencies |

## Run on Windows (VS Code / PowerShell)

Open the folder containing these files in VS Code. Open **Terminal → New Terminal**, then run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Step 1 — test the core code

```powershell
python -m unittest test_agent.py
```

### Step 2 — run the conversational agent

```powershell
python chatbot.py
```

Example request:

`family of 5, budget 12 lakh, 15000 km per year, mostly city, safe SUV, good resale`

The chatbot accepts follow-ups such as `increase my budget to 15 lakh`, `prefer diesel`, `like 1`, and `dislike 2`. It is rule-based and runs offline; no LLM API key is required.

### Step 3 — run the evaluation

```powershell
python evaluate.py
```

The evaluation writes CSV files and charts to the `results` folder.

### Step 4 — launch the dashboard

```powershell
streamlit run app.py
```

Streamlit normally opens the dashboard at `http://localhost:8501`.

## Methodology

- **Hard constraints:** on-road budget, minimum seats, minimum safety rating, allowed fuel and optional body type.
- **TCO:** estimated depreciation + fuel / charging + servicing + insurance over the selected ownership period.
- **Resale model:** Random Forest regression predicts retained resale percentage.
- **Content recommendation:** cosine similarity between the buyer's ideal profile and catalogue features.
- **Multi-criteria ranking:** TOPSIS with buyer-priority and entropy-derived weights.
- **Collaborative component:** SVD matrix completion over simulated user ratings, followed by nearest-archetype matching.
- **Hybrid decision:** weighted combination of TOPSIS, content similarity and peer scores.
- **Learning:** likes / dislikes update buyer-priority weights and trigger a new ranking.
- **Robustness:** priority weights are perturbed by ±20% to estimate top-three stability.

## Important academic limitations

1. The catalogue currently contains **61 models**, not 1,000+ live variants.
2. Vehicle prices, specifications, fuel rates, service costs and insurance are approximate academic assumptions. They are not live quotations and should be checked against current official sources.
3. The resale model is trained on **synthetically generated targets**. Its MAE / RMSE / R² describe performance against that synthetic simulation, not real resale transactions.
4. The SVD collaborative component uses simulated users and ratings, not real customer behaviour.
5. Maintenance and insurance are estimated by formulas; they are not trained ML prediction models.
6. The terminal chatbot uses rule-based natural-language parsing. It is not an LLM, does not use RAG and does not call Anthropic / OpenAI APIs.
7. The displayed recommendation score is a **relative score within the current feasible shortlist**, not a calibrated probability that a buyer will like the car.

This is an educational prototype and not financial, safety, or purchase advice. Verify exact variant, on-road quotation, safety rating, warranty, charging access, service costs and local fuel prices before making a real purchase decision.
