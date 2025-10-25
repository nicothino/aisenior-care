# AI Senior Care — Agentic ML Stack (Starter)

This repo is a **from-scratch** scaffold for a multi-agent elderly-care assistant with real ML models.

## Services
- `services/nlp_intent`: fine-tune and serve an **intent classifier** (reminder_medication, appointment, mood, symptom_alert, smalltalk).
- `services/nlp_emotion`: fine-tune and serve an **emotion classifier** (happy, sad, anxious, lonely, neutral).
- `services/gateway`: a **FastAPI** gateway that orchestrates models and exposes endpoints for your PWA/mobile UI.

## Quickstart (Colab)
```bash
pip install -r requirements.txt

# Train tiny demo models (you will replace with your data later)
make train-intent
make train-emotion

# Serve gateway API (uses saved models)
make serve-gateway
```

Open Swagger at `http://localhost:8000/docs`.
