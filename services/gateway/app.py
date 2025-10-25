import os, dateparser, pathlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from services.common.schemas import TextIn, OrchestratedOut, IntentOut, EmotionOut
from services.nlp_intent.model import IntentModel
from services.nlp_emotion.model import EmotionModel

# -----------------------
# Modelos y rutas de salida
# -----------------------
INTENT_PATH = os.getenv("INTENT_PATH", "services/nlp_intent/out")
EMOTION_PATH = os.getenv("EMOTION_PATH", "services/nlp_emotion/out")

intent_model = IntentModel(INTENT_PATH)
emotion_model = EmotionModel(EMOTION_PATH)

# -----------------------
# App principal
# -----------------------
app = FastAPI(title="AI Senior Care — Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Montar UI estática SIN pisar las rutas API
# -----------------------
UI_DIR = pathlib.Path(__file__).resolve().parents[2] / "ui"
if UI_DIR.exists():
    # Sirve archivos estáticos en /ui (CSS, JS, etc.)
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    # Devuelve el index.html en la raíz /
    @app.get("/")
    def root():
        return FileResponse(UI_DIR / "index.html")

# -----------------------
# Rutas API
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer/intent", response_model=IntentOut)
def infer_intent(inp: TextIn):
    intent, scores = intent_model.infer(inp.text)
    return {"intent": intent, "scores": scores}

@app.post("/infer/emotion", response_model=EmotionOut)
def infer_emotion(inp: TextIn):
    emotion, scores = emotion_model.infer(inp.text)
    return {"emotion": emotion, "scores": scores}

@app.post("/orchestrate/process-text", response_model=OrchestratedOut)
def process_text(inp: TextIn):
    intent, iscores = intent_model.infer(inp.text)
    emotion, escores = emotion_model.infer(inp.text)

    actions = []
    t = inp.text.lower()

    if intent in ("reminder_medication", "appointment"):
        when = dateparser.parse(t, languages=["es", "en"])
        actions.append({"type": "schedule", "when": when.isoformat() if when else None, "text": inp.text})

    if intent == "symptom_alert":
        sev = "high" if any(k in t for k in ["sangre", "desmayo", "asfixia", "emergencia"]) else "medium"
        actions.append({"type": "medical_alert", "severity": sev})

    if intent == "mood":
        actions.append({"type": "support", "message": "Estoy contigo. ¿Deseas que contacte a tu cuidador?"})

    return {
        "text": inp.text,
        "intent": {"intent": intent, "scores": iscores},
        "emotion": {"emotion": emotion, "scores": escores},
        "actions": actions,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
