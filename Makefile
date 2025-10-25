PY=python

INTENT_OUT=services/nlp_intent/out
EMOTION_OUT=services/nlp_emotion/out

train-intent:
	$(PY) services/nlp_intent/train_intent.py --epochs 1 --out $(INTENT_OUT)

train-emotion:
	$(PY) services/nlp_emotion/train_emotion.py --epochs 1 --out $(EMOTION_OUT)

serve-gateway:
	$(PY) services/gateway/app.py
