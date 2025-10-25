from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionModel:
    def __init__(self, path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.id2label = self.model.config.id2label

    def infer(self, text:str):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=96)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        scores = {self.id2label[i]: float(p) for i,p in enumerate(probs)}
        emotion = max(scores, key=scores.get)
        return emotion, scores
