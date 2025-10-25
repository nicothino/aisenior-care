from pydantic import BaseModel, Field
from typing import List, Optional

class TextIn(BaseModel):
    text: str = Field(..., min_length=1)

class IntentOut(BaseModel):
    intent: str
    scores: dict

class EmotionOut(BaseModel):
    emotion: str
    scores: dict

class OrchestratedOut(BaseModel):
    text: str
    intent: IntentOut
    emotion: EmotionOut
    actions: list
