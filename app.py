"""
FastAPI app exposing /analyze
POST JSON: { "transcript": "<full transcript text>" }
Returns same structure as demo.py analyze_transcript.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from nlp_pipeline import analyze_transcript

app = FastAPI(title="Physician Notetaker API")


class TranscriptRequest(BaseModel):
    transcript: str


@app.post("/analyze")
async def analyze(req: TranscriptRequest):
    res = analyze_transcript(req.transcript)
    return res


@app.get("/")
def root():
    return {"msg": "Physician Notetaker — send POST /analyze with transcript"}
