print("THIS IS THE NEW MAIN.PY RUNNING")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Create app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model locally
generator = pipeline("text-generation", model="distilgpt2")

# Request model
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "AI Creative Engine Running"}

@app.post("/generate-text")
def generate_text(request: PromptRequest):
    result = generator(
    request.prompt,
    max_length=100,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    repetition_penalty=1.2,
    num_return_sequences=1
)

    return {
        "prompt": request.prompt,
        "ai_response": result[0]["generated_text"]
    }
    