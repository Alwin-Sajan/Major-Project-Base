import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

assitant_router = APIRouter(prefix="/assistant", tags=["assistant"])

class ChatRequest(BaseModel):
    message: str
    context: str = None

@assitant_router.post("/chat")
async def assistant_help(request: ChatRequest):
    try:
        # Define a Marine Biologist Persona
        system_instruction = "You are a specialized Marine Biologist AI. Provide concise, scientific, and helpful information about marine species." \
        "Politely deny any requests for non-marine information and guide users to ask about marine life. Always maintain a professional tone."
        full_prompt = f"{system_instruction}\nContext: User is viewing {request.context}\nUser: {request.message}"
        
        response = model.generate_content(full_prompt)
        return {"response": response.text}
    except Exception as e: 
        print(f"Error: {e}")
        return {"response": "I'm having trouble connecting to my knowledge base right now."}

app.include_router(assitant_router)
 
# Your existing OOD prediction route would go here...