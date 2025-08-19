from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# FastAPI app
app = FastAPI(title="Simple Travel Planner")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

# Simple models (no email validation)
class TripRequest(BaseModel):
    city: str
    days: int
    email: str  # Just a regular string, no validation

class TripResponse(BaseModel):
    city: str
    days: int
    email: str
    itinerary: str

@app.get("/")
def home():
    return {"message": "Simple Travel Planner API is running!"}

@app.post("/create-itinerary")
def create_itinerary(request: TripRequest):
    try:
        # Simple prompt
        prompt = f"""Create a {request.days}-day travel itinerary for {request.city}.
        
        Include:
        - Day-by-day activities
        - Morning, afternoon, evening plans
        - Must-visit places
        - Local food recommendations
        - Practical tips
        
        Format it clearly with bullet points."""
        
        # Get AI response
        response = llm.invoke(prompt)
        itinerary = response.content
        
        return TripResponse(
            city=request.city,
            days=request.days,
            email=request.email,
            itinerary=itinerary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

