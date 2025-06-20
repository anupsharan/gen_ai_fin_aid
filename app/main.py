# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router as api_router

app = FastAPI(
   title="AI Stock Screener API",
   description="An API that uses yfinance and AI to provide stock analysis and recommendations.",
   version="1.1.2",
)

# Mount the static directory to serve index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORSMiddleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Include the API router
app.include_router(api_router, prefix="/api")

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def read_index():
   """Serves the main index.html file."""
   return "static/index.html"

# To run the app, use the command:
# uvicorn app.main:app --reload