# app/core/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
   API_KEY: str = os.getenv("GEMINI_API_KEY")
   GEMINI_API_URL: str = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

settings = Settings()