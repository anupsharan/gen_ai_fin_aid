# app/services/gemini_client.py
import asyncio
import requests
import json
from typing import Optional
from fastapi import HTTPException
from app.core.config import settings

async def call_gemini_api(prompt: str) -> Optional[str]:
   """Asynchronously sends a prompt to the Gemini API and cleans the response."""
   if not settings.API_KEY or settings.API_KEY == "YOUR_API_KEY":
       raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

   headers = {'Content-Type': 'application/json'}
   data = {"contents": [{"parts": [{"text": prompt}]}]}

   loop = asyncio.get_event_loop()
   try:
       response = await loop.run_in_executor(
           None,
           lambda: requests.post(settings.GEMINI_API_URL, headers=headers, data=json.dumps(data), timeout=20)
       )
       response.raise_for_status()
       result = response.json()
      
       if not result.get('candidates') or not result['candidates'][0].get('content', {}).get('parts'):
           raise HTTPException(status_code=500, detail="Invalid response structure from Gemini API: No content found.")
      
       text = result['candidates'][0]['content']['parts'][0]['text']
      
       if text.strip().startswith("```json"):
           text = text.strip()[7:-3]
       elif text.strip().startswith("```"):
           text = text.strip()[3:-3]
          
       return text.strip()

   except requests.exceptions.RequestException as e:
       print(f"API Error: {e}")
       raise HTTPException(status_code=503, detail=f"Gemini API request failed: {e}")
   except (KeyError, IndexError) as e:
       print(f"API Error - Key/Index Error: {e}. Response: {result}")
       raise HTTPException(status_code=500, detail="Invalid response format from Gemini API.")