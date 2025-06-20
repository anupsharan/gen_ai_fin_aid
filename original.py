import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
import pandas_ta as ta # Import pandas_ta
import requests
import json
from typing import List, Optional
import os


# --- Configuration & Setup ---


app = FastAPI(
   title="AI Stock Screener API",
   description="An API that uses yfinance and AI to provide stock analysis and recommendations.",
   version="1.1.2",
)


# Add CORSMiddleware to allow the frontend to communicate with the backend.
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"], # Allows all origins for simplicity
   allow_credentials=True,
   allow_methods=["*"], # Allows all methods
   allow_headers=["*"], # Allows all headers
)




# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google AI Studio API key.
API_KEY = "AIzaSyDlnEGPI_eE1B_q6ZCbXpnO9CWhipOMt8k"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"


# --- Pydantic Models (for data validation and response structure) ---


class TradeRequest(BaseModel):
   ticker: str


class FundamentalData(BaseModel):
   company_name: str
   price: float
   analyst_price_target: Optional[float] = None
   pe_ratio: Optional[float] = None
   revenue_growth_yoy: Optional[float] = None
   forward_eps: Optional[float] = None
   recommendation: str


class TechnicalData(BaseModel):
   rsi_14: Optional[float] = None
   ema_50: Optional[float] = None
   adx_14: Optional[float] = None
   price: float
   recommendation: str


class SentimentData(BaseModel):
   sentiment_summary: str
   recommendation: str
   reasoning: str


class FinalRecommendation(BaseModel):
   overall_recommendation: str
   overall_reasoning: str


class StockAnalysis(BaseModel):
   ticker: str
   fundamental: FundamentalData
   technical: TechnicalData
   sentiment: SentimentData
   final_recommendation: FinalRecommendation


class UndervaluedStock(BaseModel):
   ticker: str
   company_name: str
   reason: str


# --- AGENT FUNCTIONS (adapted for async FastAPI) ---


async def call_gemini_api(prompt: str) -> Optional[str]:
   """Asynchronously sends a prompt to the Gemini API and cleans the response."""
   if API_KEY == "YOUR_API_KEY":
       # This part is for demonstration if no API key is provided
       print("Warning: API_KEY not set. Returning mock data.")
       mock_responses = {
           "sentiment": '{"sentiment_summary": "General sentiment is positive.", "recommendation": "BUY", "reasoning": "Positive news coverage."}',
           "recommendation": '{"overall_recommendation": "BUY", "overall_reasoning": "Strong fundamentals and positive sentiment."}',
           "undervalued": '[{"ticker": "PFE", "company_name": "Pfizer Inc.", "reason": "Stable dividend yield suggests undervaluation in DCF model."}, {"ticker": "INTC", "company_name": "Intel Corporation", "reason": "Heavy CapEx for foundries may be underappreciated in current price according to DCF."}, {"ticker": "DELL", "company_name": "Dell Technologies", "reason": "Strong enterprise free cash flow points to DCF value."}, {"ticker": "V", "company_name": "Visa Inc.", "reason": "Consistent transaction growth supports a high DCF valuation."}, {"ticker": "JPM", "company_name": "JPMorgan Chase & Co.", "reason": "Strong balance sheet provides a solid base for DCF analysis."}]',
           "technical": '{"recommendation": "HOLD"}',
           "fundamental": '{"recommendation": "BUY"}' # Added mock for fundamental agent
       }
       if "sentiment" in prompt.lower(): return mock_responses["sentiment"]
       if "undervalued" in prompt.lower(): return mock_responses["undervalued"]
       if "technical indicators" in prompt.lower(): return mock_responses["technical"]
       if "fundamental data" in prompt.lower(): return mock_responses["fundamental"]
       return mock_responses["recommendation"]


   headers = {'Content-Type': 'application/json'}
   data = {"contents": [{"parts": [{"text": prompt}]}]}


   loop = asyncio.get_event_loop()
   try:
       response = await loop.run_in_executor(
           None,
           lambda: requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(data), timeout=20)
       )
       response.raise_for_status()
       result = response.json()
      
       if not result.get('candidates') or not result['candidates'][0].get('content', {}).get('parts'):
            raise HTTPException(status_code=500, detail="Invalid response structure from Gemini API: No content found.")
      
       text = result['candidates'][0]['content']['parts'][0]['text']
      
       # Clean the response to remove markdown formatting before returning.
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




async def fundament_agent(ticker: yf.Ticker) -> FundamentalData:
   """Agent 1: Gathers fundamental data and uses an LLM to get a recommendation."""
   info = await asyncio.to_thread(lambda: ticker.info)


   data = {
       "company_name": info.get("longName"),
       "price": info.get("currentPrice", info.get("previousClose", 0)),
       "analyst_price_target": info.get("targetMeanPrice"),
       "pe_ratio": info.get("trailingPE"),
       "revenue_growth_yoy": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None,
       "forward_eps": info.get("forwardEps"),
       "recommendation": "HOLD" # Default value
   }


   # UPDATED: Handle potential None values from yfinance before formatting
   price = data.get('price') or 0
   analyst_target = data.get('analyst_price_target') or 0
   pe = data.get('pe_ratio') or 0
   rev_growth = data.get('revenue_growth_yoy') or 0


   prompt = f"""
       Based on the following fundamental data for a stock:
       - Current Price: ${price:.2f}
       - Analyst Price Target: ${analyst_target:.2f}
       - P/E Ratio: {pe:.2f}
       - Revenue Growth (YoY): {rev_growth:.2f}%


       A price below the analyst target, a low P/E ratio (<30), and strong revenue growth (>5%) are generally positive signs.


       Return a single valid JSON object with one key: "recommendation", with a value of "BUY", "SELL", or "HOLD".
   """
  
   try:
       response_text = await call_gemini_api(prompt)
       if response_text:
           rec_data = json.loads(response_text)
           data["recommendation"] = rec_data.get("recommendation", "HOLD")
   except Exception as e:
       print(f"Could not get AI fundamental recommendation: {e}. Defaulting to HOLD.")
       # The recommendation will remain "HOLD" if the API call fails


   return FundamentalData(**data)




async def technical_agent(ticker: yf.Ticker) -> TechnicalData:
   """Agent 2: Gathers technical data and uses an LLM to get a recommendation."""
   hist = await asyncio.to_thread(ticker.history, period="1y")
   if hist.empty:
       raise HTTPException(status_code=404, detail="Could not fetch historical data.")


   # Use pandas_ta to calculate indicators
   hist.ta.adx(length=14, append=True)
   hist.ta.rsi(length=14, append=True)
   hist.ta.ema(length=50, append=True)
   hist.dropna(inplace=True)


   latest_data = hist.iloc[-1]
  
   data = {
       "rsi_14": latest_data['RSI_14'],
       "ema_50": latest_data['EMA_50'],
       "adx_14": latest_data['ADX_14'],
       "price": latest_data['Close'],
       "recommendation": "HOLD" # Default value
   }
  
   # UPDATED: Handle potential None values before formatting
   rsi = data.get('rsi_14') or 0
   ema = data.get('ema_50') or 0
   adx = data.get('adx_14') or 0
   price = data.get('price') or 0


   prompt = f"""
       Based on the following technical indicators for a stock, provide a recommendation.
       - RSI (14-day): {rsi:.2f}
       - EMA (50-day): ${ema:.2f}
       - ADX (14-day): {adx:.2f}
       - Current Price: ${price:.2f}


       A high ADX (>25) indicates a strong trend. RSI > 70 is overbought, RSI < 30 is oversold. Price > EMA is bullish, Price < EMA is bearish.


       Return a single valid JSON object with one key: "recommendation", with a value of "BUY", "SELL", or "HOLD".
   """
  
   try:
       response_text = await call_gemini_api(prompt)
       if response_text:
           rec_data = json.loads(response_text)
           data["recommendation"] = rec_data.get("recommendation", "HOLD")
   except Exception as e:
       print(f"Could not get AI technical recommendation: {e}. Defaulting to HOLD.")
       # The recommendation will remain "HOLD" if the API call fails
  
   return TechnicalData(**data)




async def sentiment_agent(ticker_symbol: str, company_name: str) -> SentimentData:
   """Agent 3: Uses LLM to gauge market sentiment."""
   prompt = f'''
       Analyze market sentiment for "{company_name} ({ticker_symbol})".
       Return a valid JSON object with three string fields:
       1. "sentiment_summary": A one-sentence summary of the general consensus.
       2. "recommendation": Your sentiment-based verdict ("BUY", "SELL", or "HOLD").
       3. "reasoning": A single sentence explaining the sentiment-based recommendation.
   '''
   response_text = await call_gemini_api(prompt)
   try:
       return SentimentData(**json.loads(response_text))
   except json.JSONDecodeError:
       print(f"Failed to decode JSON from sentiment_agent. Response: '{response_text}'")
       raise HTTPException(status_code=500, detail="Invalid JSON response from sentiment analysis.")




async def recommendation_agent(fundamental, technical, sentiment) -> FinalRecommendation:
   """Agent 4: Provides the final recommendation."""
   prompt = f"""
       Given the following reports:
       1. Fundamental: {fundamental.model_dump_json()}
       2. Technical: {technical.model_dump_json()}
       3. Sentiment: {sentiment.model_dump_json()}
       Provide a final verdict. Return a valid JSON object with "overall_recommendation" and "overall_reasoning".
   """
   response_text = await call_gemini_api(prompt)
   try:
       return FinalRecommendation(**json.loads(response_text))
   except json.JSONDecodeError:
       print(f"Failed to decode JSON from recommendation_agent. Response: '{response_text}'")
       raise HTTPException(status_code=500, detail="Invalid JSON response from recommendation analysis.")




# --- API Endpoints ---


@app.get("/", response_class=FileResponse)
async def read_index():
   return "index.html"


@app.get("/api/portfolio", response_model=List[str], tags=["Portfolio"])
async def get_portfolio():
   """
   Reads tickers from portfolio.txt, de-duplicates, and sorts them.
   """
   portfolio_path = "portfolio.txt"
   default_portfolio = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
   if not os.path.exists(portfolio_path):
       return sorted(default_portfolio)
   try:
       with open(portfolio_path, "r") as f:
           tickers = {line.strip().upper() for line in f if line.strip()}
           return sorted(list(tickers)) if tickers else sorted(default_portfolio)
   except Exception as e:
       print(f"Error reading portfolio.txt: {e}")
       return sorted(default_portfolio)


@app.get("/api/analyze/{ticker_symbol}", response_model=StockAnalysis, tags=["Analysis"])
async def analyze_stock(ticker_symbol: str):
   """
   Performs a full analysis (fundamental, technical, sentiment) for a given stock ticker.
   """
   try:
       ticker = yf.Ticker(ticker_symbol)
       if not ticker.info.get('longName'):
            raise HTTPException(status_code=404, detail=f"Ticker symbol '{ticker_symbol}' not found or invalid.")


       fundamental_task = fundament_agent(ticker)
       technical_task = technical_agent(ticker)
       fundamental_data, technical_data = await asyncio.gather(fundamental_task, technical_task)


       sentiment_data = await sentiment_agent(ticker_symbol, fundamental_data.company_name)
       final_rec_data = await recommendation_agent(fundamental_data, technical_data, sentiment_data)


       return StockAnalysis(
           ticker=ticker_symbol.upper(),
           fundamental=fundamental_data,
           technical=technical_data,
           sentiment=sentiment_data,
           final_recommendation=final_rec_data
       )


   except HTTPException as e:
       raise e
   except Exception as e:
       print(f"An unexpected error occurred: {e}")
       raise HTTPException(status_code=500, detail=f"An unexpected error occurred for ticker {ticker_symbol}: {str(e)}")




@app.get("/api/undervalued-stocks", response_model=List[UndervaluedStock], tags=["Discovery"])
async def get_undervalued_stocks():
   """
   Returns a list of potentially undervalued stocks based on AI analysis.
   """
   prompt = """
       List 5 random potentially undervalued stocks, selected primarily based on a discounted cash flow (DCF) model analysis.
       For each, provide a valid JSON object with 'ticker', 'company_name', and a brief 'reason' explaining the DCF angle.
       Return the response as a valid JSON array of these objects.
   """
   try:
       response_text = await call_gemini_api(prompt)
       stocks_data = json.loads(response_text)
       return [UndervaluedStock(**stock) for stock in stocks_data]
   except json.JSONDecodeError:
       print(f"Failed to decode JSON from get_undervalued_stocks. Response: '{response_text}'")
       raise HTTPException(status_code=500, detail="Invalid JSON response for undervalued stocks.")
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Failed to get undervalued stocks: {e}")


@app.post("/api/trade/buy", tags=["Trading"])
async def trade_buy(request: TradeRequest):
   """Simulates placing a buy order."""
   print(f"Received BUY request for {request.ticker}")
   return JSONResponse(content={"message": f"Your BUY order for {request.ticker} has been queued."})


@app.post("/api/trade/sell", tags=["Trading"])
async def trade_sell(request: TradeRequest):
   """Simulates placing a sell order."""
   print(f"Received SELL request for {request.ticker}")
   return JSONResponse(content={"message": f"Your SELL order for {request.ticker} has been queued."})
