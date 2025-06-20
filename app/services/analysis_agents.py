# app/services/analysis_agents.py
import asyncio
import json
import yfinance as yf
import pandas_ta as ta
from fastapi import HTTPException
from app.models.schemas import FundamentalData, TechnicalData, SentimentData, FinalRecommendation, UndervaluedStock
from app.services.gemini_client import call_gemini_api

async def fundament_agent(ticker: yf.Ticker) -> FundamentalData:
   """Agent 1: Gathers fundamental data and uses an LLM to get a recommendation."""
   info = await asyncio.to_thread(lambda: ticker.info)
   data = {
       "company_name": info.get("longName"), "price": info.get("currentPrice", info.get("previousClose", 0)),
       "analyst_price_target": info.get("targetMeanPrice"), "pe_ratio": info.get("trailingPE"),
       "revenue_growth_yoy": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None,
       "forward_eps": info.get("forwardEps"), "recommendation": "HOLD"
   }
   prompt = f"""
       Based on the following fundamental data for a stock:
       - Current Price: ${data.get('price') or 0:.2f}
       - Analyst Price Target: ${data.get('analyst_price_target') or 0:.2f}
       - P/E Ratio: {data.get('pe_ratio') or 0:.2f}
       - Revenue Growth (YoY): {data.get('revenue_growth_yoy') or 0:.2f}%
       A price below the analyst target, a low P/E ratio (<30), and strong revenue growth (>5%) are generally positive signs.
       Return a single valid JSON object with one key: "recommendation", with a value of "BUY", "SELL", or "HOLD".
   """
   try:
       response_text = await call_gemini_api(prompt)
       if response_text:
           data["recommendation"] = json.loads(response_text).get("recommendation", "HOLD")
   except Exception as e:
       print(f"Could not get AI fundamental recommendation: {e}. Defaulting to HOLD.")
   return FundamentalData(**data)


async def technical_agent(ticker: yf.Ticker) -> TechnicalData:
   """Agent 2: Gathers technical data and uses an LLM to get a recommendation."""
   hist = await asyncio.to_thread(ticker.history, period="1y")
   if hist.empty:
       raise HTTPException(status_code=404, detail="Could not fetch historical data.")
   hist.ta.adx(length=14, append=True); hist.ta.rsi(length=14, append=True); hist.ta.ema(length=50, append=True)
   hist.dropna(inplace=True)
   latest_data = hist.iloc[-1]
   data = {
       "rsi_14": latest_data['RSI_14'], "ema_50": latest_data['EMA_50'],
       "adx_14": latest_data['ADX_14'], "price": latest_data['Close'], "recommendation": "HOLD"
   }
   prompt = f"""
       Based on the following technical indicators for a stock, provide a recommendation.
       - RSI (14-day): {data.get('rsi_14') or 0:.2f}
       - EMA (50-day): ${data.get('ema_50') or 0:.2f}
       - ADX (14-day): {data.get('adx_14') or 0:.2f}
       - Current Price: ${data.get('price') or 0:.2f}
       A high ADX (>25) indicates a strong trend. RSI > 70 is overbought, RSI < 30 is oversold. Price > EMA is bullish, Price < EMA is bearish.
       Return a single valid JSON object with one key: "recommendation", with a value of "BUY", "SELL", or "HOLD".
   """
   try:
       response_text = await call_gemini_api(prompt)
       if response_text:
           data["recommendation"] = json.loads(response_text).get("recommendation", "HOLD")
   except Exception as e:
       print(f"Could not get AI technical recommendation: {e}. Defaulting to HOLD.")
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
       raise HTTPException(status_code=500, detail="Invalid JSON response from recommendation analysis.")

async def find_undervalued_stocks() -> list[UndervaluedStock]:
    """Uses an LLM to find potentially undervalued stocks."""
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
        raise HTTPException(status_code=500, detail="Invalid JSON response for undervalued stocks.")