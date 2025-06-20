# app/api/endpoints.py
import asyncio
import os
import yfinance as yf
import json
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List

from app.models.schemas import StockAnalysis, UndervaluedStock, TradeRequest
from app.services.analysis_agents import (
    fundament_agent,
    technical_agent,
    sentiment_agent,
    recommendation_agent,
    find_undervalued_stocks
)

router = APIRouter()

@router.get("/portfolio", response_model=List[str], tags=["Portfolio"])
async def get_portfolio():
   """Reads tickers from portfolio.txt, de-duplicates, and sorts them."""
   default_portfolio = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
   if not os.path.exists("portfolio.txt"):
       return sorted(default_portfolio)
   try:
       with open("portfolio.txt", "r") as f:
           tickers = {line.strip().upper() for line in f if line.strip()}
           return sorted(list(tickers)) if tickers else sorted(default_portfolio)
   except Exception as e:
       return sorted(default_portfolio)


@router.get("/analyze/{ticker_symbol}", response_model=StockAnalysis, tags=["Analysis"])
async def analyze_stock(ticker_symbol: str):
   """Performs a full analysis (fundamental, technical, sentiment) for a given stock ticker."""
   try:
       ticker = yf.Ticker(ticker_symbol)
       if not ticker.info.get('longName'):
           raise HTTPException(status_code=404, detail=f"Ticker '{ticker_symbol}' not found.")

       fundamental_data = await fundament_agent(ticker)
       technical_data = await technical_agent(ticker)
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
       raise HTTPException(status_code=500, detail=f"An unexpected error occurred for {ticker_symbol}: {e}")


@router.get("/undervalued-stocks", response_model=List[UndervaluedStock], tags=["Discovery"])
async def get_undervalued_stocks():
    """Returns a list of potentially undervalued stocks based on AI analysis."""
    try:
        return await find_undervalued_stocks()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get undervalued stocks: {e}")


@router.post("/trade/buy", tags=["Trading"])
async def trade_buy(request: TradeRequest):
   """Simulates placing a buy order."""
   print(f"Received BUY request for {request.ticker}")
   return JSONResponse(content={"message": f"Your BUY order for {request.ticker} has been queued."})


@router.post("/trade/sell", tags=["Trading"])
async def trade_sell(request: TradeRequest):
   """Simulates placing a sell order."""
   print(f"Received SELL request for {request.ticker}")
   return JSONResponse(content={"message": f"Your SELL order for {request.ticker} has been queued."})