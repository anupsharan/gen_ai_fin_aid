# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

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