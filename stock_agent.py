import os
import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from duckduckgo_search import DDGS
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()

# --- NEW HUGGING FACE IMPORTS ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

# Ensure API Key is loaded
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    raise ValueError("Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")

# 1. DATA EXTRACTION TOOL: Financials & Technicals via yfinance
def fetch_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Historical data for Technical Analysis (6 months)
        hist = ticker.history(period="6mo")
        if hist.empty:
            return None
        
        # Calculate Technicals using 'ta' library
        close_prices = hist['Close']
        rsi = RSIIndicator(close=close_prices, window=14).rsi().iloc[-1]
        macd_obj = MACD(close=close_prices)
        macd = macd_obj.macd().iloc[-1]
        macd_signal = macd_obj.macd_signal().iloc[-1]
        sma_50 = SMAIndicator(close=close_prices, window=50).sma_indicator().iloc[-1]
        sma_200 = SMAIndicator(close=close_prices, window=200).sma_indicator().iloc[-1]
        
        # Package metrics safely
        data_pack = {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or close_prices.iloc[-1],
            "business_summary": info.get("longBusinessSummary", "N/A"),
            "revenue_growth": info.get("revenueGrowth", "N/A"),
            "earnings_growth": info.get("earningsGrowth", "N/A"),
            "profit_margins": info.get("profitMargins", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "free_cashflow": info.get("freeCashflow", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "peg_ratio": info.get("pegRatio", "N/A"),
            "ev_ebitda": info.get("enterpriseToEbitda", "N/A"),
            "pb_ratio": info.get("priceToBook", "N/A"),
            "roe": info.get("returnOnEquity", "N/A"),
            "institutional_ownership": info.get("heldPercentInstitutions", "N/A"),
            "rsi_14": round(rsi, 2) if not pd.isna(rsi) else "N/A",
            "macd": round(macd, 4) if not pd.isna(macd) else "N/A",
            "macd_signal": round(macd_signal, 4) if not pd.isna(macd_signal) else "N/A",
            "sma_50": round(sma_50, 2) if not pd.isna(sma_50) else "N/A",
            "sma_200": round(sma_200, 2) if not pd.isna(sma_200) else "N/A",
            "recent_volume": hist['Volume'].iloc[-1],
            "avg_volume": info.get("averageVolume", "N/A")
        }
        return data_pack
    except Exception as e:
        print(f"Error gathering yfinance data: {e}")
        return None

# 2. DATA EXTRACTION TOOL: Real-time News via DuckDuckGo
def fetch_realtime_news(ticker_symbol):
    try:
        with DDGS() as ddgs:
            query = f"{ticker_symbol} stock news earnings sentiment"
            results = [r for r in ddgs.text(query, max_results=5)]
        
        news_text = ""
        for i, res in enumerate(results, 1):
            news_text += f"[{i}] {res['title']}\nSnippet: {res['body']}\n\n"
        return news_text if news_text else "No recent news found."
    except Exception as e:
        return f"Could not fetch live news due to: {e}"

# 3. AGENT DEFINITION & SYSTEM PROMPT
def run_stock_analysis_agent(ticker):
    print(f"🔄 Gathering financial data for {ticker.upper()}...")
    stock_data = fetch_stock_data(ticker)
    
    if not stock_data:
        print("❌ Error: Unable to fetch stock metrics. Please verify the ticker symbol.")
        return

    print(f"🔄 Gathering web news & market sentiment for {ticker.upper()}...")
    news_data = fetch_realtime_news(ticker)

    # --- NEW HUGGING FACE MODEL SETUP ---
    # We call the Hugging Face serverless endpoint engine using a heavy instruction model
    print(f"🧠 Initializing Hugging Face connection...")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct", 
        task="text-generation",
        temperature=0.2,
        max_new_tokens=2048, # High allocation for full data reading
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    system_role = """
You are an institutional-grade Stock Analysis and Trade Recommendation Assistant. 
Your objective is to identify high-probability investment and trading opportunities using a combination of:
Fundamental Analysis, Technical Analysis, Market Sentiment Analysis, Macroeconomic Factors, Sector Analysis, Risk Management Principles, and Real-time News.

Core Rules:
1. Never provide a Buy, Hold, or Sell recommendation without supporting evidence.
2. Always explain the reasoning behind every conclusion.
3. Use the current market information, technical metrics, and recent news provided to you.
4. Challenge your own recommendation by explicitly identifying at least three risks or bearish factors before arriving at a final decision.
5. If available information is insufficient, state "Insufficient evidence" instead of guessing.
6. Focus on probability, not certainty. Never guarantee profits or future returns.
7. Clearly separate facts, assumptions, and opinions.

Output Format:
You must strictly format your entire output using the following layout structural breakdown:

Stock Summary
Fundamental Analysis
Technical Analysis
News & Market Sentiment
Bullish Factors
Bearish Factors (Minimum 3 risks/bearish factors challenging your thesis)
Risk Assessment
Trade Setup
- Current Price: 
- Best Entry Price: 
- Stop Loss: 
- Target 1: 
- Target 2: 
- Risk/Reward Ratio: 
Confidence Score (0-100%)
Final Recommendation (Strong Buy / Buy / Watchlist / Hold / Avoid)

End every report exactly with this verbatim statement:
"This analysis is for educational and research purposes only and should not be considered financial advice. Markets involve risk, and all investment decisions should be independently verified."
"""

    user_prompt = """
Analyze the following raw data collected for Ticker: {ticker}

--- RAW FUNDAMENTAL & TECHNICAL DATA ---
Current Price: {current_price}
Business Summary: {business_summary}
Revenue Growth: {revenue_growth} | Earnings Growth: {earnings_growth} | Profit Margins: {profit_margins}
Debt to Equity: {debt_to_equity} | Free Cash Flow: {free_cashflow}
Valuation Metrics -> P/E: {pe_ratio} | PEG: {peg_ratio} | EV/EBITDA: {ev_ebitda} | P/B: {pb_ratio}
Return on Equity (ROE): {roe} | Institutional Ownership: {institutional_ownership}

Technicals -> RSI(14): {rsi_14} | MACD: {macd} | MACD Signal: {macd_signal}
Moving Averages -> 50 SMA: {sma_50} | 200 SMA: {sma_200}
Volume -> Latest Session Volume: {recent_volume} | Avg Volume: {avg_volume}

--- REAL-TIME NEWS & WEB RESEARCH ---
{news_context}

Perform your 5-step analysis framework internally and produce the final formatted report.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_role),
        ("user", user_prompt)
    ])

    # Construct execution chain
    chain = prompt_template | llm

    print(f"🧠 Processing institutional analysis report via Hugging Face...")
    response = chain.invoke({
        "ticker": ticker.upper(),
        "current_price": stock_data["current_price"],
        "business_summary": stock_data["business_summary"],
        "revenue_growth": stock_data["revenue_growth"],
        "earnings_growth": stock_data["earnings_growth"],
        "profit_margins": stock_data["profit_margins"],
        "debt_to_equity": stock_data["debt_to_equity"],
        "free_cashflow": stock_data["free_cashflow"],
        "pe_ratio": stock_data["pe_ratio"],
        "peg_ratio": stock_data["peg_ratio"],
        "ev_ebitda": stock_data["ev_ebitda"],
        "pb_ratio": stock_data["pb_ratio"],
        "roe": stock_data["roe"],
        "institutional_ownership": stock_data["institutional_ownership"],
        "rsi_14": stock_data["rsi_14"],
        "macd": stock_data["macd"],
        "macd_signal": stock_data["macd_signal"],
        "sma_50": stock_data["sma_50"],
        "sma_200": stock_data["sma_200"],
        "recent_volume": stock_data["recent_volume"],
        "avg_volume": stock_data["avg_volume"],
        "news_context": news_data
    })

    # ---- RICH COLORFUL CARD FORMATTING ----
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    console = Console()
    
    console.print("\n")
    console.print(
        Panel(
            f"[bold white]INSTITUTIONAL EQUITY RESEARCH REPORT: {ticker.upper()}[/bold white]",
            style="bold cyan on blue", 
            expand=False
        )
    )
    
    report_card = Panel(
        Markdown(response.content),
        title=f"[bold green]📊 {ticker.upper()} Analysis Card[/bold green]",
        title_align="left",
        border_style="bright_magenta",
        padding=(1, 2)
    )
    
    console.print(report_card)

if __name__ == "__main__":
    target_stock = input("Enter Stock Ticker (e.g., RELIANCE.NS, NVDA): ").strip()
    if target_stock:
        run_stock_analysis_agent(target_stock)