import os
import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# --- HUGGING FACE / LANGCHAIN IMPORTS ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

# --- MODULE PROMPTS ---
import prompts

load_dotenv()

st.set_page_config(
    page_title="Institutional Equity Research AI",
    page_icon="🏛️",
    layout="wide"
)

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
        
        return {
            "ticker": ticker_symbol.upper(),
            "name": info.get("shortName") or info.get("longName") or ticker_symbol.upper(),
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
            "recent_volume": hist['Volume'].iloc[-1] if 'Volume' in hist.columns else "N/A",
            "avg_volume": info.get("averageVolume", "N/A")
        }
    except Exception:
        return None

def fetch_realtime_news(ticker_symbol):
    try:
        query = f"{ticker_symbol} stock news earnings sentiment alpha"
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        news_text = ""
        for i, res in enumerate(results, 1):
            news_text += f"[{i}] {res['title']}\nSnippet: {res['body']}\n\n"
        return news_text if news_text else "No recent news found."
    except Exception as e:
        return f"Could not fetch live news due to: {e}"

def get_expanded_sector_list(primary_ticker, business_summary, llm):
    """Instructs the LLM to give both standard peers and 2 micro-cap/multi-bagger prospects, enforcing strict industry alignment."""
    sector_prompt = ChatPromptTemplate.from_messages([
        ("system", "You output valid Python lists containing strings. "
                   "Identify 4 active, liquid peer tickers matching the EXACT industry/sector of the input asset. "
                   "CRITICAL DATA HYGIENE RULES:\n"
                   "1. Ensure 2 items are massive industry benchmarks, and 2 are emerging high-growth stocks or potential small-cap multi-baggers from the SAME sector.\n"
                   "2. DO NOT mix industries (e.g., do not suggest biotech like SRNE for a software company like PLTR).\n"
                   "3. DO NOT return bankrupt, delisted, or dead penny stocks trading under $1.\n"
                   "Output ONLY the raw bracketed python list. Example format: ['TICKER1', 'TICKER2', 'TICKER3', 'TICKER4']. No filler text."),
        ("user", f"Asset Ticker: {primary_ticker}\nSummary: {business_summary[:500]}")
    ])
    chain = sector_prompt | llm
    
    try:
        response = chain.invoke({})
        text = response.content.strip()
        if "[" in text and "]" in text:
            raw_list = text[text.find("["):text.find("]")+1]
            import ast
            return [t.strip().upper() for t in ast.literal_eval(raw_list) if t.strip().upper() != primary_ticker.upper()]
    except Exception:
        pass
    return []

def run_stock_analysis_agent(ticker):
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ or not os.environ["HUGGINGFACEHUB_API_TOKEN"]:
        st.error("Missing API Credentials. Please enter your Hugging Face API Token in the sidebar.")
        return

    with st.status("🧠 Executing Dual-Alpha Discovery Pipeline...", expanded=True) as status:
        status.write(f"📊 Pulling quantitative indicators for {ticker.upper()}...")
        stock_data = fetch_stock_data(ticker)
        
        if not stock_data:
            status.update(label="Data Retrieval Failed", state="error")
            st.error("Unable to process stock metrics. Please verify the ticker symbol.")
            return

        status.write(f"🌐 Sourcing live news & structural catalyst tracking data...")
        news_data = fetch_realtime_news(ticker)

        status.write("🔧 Instantiating Qwen-72B LLM Core Layer...")
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-72B-Instruct", 
            task="text-generation",
            temperature=0.1,
            max_new_tokens=2500,
        )
        llm = ChatHuggingFace(llm=llm_endpoint)

        status.write("🗂️ Generating expanded sector competitor array (Blue Chips + High Growth)...")
        competitor_tickers = get_expanded_sector_list(ticker.upper(), stock_data["business_summary"], llm)
        
        status.write("📊 Compiling live peer data matrices...")
        peer_data_map = {}
        matrix_context = ""
        for t in competitor_tickers:
            p_data = fetch_stock_data(t)
            if p_data:
                peer_data_map[t.upper()] = p_data
                matrix_context += f"Candidate Ticker: {p_data['ticker']} | Company Name: {p_data['name']} | Price: {p_data['current_price']} | Rev Growth: {p_data['revenue_growth']}\n"

        status.write("🎯 Isolating the high-growth multi-bagger addition programmatically...")
        select_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.SECTOR_RANKING_SYSTEM),
            ("user", prompts.SECTOR_RANKING_USER)
        ])
        selector_chain = select_prompt | llm
        
        try:
            selection_res = selector_chain.invoke({"sector_matrix_context": matrix_context})
            selected_mb_ticker = selection_res.content.strip().split()[0].strip("`'\".,").upper()
            if selected_mb_ticker not in peer_data_map:
                selected_mb_ticker = list(peer_data_map.keys())[0]
            mb_data = peer_data_map[selected_mb_ticker]
        except Exception:
            selected_mb_ticker = "N/A"
            mb_data = {"ticker": "N/A", "current_price": "N/A", "revenue_growth": "N/A", "pe_ratio": "N/A", "rsi_14": "N/A"}
        # Programmatic Filter: Map remaining assets exclusively into Stable Peers to avoid multi-bagger tunnel vision
        stable_peers_context = ""
        for t, data in peer_data_map.items():
            if t != selected_mb_ticker:
                stable_peers_context += f"- Ticker: {data['ticker']} | Price: {data['current_price']} | P/E: {data['pe_ratio']} | Rev Growth: {data['revenue_growth']} | Margin: {data['profit_margins']}\n"

        status.write(f"🤖 Synthesizing complete balanced institutional research dossier...")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompts.SYSTEM_ROLE),
            ("user", prompts.USER_PROMPT)
        ])
        chain = prompt_template | llm

        try:
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
                
                # Verified allocations split explicitly to match prompts.py parameters
                "stable_peers_context": stable_peers_context if stable_peers_context else "No distinct stable peer data recorded.",
                "mb_ticker": mb_data["ticker"],
                "mb_price": mb_data["current_price"],
                "mb_revenue_growth": mb_data["revenue_growth"],
                "mb_pe": mb_data["pe_ratio"],
                
                "news_context": news_data
            })
            status.update(label="Dual Alpha Report Dispatched!", state="complete")
        except Exception as e:
            status.update(label="Pipeline Processing Failure", state="error")
            st.error(f"Error during execution: {e}")
            return

    raw_content = response.content
    try:
        part_1_raw = raw_content.split("PART_2:")[0].replace("PART_1:", "").strip()
        part_2_raw = raw_content.split("PART_2:")[1].split("PART_3:")[0].strip()
        part_3_raw = raw_content.split("PART_3:")[1].strip()
    except IndexError:
        part_1_raw = "### 1. Core Fundamental Architecture\n" + raw_content[:len(raw_content)//3]
        part_2_raw = "### 2. Sector Volatility & Technical Setup\n" + raw_content[len(raw_content)//3: 2*len(raw_content)//3]
        part_3_raw = "### 3. Comparative Allocations & Alpha Recommendations\n" + raw_content[2*len(raw_content)//3:]

    st.markdown(f"## 🏛️ Comprehensive Sector Core & Multi-Bagger Report: {ticker.upper()}")
    st.write("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div style='border-top: 5px solid #d97706; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(part_1_raw)
    with col2:
        st.markdown("<div style='border-top: 5px solid #059669; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(part_2_raw)
    with col3:
        st.markdown("<div style='border-top: 5px solid #4f46e5; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(part_3_raw)
        
    st.write("---")
    st.caption("Educational research data framework. All equity positions introduce varying investment risks.")

def main():
    st.sidebar.title("🔐 Agent Controls")
    hf_api_token = st.sidebar.text_input("Hugging Face Hub API Token", type="password")
    ticker_input = st.sidebar.text_input("Enter Ticker Asset (e.g., MSFT, TATAGOLD.NS, RELIANCE.NS)", value="MSFT")
    trigger_analysis = st.sidebar.button("Run Portfolio Discovery", type="primary")
    
    if trigger_analysis:
        if hf_api_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
            run_stock_analysis_agent(ticker_input.strip())
        else:
            st.sidebar.error("Error: Please provide a valid token configuration.")

if __name__ == "__main__":
    main()