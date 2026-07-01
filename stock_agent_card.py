import os
import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# --- HUGGING FACE IMPORTS ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Set up Streamlit Page Configuration
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
        st.error(f"Error gathering yfinance data: {e}")
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

# 3. AGENT EXECUTION AND RENDERING ENGINE
def run_stock_analysis_agent(ticker):
    # Fallback to check API token presence inside Streamlit process
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ or not os.environ["HUGGINGFACEHUB_API_TOKEN"]:
        st.error("Missing API Credentials. Please enter your Hugging Face API Token in the sidebar.")
        return

    with st.status("🧠 Executing Institutional Analysis Pipeline...", expanded=True) as status:
        status.write(f"📊 Pulling quantitative indicators for {ticker.upper()}...")
        stock_data = fetch_stock_data(ticker)
        
        if not stock_data:
            status.update(label="Data Retrieval Failed", state="error")
            st.error("Unable to process stock metrics. Please verify the ticker symbol.")
            return

        status.write(f"🌐 Fetching live market sentiment and news for {ticker.upper()}...")
        news_data = fetch_realtime_news(ticker)

        status.write(f"🤖 Interfacing with Qwen-72B LLM Core...")
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-72B-Instruct", 
            task="text-generation",
            temperature=0.2,
            max_new_tokens=2500,
        )
        llm = ChatHuggingFace(llm=llm_endpoint)

        # Altered system role to guarantee cleanly tokenized markdown string splits
#         system_role = """
# You are an institutional-grade Stock Analysis and Trade Recommendation Assistant. 
# Your objective is to generate a comprehensive asset research dossier. 
# You must divide your final response into EXACTLY 3 sections using these precise token markers:
# "PART_1: Fundamental & Business Strategy"
# "PART_2: Technicals & Sentiment Analysis"
# "PART_3: Risks & Complete Trade Setup"

# Core Rules within the layout:
# 1. Challenge your own recommendation with at least three risks or bearish factors in Part 3.
# 2. In Part 3, include the full Trade Setup breakdown: Current Price, Best Entry Price, Stop Loss, Target 1, Target 2, Risk/Reward Ratio, Confidence Score, and Final Recommendation.
# 3. Conclude each section with a short paragraph titled "Historical Cycle Precedent:" before moving to the next section token.
# """
        # system_role = """
        # You are an institutional-grade Stock Analysis and Trade Recommendation Assistant. 
        # Your objective is to generate a comprehensive asset research dossier. 
        # You must divide your final response into EXACTLY 3 sections using these precise token markers:
        # "PART_1: Fundamental & Business Strategy"
        # "PART_2: Technicals & Sentiment Analysis"
        # "PART_3: Risks, Competitor Peer Comparison & Trade Setup"

        # Core Rules within the layout:
        # 1. Challenge your own recommendation with at least three risks or bearish factors in Part 3.
        # 2. In Part 3, you MUST cross-examine the requested stock against its industry/sector peers. Suggest at least ONE alternative stock/share from the same sector that has better valuation metrics, stronger growth, or safer entry technicals than the requested stock. 
        # 3. Provide a clear "Alternative Buy Suggestion" section in Part 3 detailing why that alternative share is a better buy right now, along with an actionable trade setup.
        # 4. Conclude each section with a short paragraph titled "Historical Cycle Precedent:" before moving to the next section token.
        # """
        system_role = """
        You are an institutional-grade Stock Analysis and Trade Recommendation Assistant. 
        Your objective is to generate a comprehensive asset research dossier. 
        You must divide your final response into EXACTLY 3 sections using these precise token markers:
        "PART_1: Fundamental & Business Strategy"
        "PART_2: Technicals & Sentiment Analysis"
        "PART_3: Risks & Complete Trade Setup"

        Core Rules within the layout:
        1. Challenge your own recommendation with at least three risks or bearish factors in Part 3.
        2. In Part 3, evaluate the searched stock against its primary industry peers. You MUST suggest at least ONE better alternative stock/share from the same sector with superior fundamentals or technical setups.
        3. Conclude Part 3 with a distinct markdown block titled "### 🏛️ FINAL INVESTMENT VERDICT". In this block, you must make a definitive choice: State explicitly WHICH specific stock to buy (the searched stock OR the alternative). 
        4. The Final Verdict must include:
        - **Winner**: [Exact Ticker to Buy]
        - **Core Thesis**: [1-2 sentences explaining why it beats the other]
        - **Optimal Buy Price**: [Specific entry price or range]
        - **Target Price**: [Specific target price]
        - **Estimated Horizon**: [X Days - provide a specific number or range of days to achieve the target]
        5. Conclude each section with a short paragraph titled "Historical Cycle Precedent:" before moving to the next section token.
        """
#         user_prompt = """
# Analyze the following raw data collected for Ticker: {ticker}

# --- RAW FUNDAMENTAL & TECHNICAL DATA ---
# Current Price: {current_price}
# Business Summary: {business_summary}
# Revenue Growth: {revenue_growth} | Earnings Growth: {earnings_growth} | Profit Margins: {profit_margins}
# Debt to Equity: {debt_to_equity} | Free Cash Flow: {free_cashflow}
# Valuation Metrics -> P/E: {pe_ratio} | PEG: {peg_ratio} | EV/EBITDA: {ev_ebitda} | P/B: {pb_ratio}
# Return on Equity (ROE): {roe} | Institutional Ownership: {institutional_ownership}

# Technicals -> RSI(14): {rsi_14} | MACD: {macd} | MACD Signal: {macd_signal}
# Moving Averages -> 50 SMA: {sma_50} | 200 SMA: {sma_200}
# Volume -> Latest Session Volume: {recent_volume} | Avg Volume: {avg_volume}

# --- REAL-TIME NEWS & WEB RESEARCH ---
# {news_context}

# Perform your analysis framework internally and produce your structured text output.
# """
        # user_prompt = """
        # Analyze the following raw data collected for Ticker: {ticker}

        # --- RAW FUNDAMENTAL & TECHNICAL DATA ---
        # Current Price: {current_price}
        # Business Summary: {business_summary}
        # Revenue Growth: {revenue_growth} | Earnings Growth: {earnings_growth} | Profit Margins: {profit_margins}
        # Debt to Equity: {debt_to_equity} | Free Cash Flow: {free_cashflow}
        # Valuation Metrics -> P/E: {pe_ratio} | PEG: {peg_ratio} | EV/EBITDA: {ev_ebitda} | P/B: {pb_ratio}
        # Return on Equity (ROE): {roe} | Institutional Ownership: {institutional_ownership}

        # Technicals -> RSI(14): {rsi_14} | MACD: {macd} | MACD Signal: {macd_signal}
        # Moving Averages -> 50 SMA: {sma_50} | 200 SMA: {sma_200}
        # Volume -> Latest Session Volume: {recent_volume} | Avg Volume: {avg_volume}

        # --- REAL-TIME NEWS & WEB RESEARCH ---
        # {news_context}

        # Perform your analysis framework internally. In your assessment, identify the sector/industry of {ticker}. Compare it to its prominent market peers, explicitly identify any other stock in that sector that features superior fundamentals/technicals right now, and outline why it represents a better buying opportunity. Produce your structured text output according to your system rules.
        # """
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

Perform your analysis framework internally. Cross-reference {ticker} against its sector competitors using your internal knowledge base. Identify a superior peer alternative, compare them, and issue your definitive final verdict. Ensure your verdict clearly provides a exact buying price and an explicit target horizon specified in **number of days**. Produce your structured text output.
"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_role),
            ("user", user_prompt)
        ])

        chain = prompt_template | llm
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
        
        status.update(label="Research Report Dispatched!", state="complete")

    # --- PARSING TEXT RESPONSE INTO GRID COLUMNS ---
    raw_content = response.content
    
    # Safely divide the raw LLM markdown payload into individual variables
    try:
        part_1_raw = raw_content.split("PART_2:")[0].replace("PART_1:", "").strip()
        part_2_raw = raw_content.split("PART_2:")[1].split("PART_3:")[0].strip()
        part_3_raw = raw_content.split("PART_3:")[1].strip()
    except IndexError:
        # Fallback handling structure in case of token variations
        part_1_raw = "### 1. Fundamentals Analysis\n" + raw_content[:len(raw_content)//3]
        part_2_raw = "### 2. Technical Dynamics\n" + raw_content[len(raw_content)//3: 2*len(raw_content)//3]
        part_3_raw = "### 3. Risk & Final Trade Framework\n" + raw_content[2*len(raw_content)//3:]

    # Display clean wide banner
    st.markdown(f"## 🏛️ Equity Valuation Dossier: {ticker.upper()}")
    st.write("---")

    # Create Side-by-Side Card Interface (Matching Image 2 Layout)
    col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     # Orange Accent Border Bar
    #     st.markdown("<div style='border-top: 5px solid #d97706; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    #     st.markdown(part_1_raw)

    # with col2:
    #     # Teal/Green Accent Border Bar
    #     st.markdown("<div style='border-top: 5px solid #059669; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    #     st.markdown(part_2_raw)

    # with col3:
    #     # Crimson Red Accent Border Bar
    #     st.markdown("<div style='border-top: 5px solid #dc2626; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
    #     st.markdown(part_3_raw)
# Create Side-by-Side Card Interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div style='border-top: 5px solid #d97706; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(part_1_raw)

    with col2:
        st.markdown("<div style='border-top: 5px solid #059669; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(part_2_raw)

    with col3:
        # Changed the accent color to blue/indigo to emphasize "Suggestions & Trade Setup"
        st.markdown("<div style='border-top: 5px solid #4f46e5; margin-bottom: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(part_3_raw)
    st.write("---")
    st.caption(
        "This analysis is for educational and research purposes only and should not be considered financial advice. "
        "Markets involve risk, and all investment decisions should be independently verified."
    )

# --- STREAMLIT CONTROL FRONTEND ---
def main():
    st.sidebar.title("🔐 Agent Controls")
    
    # Prompt user for their token via the web UI sidebar safely
    hf_api_token = st.sidebar.text_input("Hugging Face Hub API Token", type="password")
    
    ticker_input = st.sidebar.text_input("Enter Ticker Asset (e.g., RELIANCE.NS, TSLA, AAPL)", value="RELIANCE.NS")
    trigger_analysis = st.sidebar.button("Run Research Agent", type="primary")
    
    if trigger_analysis:
        if hf_api_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_token
            run_stock_analysis_agent(ticker_input.strip())
        else:
            st.sidebar.error("Error: Please provide a valid Hugging Face Token.")

if __name__ == "__main__":
    main()