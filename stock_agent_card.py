import os
import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

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

# --- RETRY HELPER: track attempt counts so the UI can show live retry progress ---
def _log_retry_attempt(retry_state, label):
    """Called by tenacity before each sleep between retries. Surfaces progress in Streamlit."""
    attempt = retry_state.attempt_number
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    msg = f"⚠️ {label} failed on attempt {attempt} ({exc}). Retrying in {wait_time:.1f}s..."
    # st.status/st.write only work when a status container is active in this run;
    # guard with try/except so retries never crash on a UI issue.
    try:
        st.toast(msg)
    except Exception:
        pass
    print(msg)


# 1. DATA EXTRACTION TOOL: Financials & Technicals via yfinance
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: _log_retry_attempt(rs, "yfinance data fetch"),
    reraise=False,
)
def _fetch_stock_data_raw(ticker_symbol):
    """Network call wrapped in retry. Raises on empty/invalid data so tenacity retries it."""
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    # Historical data for Technical Analysis (6 months)
    hist = ticker.history(period="6mo")
    if hist.empty:
        raise ValueError(f"Empty price history returned for {ticker_symbol}")

    return ticker, info, hist


def fetch_stock_data(ticker_symbol):
    try:
        ticker, info, hist = _fetch_stock_data_raw(ticker_symbol)

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
    except RetryError as e:
        st.error(f"Error gathering yfinance data after multiple retries: {e.last_attempt.exception()}")
        return None
    except Exception as e:
        st.error(f"Error gathering yfinance data: {e}")
        return None

# 2. DATA EXTRACTION TOOL: Real-time News via DuckDuckGo
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: _log_retry_attempt(rs, "DuckDuckGo news search"),
    reraise=False,
)
def _fetch_ddgs_results(query, max_results=5):
    with DDGS() as ddgs:
        return [r for r in ddgs.text(query, max_results=max_results)]


def fetch_realtime_news(ticker_symbol):
    try:
        query = f"{ticker_symbol} stock news earnings sentiment"
        results = _fetch_ddgs_results(query, max_results=5)

        news_text = ""
        for i, res in enumerate(results, 1):
            news_text += f"[{i}] {res['title']}\nSnippet: {res['body']}\n\n"
        return news_text if news_text else "No recent news found."
    except RetryError as e:
        return f"Could not fetch live news after multiple retries: {e.last_attempt.exception()}"
    except Exception as e:
        return f"Could not fetch live news due to: {e}"


# 2b. PEER IDENTIFICATION + REAL DATA FETCH
# The LLM is only trusted to name a plausible peer ticker (stable, low-stakes knowledge).
# All actual numbers for that peer (price, fundamentals, technicals) come from yfinance,
# the same pipeline used for the primary ticker, so both are equally fresh.
def identify_peer_ticker(primary_ticker, business_summary, llm):
    """Ask the LLM for ONE peer ticker symbol only — no prices, no analysis."""
    peer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You identify publicly-traded peer/competitor companies. "
         "Respond with ONLY the peer's stock ticker symbol in the exact format used by Yahoo Finance "
         "(e.g. 'MSFT', 'RELIANCE.NS', 'TCS.NS'). No extra text, no explanation, no punctuation."),
        ("user",
         "Primary company ticker: {ticker}\n"
         "Business summary: {summary}\n\n"
         "Name ONE direct, actively-traded, same-sector competitor. "
         "Respond with just its ticker symbol.")
    ])
    chain = peer_prompt | llm
    try:
        result = chain.invoke({
            "ticker": primary_ticker,
            "summary": (business_summary or "")[:500]  # keep it short/cheap
        })
        candidate = result.content.strip().split()[0]
        # Strip stray punctuation/backticks the model might add
        candidate = candidate.strip("`'\".,")
        return candidate
    except Exception:
        return None


def fetch_peer_data(primary_ticker, business_summary, llm, max_candidates=3):
    """
    Try to resolve a peer ticker and pull REAL, current data for it via fetch_stock_data.
    Returns (peer_ticker, peer_data_dict) or (None, None) if no valid peer could be resolved.
    """
    tried = set()
    for _ in range(max_candidates):
        candidate = identify_peer_ticker(primary_ticker, business_summary, llm)
        if not candidate or candidate.upper() == primary_ticker.upper() or candidate in tried:
            continue
        tried.add(candidate)

        peer_data = fetch_stock_data(candidate)
        if peer_data:
            return candidate, peer_data
    return None, None

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

        status.write("🔧 Connecting to Qwen-72B LLM Core...")
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-72B-Instruct", 
            task="text-generation",
            temperature=0.2,
            max_new_tokens=2500,
        )
        llm = ChatHuggingFace(llm=llm_endpoint)

        status.write("🔍 Identifying sector peer and pulling its live market data...")
        peer_ticker, peer_data = fetch_peer_data(ticker.upper(), stock_data["business_summary"], llm)
        if peer_data:
            status.write(f"✅ Peer resolved: {peer_ticker} (current price: {peer_data['current_price']})")
            peer_context = f"""
Peer Ticker: {peer_ticker}
Peer Current Price: {peer_data['current_price']}
Peer Revenue Growth: {peer_data['revenue_growth']} | Peer Earnings Growth: {peer_data['earnings_growth']} | Peer Profit Margins: {peer_data['profit_margins']}
Peer Valuation -> P/E: {peer_data['pe_ratio']} | PEG: {peer_data['peg_ratio']} | EV/EBITDA: {peer_data['ev_ebitda']} | P/B: {peer_data['pb_ratio']}
Peer ROE: {peer_data['roe']} | Peer Debt/Equity: {peer_data['debt_to_equity']}
Peer Technicals -> RSI(14): {peer_data['rsi_14']} | MACD: {peer_data['macd']} | MACD Signal: {peer_data['macd_signal']}
Peer Moving Averages -> 50 SMA: {peer_data['sma_50']} | 200 SMA: {peer_data['sma_200']}
"""
        else:
            status.write("⚠️ Could not resolve a verifiable peer with live data. Peer comparison will be skipped.")
            peer_context = "No verifiable peer data could be retrieved. Do not fabricate a peer comparison — state that peer data was unavailable."

        status.write(f"🤖 Running full analysis with Qwen-72B LLM Core...")

        # Altered system role to guarantee cleanly tokenized markdown string splits

        system_role = """
You are an institutional-grade Stock Analysis and Trade Recommendation Assistant. 
Your objective is to generate a comprehensive asset research dossier divided into EXACTLY 3 sections using these precise token markers:
"PART_1: Fundamental & Business Strategy"
"PART_2: Technicals & Sentiment Analysis"
"PART_3: Risks & Complete Trade Setup"

Core Rules for Peer Comparison and Fallbacks in Part 3:
1. IF VALID PEER DATA IS AVAILABLE: Conduct a rigorous side-by-side comparison. State clearly which asset offers the superior risk-adjusted opportunity based on valuation, technicals, and growth.
2. IF PEER DATA IS UNAVAILABLE OR CONTAINING ERRORS: Explicitly state under your peer analysis heading: "TECHNICAL ISSUE: Unable to retrieve live competitor metrics from the data extraction pipeline." Do not invent or hallucinate metrics.
3. HANDLING DATA INSUFFICIENCY: If crucial primary or macro data is missing, state clearly: "INSUFFICIENT DATA AVAILABLE — Unable to issue a definitive recommendation to avoid a blind call." 
4. THE STANDALONE BUY PATHWAY: If peer data is missing due to a technical issue, but the searched stock's data is fully complete AND displays strong core fundamentals, solid technical trends (e.g., healthy moving average alignment), and a favorable macroeconomic backdrop (e.g., industry tailwinds, supportive interest rate cycle), you must issue a strong and definitive BUY verdict for the searched stock.

Conclude Part 3 with a markdown block titled "### 🏛️ FINAL INVESTMENT VERDICT" structured exactly like this:
- **Verdict**: [BUY <TICKER> / WAIT / INSUFFICIENT DATA]
- **Core Thesis**: [1-2 sentences explaining the decision based on fundamental, technical, macro economic indicators, or data availability]
- **BUY**: Optimal Buy Price: [Price], Target Price: [Price], Estimated Horizon: [X Days]
- **WAIT/INSUFFICIENT DATA**: Watch-List Trigger / Next Steps: [Specific condition or missing element to re-evaluate], Re-Evaluate In: [X Days]

Conclude each section with a short paragraph titled "Historical Cycle Precedent:" before moving to the next section token.
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

--- LIVE PEER/COMPETITOR DATA ---
{peer_context}

Execution Framework Instructions:
1. Examine the 'LIVE PEER/COMPETITOR DATA' section. If it indicates data is unavailable or contains fallback text, explicitly output that a technical issue prevented peer metric collection. 
2. If peer data is present, evaluate both stocks and pick the absolute best candidate.
3. If peer data is missing but the primary ticker has stellar metrics, high institutional ownership, positive macro momentum, and clean technicals, do not default to wait. Issue a clear BUY verdict for {ticker}.
4. If essential indicators are listed as 'N/A' or missing entirely, do not issue a blind call. State that data is insufficient.
5. If a BUY verdict is reached, ensure the optimal entry price, target price, and horizon in **number of days** are mathematically aligned with the provided market figures. Produce your structured text output.
"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_role),
            ("user", user_prompt)
        ])

        chain = prompt_template | llm

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=3, max=25),
            retry=retry_if_exception_type(Exception),
            before_sleep=lambda rs: _log_retry_attempt(rs, "LLM request"),
            reraise=False,
        )
        def _invoke_llm(payload):
            return chain.invoke(payload)

        try:
            response = _invoke_llm({
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
                "news_context": news_data,
                "peer_context": peer_context
            })
        except RetryError as e:
            status.update(label="LLM Request Failed After Retries", state="error")
            st.error(f"The model failed to respond after multiple attempts: {e.last_attempt.exception()}")
            return
        except Exception as e:
            status.update(label="LLM Request Failed", state="error")
            st.error(f"The model failed to respond: {e}")
            return

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