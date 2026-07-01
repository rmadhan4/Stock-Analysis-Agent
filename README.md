# Stock Analysis Agent

A Python-based stock analysis agent that fetches financial data via `yfinance`, computes technical indicators with `ta`, gathers news via `duckduckgo-search`, and uses LangChain-compatible LLM integrations for report generation.

## Project Files

- `stock_agent.py` - main stock analysis agent script.
- `stock_agent_card.py` - alternate agent or UI card wrapper.
- `stock_agent_gemini.py` - Gemini-specific agent variant.
- `requirements.txt` - pinned Python dependencies.
- `.env` - environment variables (not committed to GitHub via `.gitignore`).

## Setup

1. Create a virtual environment:

```bash
cd '/Users/mahalakshmisrinivasan/Madhan/Stock Analysis Agent'
python3 -m venv venv
```

2. Activate the virtual environment:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root with the keys required by the script you want to run.

Example:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_api_token_here
```

## Running

Run the main agent:

```bash
./venv/bin/python stock_agent.py
```

Or run an alternate script:

```bash
./venv/bin/python stock_agent_card.py
```

## Notes

- Keep `.env` out of source control.
- The repository has already been initialized and pushed to GitHub at `https://github.com/rmadhan4/Stock-Analysis-Agent`.
