# AI Perfume Lab

A responsive Vietnamese educational perfume-concept web app with four AI features:
- Formula Studio
- Scent Atelier
- Memory-to-Scent
- What-if Lab

## Run locally

1. Install Python 3.11+.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Set your API key:
   - Windows PowerShell: `$env:OPENAI_API_KEY="your_api_key"`
   - macOS/Linux: `export OPENAI_API_KEY="your_api_key"`
4. Start:
   `uvicorn app:app --host 0.0.0.0 --port 8000`
5. Open `http://localhost:8000`.

Optional environment variable: `OPENAI_MODEL` (defaults to `gpt-4.1-mini`).

## Deploy on Render

- Create a **Web Service** from this folder/repository.
- Runtime: Python.
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Add `OPENAI_API_KEY` in Render → your service → Environment.
- Optional: add `OPENAI_MODEL`.

## Important notes

- The personal library is stored in browser `localStorage`, not a user account or cloud database.
- API keys must stay on the server as environment variables; never put them in `index.html`.
- The formulas are educational concepts only, not validated formulations or safety advice.
- For production, add rate limiting, authentication if needed, logging, and a privacy notice.
