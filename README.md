# AI Perfume Lab — Sound-to-Scent backend

This starter connects the existing HTML website to a Python API that measures audio features and asks an OpenAI model to interpret those features into three fragrance concepts.

## What it really analyzes
- Duration (up to 45 seconds of audio is analyzed)
- Estimated tempo (BPM)
- RMS energy
- Spectral centroid, bandwidth, and roll-off
- Zero-crossing rate and spectral flatness

These are signal measurements. They do not prove the emotional meaning of a song or identify perfume smells. The language model produces a creative interpretation conditioned on the measured features and the user's selected mood.

## Run locally (Windows PowerShell)
1. Install Python 3.11.
2. Open this folder in VS Code or Terminal.
3. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. Set your API key in the terminal session (never put it in HTML):
   ```powershell
   $env:OPENAI_API_KEY="YOUR_API_KEY"
   $env:OPENAI_MODEL="gpt-5-mini"
   ```
   Choose a model available to your API project if that model name is unavailable.
5. Start the app:
   ```powershell
   uvicorn main:app --reload
   ```
6. Visit http://127.0.0.1:8000. Do not open `static/index.html` directly; the page calls the backend at `/api/sound-to-scent`.

## Deploy online with Render
1. Create a Git repository and upload all files in this folder (including `static/index.html`).
2. In Render, create a new Web Service from that repository.
3. Render will read `render.yaml`; if creating the service manually, use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. In the service's Environment settings, add `OPENAI_API_KEY` as a secret value. Set `OPENAI_MODEL` to a model available to your API project (default starter value: `gpt-5-mini`). Never commit the key to Git or put it in the browser code.
5. Deploy and test the `/` page by uploading a short audio file. Same-origin frontend/backend means no separate CORS setup is needed.

## Limits and production notes
- Upload max: 15 MB. Analysis window: first 45 seconds.
- For MP3/M4A/OGG decoding, the host's audio decoder libraries must support the format; if deployment fails on a codec, add FFmpeg to the host image or start with WAV.
- This is a learning prototype, not a validated perfumery or skin-safety tool. Formula percentages are normalized to 100 for display and must not be used as a real mixing or skin-application recipe.
- Before public launch, add rate limiting, authentication/abuse controls, request-size limits at the proxy, and a privacy notice. Do not log audio content or API keys.

## Endpoint
`POST /api/sound-to-scent` multipart form fields:
- `file`: audio file
- `mood`: one of `serene`, `mysterious`, `romantic`, `energetic`, `nostalgic`

Returns measured feature data, AI interpretation, mapping explanation, and three formulas.
