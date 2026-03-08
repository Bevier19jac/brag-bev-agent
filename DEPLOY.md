# Deploy Brag & Bev AI Agent on Render

## 1. Prepare the repo

- Ensure `data/`, `data/raw/`, and `data/chroma/` exist (the app creates them if missing).
- Commit and push to GitHub (include `app.py`, `requirements.txt`, and optionally existing `data/` so Chroma is pre-populated).

## 2. Create the Web Service on Render

1. Go to [dashboard.render.com](https://dashboard.render.com) and sign in (GitHub).
2. **New** → **Web Service**.
3. Connect the GitHub repo that contains this project.
4. Configure:
   - **Name:** `brag-bev-agent` (or any name).
   - **Region:** Choose one (e.g. Oregon).
   - **Branch:** `main` (or your default).
   - **Runtime:** `Python 3`.
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
     ```
   - **Instance type:** Free (or upgrade if you need more memory for sentence-transformers).

5. **Environment:**
   - Add variable: **Key** `GROQ_API_KEY`, **Value** your Groq API key (from [console.groq.com](https://console.groq.com)).

6. Click **Create Web Service**. Render will build and deploy.

## 3. Exact deploy commands (summary)

| Step        | Command / action |
|------------|-------------------|
| Build      | `pip install -r requirements.txt` |
| Start      | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` |
| Env var    | `GROQ_API_KEY` = your Groq key |

Render sets `PORT` automatically; the start command uses it so Streamlit listens on the right port.

## 4. Optional: render.yaml

If you use a **Blueprint** (e.g. `render.yaml` in the repo), the service can look like:

```yaml
services:
  - type: web
    name: brag-bev-agent
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: GROQ_API_KEY
        sync: false
```

Set `GROQ_API_KEY` in the Render dashboard (Environment) when `sync: false`.

---

# Quick test procedure

## Local test (ingestion + retrieval)

1. **Install and run**
   ```bash
   cd brag-bev-agent
   pip install -r requirements.txt
   ```
   Create a `.env` file with:
   ```
   GROQ_API_KEY=your_key_here
   ```
   Then:
   ```bash
   streamlit run app.py
   ```

2. **Ingestion**
   - Open the app in the browser.
   - In the sidebar, upload a small PDF, TXT, or DOCX.
   - Click **Ingest uploaded files**.
   - Confirm you see a success message and a chunk count.

3. **Retrieval**
   - In the main area, type a question that should be answered by the uploaded file (e.g. “What is this document about?”).
   - Click **Run AI**.
   - Confirm you get an answer and that “Retrieved context” shows chunks from your file.

4. **Error handling**
   - Clear `GROQ_API_KEY` from `.env`, restart the app, and run a question again — you should see a clear error about `GROQ_API_KEY`.
   - Upload a file with an unsupported extension (e.g. `.xyz`) and try to ingest — you should see an “Unsupported file type” message.

## After deploying on Render

1. Open the service URL from the Render dashboard.
2. Upload a test file and click **Ingest uploaded files**.
3. Ask a question and click **Run AI**.
4. Confirm the answer and that Chroma persistence works: ask another question or refresh the page and query again (no need to re-ingest).

If the free instance runs out of memory during build or startup (e.g. when loading sentence-transformers), use a paid instance with more RAM or pre-build Chroma locally and commit `data/chroma/` so the app only loads the existing DB.
