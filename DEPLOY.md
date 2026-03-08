# Deploy Brag & Bev AI Agent on Render (query-only)

Query-only RAG: prebuilt `data/chroma` only. No uploads, no ingestion.

---

## 1. Exact git commands to commit and deploy

```bash
cd brag-bev-agent
git add app.py requirements.txt render.yaml .python-version runtime.txt data/chroma data/raw
git status   # ensure data/chroma is included and no .env
git commit -m "Query-only RAG: prebuilt Chroma, no uploads"
git push origin main
```

Then in Render: connect the repo, set `GROQ_API_KEY` in Environment, deploy.

**Important:** `data/chroma` must be in the repo. If you haven’t built it yet, run indexing locally (e.g. `python index_docs.py`), then commit `data/chroma` and push.

---

## 2. Render settings to verify

| Setting | Value |
|--------|--------|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` |
| **Environment** | `GROQ_API_KEY` = your Groq API key |
| **Python** | 3.11 (`.python-version` or `PYTHON_VERSION=3.11.9`) |

---

## 3. Exact test steps

### Step 1: Local smoke test

1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. Open `http://localhost:8501/?smoke=1`
4. Expect: **"Smoke test OK — Streamlit is running."** and Python/Streamlit versions.

### Step 2: Local query (needs data/chroma)

1. Open `http://localhost:8501` (no `?smoke=1`)
2. If you see **"No prebuilt vector database found in data/chroma"**: build the DB locally (`python index_docs.py`), then restart the app.
3. Enter a question (e.g. "What is the dumpster diver?") and click **Run AI**.
4. Expect: an answer and "Retrieved context" with chunks.

### Step 3: Render smoke test

1. Deploy to Render. Wait for "Live".
2. Open `https://<your-service>.onrender.com/?smoke=1`
3. Expect: same smoke message. If blank, check start command and logs.

### Step 4: Render query

1. Open `https://<your-service>.onrender.com` (no query param)
2. If you see **"No prebuilt vector database found in data/chroma"**: ensure `data/chroma` is committed and redeploy.
3. Enter a question and click **Run AI**.
4. Expect: answer and retrieved chunks. First Run AI may be slower (embeddings load).

### Step 5: If Run AI fails

- **GROQ_API_KEY** message → set in Render Dashboard → Environment.
- **Vectorstore / Retrieval error** → check Render logs; ensure `data/chroma` was built with the same embedding model (`sentence-transformers/all-MiniLM-L6-v2`).

---

## 4. If blank page still happens

1. Try **?smoke=1** on the Render URL. If smoke works, the failure is after Chroma/embeddings load (first Run AI); check logs.
2. Check **Start command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
3. Check **Logs** for `ImportError`, `ModuleNotFoundError`, OOM.
4. Ensure **Python 3.11** (`.python-version` or `PYTHON_VERSION=3.11.9`).
