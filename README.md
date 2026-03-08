# Brag and Bev AI Agent

Chat with your company documents (e.g. the **dumpster diver**, products, brand, partner notes) via a RAG agent. Runs in the cloud so you don’t need much RAM locally.

## Quick start (local)

1. **Env**  
   Create a `.env` file with:
   ```bash
   GROQ_API_KEY=your_groq_api_key
   ```
   Get a key: [console.groq.com](https://console.groq.com).

2. **Index docs (if not already done)**  
   Put PDFs, DOCX, or TXT in `data/raw/` (e.g. in subfolders like `product/`, `brand/`). Then:
   ```bash
   pip install -r requirements.txt
   python index_docs.py
   ```
   This builds `data/chroma/`. If indexing fails on your machine (e.g. low RAM), you can do this step in the cloud (see below).

3. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Open the URL (e.g. http://localhost:8501) and ask about the dumpster diver or any topic in your docs.

---

## Deploy to the cloud (Render)

So you can use the agent without running anything on your own machine:

1. **Push the repo to GitHub**  
   Include **both** `data/raw/` and `data/chroma/` so the deployed app has your documents and the vector index.  
   - If you don’t have `data/chroma/` (e.g. couldn’t run `index_docs.py` locally), run it once on a machine with enough RAM, or in a [Render Shell](https://render.com/docs/shell), then commit the generated `data/chroma/` folder.

2. **Create a Render account**  
   [render.com](https://render.com) → Sign up (GitHub).

3. **New Web Service**  
   - Connect the GitHub repo.  
   - Render will use `render.yaml` if present.  
   - Otherwise set:
     - **Build:** `pip install -r requirements.txt`
     - **Start:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.

4. **Environment variable**  
   In the Render dashboard for the service: **Environment** → Add:
   - `GROQ_API_KEY` = your Groq API key.

5. **Deploy**  
   Save; Render builds and deploys. Open the service URL and chat (e.g. about the dumpster diver).

---

## Project layout

- `app.py` — Streamlit RAG chat UI (Groq + Chroma).
- `index_docs.py` — Loads `data/raw/`, splits text, builds `data/chroma/`.
- `src/ingest.py` — Document loading and splitting.
- `src/retriever.py` — Chroma + HuggingFace embeddings.
- `data/raw/` — Your PDF/DOCX/TXT files.
- `data/chroma/` — Vector index (created by `index_docs.py` or during cloud build).

---

## If you hit memory limits

- **Local:** Run only `streamlit run app.py` and use an already-built `data/chroma/` (e.g. from another machine or from a cloud build).
- **Render build:** If the build fails (e.g. out of memory), build `data/chroma/` once on a machine with enough RAM (or in Render Shell), commit `data/chroma/` to the repo, and deploy again. You can also upgrade the Render plan for more memory.
