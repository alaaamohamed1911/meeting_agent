# Meeting → Action Agent · Streamlit deployment

Adapted from your uploaded `Meeting_Action_Agent_Refactored.ipynb`.

## Run locally

1. Use Python 3.11 or 3.12.
2. `pip install -r requirements.txt`
3. Set environment variables `GEMINI_API_KEY` and `APP_PASSWORD`, or create `.streamlit/secrets.toml` from `secrets.example.toml`.
4. `streamlit run app.py`

## Streamlit Community Cloud

1. Upload the files in this folder to a **private GitHub repository** (the Streamlit entrypoint is `app.py`).
2. In the app's **Settings → Secrets**, enter:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   APP_PASSWORD = "a-long-private-password"
   ```
3. Deploy using Streamlit Community Cloud, selecting `app.py`.
4. Sign in using `APP_PASSWORD`, then submit a meeting; answer clarification questions and separately approve or reject each task.

### Storage warning

**Streamlit Community Cloud local files are ephemeral.** SQLite tasks and LangGraph checkpoints will **not survive every restart, redeploy, or instance replacement**. This demo defaults to `.data/` and is suitable for a single reviewer, not production persistence. For durable data, host on a service with a persistent filesystem/volume (`MEETING_AGENT_STORAGE=/data`) or migrate tasks and checkpoints to durable storage. The app password is a simple demo gate, not a multi-user authentication system.

### Notes

- This app can show tasks with `needs_review` and an optional human approval. A missing owner is intentionally *not* invented.
- The Gemini model is inherited from the notebook (`gemini-3.1-flash-lite`). Confirm your key has access to that model before deployment.
- Meeting transcripts and saved tasks can contain sensitive data: use test data for a public demo and restrict access.
- State is saved to SQLite only after user approval. The resolution loop may consume additional Gemini requests.
- Never commit `.streamlit/secrets.toml` or database files to GitHub.

## Included

- `agent.py` — extracted working agent nodes, tools, security, data models, database, LangGraph
- `app.py` — Streamlit UI with clarification → approval → saved tasks
- `requirements.txt` — Python packages
- `.streamlit/config.toml` — Streamlit defaults
- `secrets.example.toml` — sample configuration without credentials
- `.gitignore` — excludes secrets/database files
