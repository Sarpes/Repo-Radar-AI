# RepoRadarAI

Personal copy of the DS21 final project: GitHub repo radar. A CatBoost model scores
"abandoned vs. alive" for AI repos; a RAG chatbot (n8n + Ollama + pgvector) answers questions.
Remote: https://github.com/Sarpes/Repo-Radar-AI (personal). The team repo is a separate project; do not touch it.
Details live in `README.md` (Turkish, ~1500 lines; section index at the top).

## Layout
- `DS21_Final_v2.py` training chain, `score_today.py` live scoring, `DS21_RAG.py` RAG indexing.
- `aylik.bat` / `haftalik.bat` monthly / weekly refresh chains; `tarihce.py`, `refresh_csv.py`, `swap_readmes.py`.
- `n8n/` five workflow JSONs (import via n8n UI; credentials are NOT inside them). `tools/`, `panel_v2/`, `web/`.
- `compose.yaml` pgvector database: compose project `repo-radar-ai`, container `repo-radar-ai-pgvector`,
  image `pgvector/pgvector:pg17`, port `127.0.0.1:5433`, EXTERNAL volume `ds21_pgvector_data`.
- `docs/` handovers, architecture, guides, artifacts, archive.

## Commands
- Python venv: `.venv` from `requirements.txt` (built with Python 3.13.9, anaconda). Tooling python: `C:\Users\Sarp\anaconda3\python.exe`.
- DB config: `docker compose config` to validate; `.env` (git-ignored) holds POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB, names in `.env.example`.
- n8n reaches the database at `host.docker.internal:5433` (n8n runs in the `n8n-platform` project).
- Refresh: `haftalik.bat`, `aylik.bat` (`--kuru` = dry run).

## Data (git-ignored `data/`)
- `REPORADAR_DATA_DIR` (default `<repo>\data\ds21final`) is the folder n8n mounts as `/data`. Scripts read it from this variable.
- `data\outputs\` = model artefacts (~940 MB), `data\raw\` = Kaggle CSV + README dump. Copies; originals live in the team repo.
- Note: the pipeline scripts still read `outputs\` and `Data\` under the repo root; `data\outputs` / `data\raw` are reference copies until that is migrated.

## Never
- Never commit `.env`, `data/`, `outputs/`, `.venv/`; never print the DB password.
- Never stop or recreate the running pgvector container or touch volume `ds21_pgvector_data` outside the planned cutover.
- Never write into the live n8n data folder by hand while n8n runs.
- Never commit straight to `main`; one branch per task, PR, merge after the owner's OK.
