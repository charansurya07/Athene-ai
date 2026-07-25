# Athene AI — Backend

Python 3.11+ / FastAPI / LangGraph backend implementing the Multimodal 7-Agent
Research Engine described in the architecture spec, and wired to match the
Athene AI frontend's expected request/response shapes exactly.

## 1. Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(or `poetry install` if you prefer Poetry — `pyproject.toml` is included.)

## 2. Configure

```bash
cp .env.example .env
```

At minimum, set `ANTHROPIC_API_KEY`. Set `TAVILY_API_KEY` (or `SERPER_API_KEY`)
too, or the Searcher agent will run with zero results and every response will
show 0% confidence.

## 3. Run

```bash
uvicorn main:app --reload --port 8000
```

- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health

## 4. Endpoints

| Method | Path              | Purpose                                                        |
|--------|-------------------|-----------------------------------------------------------------|
| GET    | `/api/v1/health`  | Liveness check                                                  |
| POST   | `/api/v1/research`| Full 7-agent pipeline. `multipart/form-data`: `prompt`, `url`, `image`, `video`, `pdf`, `code` (all optional, at least one required). Returns `{report, confidence, inputScore, recommendedScore, recommendations, triples, sources}`. |
| POST   | `/api/v1/topic`   | Standalone topic lookup behind the frontend's "Research Analysis" section. JSON body `{"topic": "..."}`. Returns `{overview, origin, keyFacts, credibility, triples}`. |
| WS     | `/ws/voice`       | Live Voice modality — stream 16kHz mono PCM binary frames in, receive `{"type":"partial"|"final","text":...}` JSON frames out. |

## 5. Architecture

```
research_ai_backend/
├── main.py                       # FastAPI app entry point
├── app/
│   ├── agents/                   # the 7 async pipeline stages
│   │   ├── ingestion_agent.py        # Stage 1
│   │   ├── planner_agent.py          # Stage 2 (backend-only — no dedicated frontend UI stage)
│   │   ├── searcher_agent.py         # Stage 3
│   │   ├── verifier_agent.py         # Stage 4
│   │   ├── recommendation_agent.py   # Stage 5
│   │   ├── knowledge_graph_agent.py  # Stage 6
│   │   └── writer_agent.py           # Stage 7
│   ├── models/                   # Pydantic schemas + shared LangGraph state
│   ├── tools/                    # search, scraping, code sandbox
│   ├── services/                 # multimodal parsing, speech, orchestrator (LangGraph graph)
│   ├── api/                      # REST + WebSocket routes
│   └── config/                   # environment settings
└── tests/
```

The **Planner agent runs on every request** exactly like the other six — it
is only left out of the frontend's visible pipeline UI, not out of the actual
execution graph:

```
ingestion → planner → searcher → verifier → recommendation → knowledge_graph → writer
```

## 6. Notes on optional dependencies

- **OCR / video / speech** (`easyocr`, `opencv-python-headless`, `openai-whisper`)
  are heavy, CPU/GPU-hungry libraries. They're wired in behind
  `app/services/multimodal_service.py` and `speech_service.py` so you can swap
  them for hosted equivalents (e.g. Google Vision OCR, Deepgram, AssemblyAI)
  without touching any agent code.
- **Vector store / Neo4j** are optional — the Searcher and Knowledge Graph
  agents both degrade gracefully to web-search-only / NetworkX-only behavior
  if `QDRANT_URL` or `NEO4J_URI` aren't reachable.
- **CORS**: update `CORS_ALLOW_ORIGINS` in `.env` to match wherever the Athene
  AI frontend is actually served from.

## 7. Tests

```bash
pytest
```
