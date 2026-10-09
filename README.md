# MIMIR Backend

MIMIR is a FastAPI semantic-cache middleware for AI applications. This repository contains the backend API and cache engine. The Payload CMS dashboard is a separate application and is not included here.

## Repository layout

```text
.
├── src/mimir_cache/   FastAPI application and cache engine
├── tests/             Backend tests
├── Dockerfile
├── docker-compose.yml Local backend + Redis stack
├── pyproject.toml
└── README.md
```

## Run locally

From this repository's root, start the API and Redis:

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`; interactive API docs are at `http://localhost:8000/docs`. Redis is optional at runtime because the backend falls back to an in-process cache if Redis is unavailable.

To run the API directly, without Docker:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
uvicorn mimir_cache.main:app --reload
```

Copy `.env.example` to `.env` to configure optional settings. Do not commit `.env` or real API keys.

## License

Add the project's chosen `LICENSE` file at this repository root before publishing it publicly.
