# docu-mind — task runner (Windows / PowerShell)
# Usage: `just` or `just --list`

set shell := ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]
set dotenv-load := true
set dotenv-filename := ".env"

# ─────────────────────────────────────────────
# Default
# ─────────────────────────────────────────────
default:
    @just --list

# ─────────────────────────────────────────────
# Setup / Sync
# ─────────────────────────────────────────────

# Install/sync all dependencies (incl. dev)
sync:
    uv sync --all-groups

# Reinstall the local package (editable) — use after pyproject changes
reinstall:
    uv sync --reinstall-package docu-mind

# Full reset: nuke lock + venv, then reinstall
reset:
    Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
    Remove-Item -Force uv.lock -ErrorAction SilentlyContinue
    uv lock
    uv sync --all-groups

# ─────────────────────────────────────────────
# Lint / Format / Typecheck
# ─────────────────────────────────────────────

# Run all linters (ruff check + ruff format check + mypy)
lint:
    uv run ruff check src tests scripts
    uv run ruff format --check src tests scripts
    uv run mypy src

# Auto-fix lint issues + format
fmt:
    uv run ruff check --fix src tests scripts
    uv run ruff format src tests scripts

# Only ruff check
ruff:
    uv run ruff check src tests scripts

# Only mypy
typecheck:
    uv run mypy src

# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

# Run unit tests (fast, no external deps)
test:
    uv run pytest tests/unit -v

# Run all tests
test-all:
    uv run pytest -v

# Run with coverage report (HTML)
cov:
    uv run pytest --cov=docu_mind --cov-report=term-missing --cov-report=html
    Write-Host "-> open htmlcov/index.html"

# Run integration tests (needs Postgres up)
test-int:
    uv run pytest tests/integration -v -m integration

# Run RAG evaluation (needs LLM API key)
test-eval:
    uv run pytest tests/eval -v -m eval

# ─────────────────────────────────────────────
# Postgres (dev container)
# ─────────────────────────────────────────────

# Start Postgres + pgvector container (detached)
up-pg:
    docker start docu-mind-pg 2>$null; if ($LASTEXITCODE -ne 0) { docker run -d --name docu-mind-pg -e POSTGRES_USER=docu_mind -e POSTGRES_PASSWORD=change_me_in_prod -e POSTGRES_DB=docu_mind -p 5432:5432 pgvector/pgvector:pg17 }

# Stop Postgres container
down-pg:
    docker stop docu-mind-pg

# Nuke Postgres container + volume (⚠️ destroys data)
nuke-pg:
    docker stop docu-mind-pg 2>$null
    docker rm docu-mind-pg 2>$null
    Write-Host "Postgres container removed."

# Tail Postgres logs
logs-pg:
    docker logs -f docu-mind-pg

# Open psql shell
psql:
    docker exec -it docu-mind-pg psql -U docu_mind -d docu_mind

# ─────────────────────────────────────────────
# Redis (dev container)
# ─────────────────────────────────────────────

# Start Redis container (detached)
up-redis:
    docker start docu-mind-redis 2>$null; if ($LASTEXITCODE -ne 0) { docker run -d --name docu-mind-redis -p 6379:6379 redis:7-alpine }

# Stop Redis container
down-redis:
    docker stop docu-mind-redis

# Nuke Redis container + volume
nuke-redis:
    docker stop docu-mind-redis 2>$null
    docker rm docu-mind-redis 2>$null
    Write-Host "Redis container removed."

# Tail Redis logs
logs-redis:
    docker logs -f docu-mind-redis

# Open redis-cli shell
redis-cli:
    docker exec -it docu-mind-redis redis-cli

# ─────────────────────────────────────────────
# Migrations (Alembic)
# ─────────────────────────────────────────────

# Create a new migration (usage: just makemigration "add events table")
makemigration name:
    uv run alembic revision --autogenerate -m "{{name}}"

# Create a blank migration (no autogenerate)
makemigration-blank name:
    uv run alembic revision -m "{{name}}"

# Apply all migrations
migrate:
    uv run alembic upgrade head

# Downgrade one migration
downgrade:
    uv run alembic downgrade -1

# Downgrade to base (⚠️ destroys all tables)
migrate-reset:
    uv run alembic downgrade base
    uv run alembic upgrade head

# Show current migration
current:
    uv run alembic current

# Show history
history:
    uv run alembic history --verbose

# ─────────────────────────────────────────────
# App (dev)
# ─────────────────────────────────────────────

# Run API with hot reload
api:
    uv run uvicorn docu_mind.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

# Run API in production-like mode (no reload)
api-prod:
    uv run uvicorn docu_mind.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4

# Open Swagger UI in browser
docs:
    Start-Process "http://localhost:8000/docs"

# Open ReDoc in browser
redoc:
    Start-Process "http://localhost:8000/redoc"

# ─────────────────────────────────────────────
# Scripts
# ─────────────────────────────────────────────

# Seed the DB with sample documents
seed:
    uv run python -m scripts.seed

# Ingest a sample file (usage: just ingest "path/to/file.pdf")
ingest path:
    uv run python -m scripts.ingest_sample "{{path}}"

# Run RAG evaluation suite
evaluate:
    uv run python -m scripts.evaluate

# ─────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────

# Clean caches and build artifacts
clean:
    Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force .mypy_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force .ruff_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force htmlcov -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force .coverage -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force coverage.xml -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ─────────────────────────────────────────────
# Shortcuts
# ─────────────────────────────────────────────

# Alias: lint + test
ci: lint test-all

# Alias: full dev startup (pg + api)
dev: up-pg api

# Start both Postgres + Redis
up-infra: up-pg up-redis

# Stop both
down-infra: down-pg down-redis
