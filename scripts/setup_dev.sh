#!/bin/bash
# One-shot development environment setup

set -e

echo "=== Financial RAG System - Dev Setup ==="

# 1. Copy env file
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — fill in your API keys"
fi

# 2. Create virtualenv
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov ruff black mypy

# 4. Start infrastructure (Qdrant, Postgres, Redis)
docker compose -f docker/docker-compose.yml up -d qdrant postgres redis

echo "Waiting for services to start..."
sleep 10

# 5. Run DB migrations
psql -h localhost -U postgres -d financial_rag -f database/migrations/001_initial_schema.sql

echo ""
echo "=== Setup complete ==="
echo "Run: uvicorn backend.main:app --reload"
echo "Frontend: streamlit run frontend/app.py"
