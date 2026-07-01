-- Initial database schema for Financial RAG System
-- Run with: psql -d financial_rag -f migrations/001_initial_schema.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Document registry
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename        VARCHAR(255) NOT NULL,
    status          VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
    company         VARCHAR(255),
    ticker          VARCHAR(20),
    year            INTEGER,
    quarter         VARCHAR(5),
    filing_type     VARCHAR(100),
    chunk_count     INTEGER DEFAULT 0,
    error           TEXT,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_documents_company ON documents(company);
CREATE INDEX idx_documents_year ON documents(year);
CREATE INDEX idx_documents_filing_type ON documents(filing_type);
CREATE INDEX idx_documents_status ON documents(status);

-- Structured financial metrics (for SQL agent)
CREATE TABLE IF NOT EXISTS financial_metrics (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id         UUID REFERENCES documents(id) ON DELETE CASCADE,
    company             VARCHAR(255) NOT NULL,
    ticker              VARCHAR(20),
    year                INTEGER NOT NULL,
    quarter             VARCHAR(5),
    filing_type         VARCHAR(100),
    revenue             NUMERIC(20, 4),
    gross_profit        NUMERIC(20, 4),
    operating_income    NUMERIC(20, 4),
    net_income          NUMERIC(20, 4),
    ebitda              NUMERIC(20, 4),
    total_assets        NUMERIC(20, 4),
    total_liabilities   NUMERIC(20, 4),
    shareholders_equity NUMERIC(20, 4),
    operating_cash_flow NUMERIC(20, 4),
    free_cash_flow      NUMERIC(20, 4),
    eps_basic           NUMERIC(10, 4),
    eps_diluted         NUMERIC(10, 4),
    shares_outstanding  NUMERIC(20, 4),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_metrics_company ON financial_metrics(company);
CREATE INDEX idx_metrics_year ON financial_metrics(year);
CREATE INDEX idx_metrics_ticker ON financial_metrics(ticker);
CREATE UNIQUE INDEX idx_metrics_unique ON financial_metrics(company, year, quarter, filing_type);

-- Query history for analytics
CREATE TABLE IF NOT EXISTS query_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         VARCHAR(255),
    session_id      VARCHAR(255),
    question        TEXT NOT NULL,
    answer          TEXT,
    intent          VARCHAR(100),
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    total_tokens    INTEGER,
    cost_usd        NUMERIC(10, 6),
    latency_ms      NUMERIC(10, 2),
    faithfulness    NUMERIC(5, 4),
    answer_relevancy NUMERIC(5, 4),
    cache_hit       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_query_user ON query_history(user_id);
CREATE INDEX idx_query_created ON query_history(created_at);

-- API keys (hashed)
CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash    VARCHAR(255) UNIQUE NOT NULL,
    user_id     VARCHAR(255) NOT NULL,
    role        VARCHAR(50) DEFAULT 'user',
    description VARCHAR(255),
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used   TIMESTAMP WITH TIME ZONE
);

-- Update trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
