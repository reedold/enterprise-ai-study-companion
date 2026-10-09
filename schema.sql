-- =====================================================================
-- AI-Powered B2B Knowledge & Exam Preparation Engine
-- Supabase / PostgreSQL schema  (run in Supabase SQL Editor)
-- =====================================================================

-- 1. Extensions --------------------------------------------------------
create extension if not exists vector;      -- pgvector
create extension if not exists pgcrypto;    -- gen_random_uuid()

-- 2. Chunk table -------------------------------------------------------
create table if not exists public.document_chunks (
    id          uuid primary key default gen_random_uuid(),
    subject_id  text        not null,
    topic_name  text,
    content     text        not null,
    metadata    jsonb       not null default '{}'::jsonb,
    embedding   vector(384) not null,          -- all-MiniLM-L6-v2
    created_at  timestamptz not null default now()
);

-- 3. Indexes -----------------------------------------------------------
-- HNSW index for cosine similarity (embeddings are L2-normalised).
create index if not exists document_chunks_embedding_hnsw_idx
    on public.document_chunks
    using hnsw (embedding vector_cosine_ops);

create index if not exists document_chunks_subject_idx
    on public.document_chunks (subject_id);

create index if not exists document_chunks_document_idx
    on public.document_chunks ((metadata ->> 'document_id'));

-- 4. Row Level Security ------------------------------------------------
-- The backend uses the service-role key (bypasses RLS). Enabling RLS with
-- no policies blocks direct access through the public anon key.
alter table public.document_chunks enable row level security;

-- 5. Similarity search RPC ---------------------------------------------
create or replace function public.match_document_chunks(
    query_embedding  vector(384),
    match_count      int     default 8,
    match_threshold  float   default 0.3,
    filter_subject_id text   default null,
    filter_topic_name text   default null
)
returns table (
    id          uuid,
    subject_id  text,
    topic_name  text,
    content     text,
    metadata    jsonb,
    similarity  float
)
language sql
stable
as $$
    select
        dc.id,
        dc.subject_id,
        dc.topic_name,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks dc
    where (filter_subject_id is null or dc.subject_id = filter_subject_id)
      and (filter_topic_name is null or dc.topic_name ilike filter_topic_name)
      and 1 - (dc.embedding <=> query_embedding) >= match_threshold
    order by dc.embedding <=> query_embedding
    limit least(greatest(match_count, 1), 50);
$$;
