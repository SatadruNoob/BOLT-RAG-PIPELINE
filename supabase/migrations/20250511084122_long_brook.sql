/*
  # Create documents and embeddings tables

  1. New Tables
    - `documents`
      - `id` (uuid, primary key)
      - `content` (text, document content)
      - `metadata` (jsonb, document metadata)
      - `created_at` (timestamp)
    - `document_embeddings`
      - `id` (uuid, primary key)
      - `document_id` (uuid, foreign key)
      - `embedding` (vector(384), document embedding)
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on both tables
    - Add policies for authenticated users
*/

-- Enable pgvector extension
create extension if not exists vector;

-- Documents table
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz default now()
);

-- Document embeddings table
create table if not exists document_embeddings (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  embedding vector(384) not null,
  created_at timestamptz default now()
);

-- Enable RLS
alter table documents enable row level security;
alter table document_embeddings enable row level security;

-- Policies for documents
create policy "Users can insert documents"
  on documents
  for insert
  to authenticated
  with check (true);

create policy "Users can read documents"
  on documents
  for select
  to authenticated
  using (true);

-- Policies for embeddings
create policy "Users can insert embeddings"
  on document_embeddings
  for insert
  to authenticated
  with check (true);

create policy "Users can read embeddings"
  on document_embeddings
  for select
  to authenticated
  using (true);

-- Create indexes
create index on document_embeddings using ivfflat (embedding vector_cosine_ops);
create index on document_embeddings (document_id);