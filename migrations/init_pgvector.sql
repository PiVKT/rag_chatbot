-- Tạo extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tạo index cho vector search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Tạo index cho các trường thường dùng
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at);