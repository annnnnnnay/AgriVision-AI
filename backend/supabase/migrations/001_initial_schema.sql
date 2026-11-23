-- AgriVision AI — Supabase Database Schema
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard → SQL Editor

-- ============================================================
-- PREDICTIONS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT,
  disease TEXT NOT NULL,
  disease_id TEXT,
  crop TEXT NOT NULL,
  confidence REAL NOT NULL,
  severity TEXT,
  is_healthy BOOLEAN DEFAULT FALSE,
  language TEXT DEFAULT 'en',
  processing_time_ms REAL,
  image_url TEXT,
  top_predictions JSONB,
  disease_info JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast history queries
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);

-- Enable Row Level Security
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Authenticated users can only read and insert their own predictions.
DROP POLICY IF EXISTS "Users can view own predictions" ON predictions;
CREATE POLICY "Users can view own predictions"
  ON predictions FOR SELECT TO authenticated
  USING (user_id = (SELECT auth.uid())::text);

DROP POLICY IF EXISTS "Anyone can insert predictions" ON predictions;
DROP POLICY IF EXISTS "Users can insert own predictions" ON predictions;
CREATE POLICY "Users can insert own predictions"
  ON predictions FOR INSERT TO authenticated
  WITH CHECK (user_id = (SELECT auth.uid())::text);

-- ============================================================
-- STORAGE BUCKET
-- ============================================================
-- The backend uploads with its server-only service role key.
INSERT INTO storage.buckets (id, name, public)
VALUES ('crop-images', 'crop-images', TRUE)
ON CONFLICT (id) DO UPDATE SET public = EXCLUDED.public;

NOTIFY pgrst, 'reload schema';

-- ============================================================
-- OPTIONAL: User profiles (if not using Supabase Auth)
-- ============================================================
-- If using the in-app auth (not Supabase Auth), uncomment this:
--
-- CREATE TABLE IF NOT EXISTS user_profiles (
--   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
--   name TEXT NOT NULL,
--   email TEXT UNIQUE NOT NULL,
--   password_hash TEXT NOT NULL,
--   location TEXT,
--   created_at TIMESTAMPTZ DEFAULT NOW()
-- );
