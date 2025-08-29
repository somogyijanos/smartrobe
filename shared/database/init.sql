-- Database initialization script for Smartrobe
-- Creates necessary extensions and initial schema

-- Enable UUID extension for PostgreSQL
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for performance optimization
-- (Tables are created automatically by SQLAlchemy)

-- Note: The actual table creation is handled by SQLAlchemy migrations
-- This script only sets up database-level configurations

-- Set timezone to UTC for consistency
SET timezone = 'UTC';

-- Log the successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Smartrobe database initialized successfully at %', NOW();
END $$;
