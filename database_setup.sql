-- Database setup for RegIntel Frontend
-- Run this in your Supabase SQL editor

-- Create user_queries table for storing user search history
CREATE TABLE IF NOT EXISTS public.user_queries (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    response_length INTEGER,
    sources_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_queries_user_id ON user_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_user_queries_timestamp ON user_queries(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_user_queries_collection ON user_queries(collection_name);

-- Enable Row Level Security (RLS)
ALTER TABLE public.user_queries ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Users can only see their own queries
CREATE POLICY "Users can view own queries" ON public.user_queries
    FOR SELECT USING (auth.uid() = user_id);

-- Users can only insert their own queries
CREATE POLICY "Users can insert own queries" ON public.user_queries
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Users can only update their own queries
CREATE POLICY "Users can update own queries" ON public.user_queries
    FOR UPDATE USING (auth.uid() = user_id);

-- Users can only delete their own queries
CREATE POLICY "Users can delete own queries" ON public.user_queries
    FOR DELETE USING (auth.uid() = user_id);

-- Grant permissions to authenticated users
GRANT ALL ON public.user_queries TO authenticated;

-- Create a function to get user query statistics
CREATE OR REPLACE FUNCTION get_user_query_stats(user_uuid UUID)
RETURNS TABLE(
    total_queries BIGINT,
    rss_feeds_count BIGINT,
    fda_warnings_count BIGINT,
    last_query_date TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_queries,
        COUNT(*) FILTER (WHERE collection_name = 'rss_feeds')::BIGINT as rss_feeds_count,
        COUNT(*) FILTER (WHERE collection_name = 'fda_warning_letters')::BIGINT as fda_warnings_count,
        MAX(timestamp) as last_query_date
    FROM public.user_queries 
    WHERE user_id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission on the function
GRANT EXECUTE ON FUNCTION get_user_query_stats(UUID) TO authenticated; 