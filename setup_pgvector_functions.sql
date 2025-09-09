-- Setup pgvector functions for RAG search
-- Run this in your Supabase SQL editor

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Function to search warning letters using pgvector
CREATE OR REPLACE FUNCTION search_warning_letters(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id int,
    chunk_id text,
    warning_letter_id text,
    text_content text,
    text_vector vector(1536),
    company_name text,
    letter_date date,
    chunk_type text,
    chunk_index int,
    total_chunks int,
    source_file text,
    violations jsonb,
    required_actions jsonb,
    systemic_issues jsonb,
    regulatory_consequences jsonb,
    created_at timestamptz,
    updated_at timestamptz,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        wlv.id,
        wlv.chunk_id,
        wlv.warning_letter_id,
        wlv.text_content,
        wlv.text_vector,
        wlv.company_name,
        wlv.letter_date,
        wlv.chunk_type,
        wlv.chunk_index,
        wlv.total_chunks,
        wlv.source_file,
        wlv.violations,
        wlv.required_actions,
        wlv.systemic_issues,
        wlv.regulatory_consequences,
        wlv.created_at,
        wlv.updated_at,
        1 - (wlv.text_vector <=> query_embedding) as similarity
    FROM public.warning_letters_vectors wlv
    WHERE 1 - (wlv.text_vector <=> query_embedding) > match_threshold
    ORDER BY wlv.text_vector <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function to search RSS feeds using pgvector
CREATE OR REPLACE FUNCTION search_rss_feeds(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.3,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id int,
    chunk_id text,
    article_id text,
    text_content text,
    text_vector vector(1536),
    article_title text,
    published_date date,
    feed_name text,
    author text,
    article_link text,
    chunk_type text,
    chunk_index int,
    total_chunks int,
    text_length int,
    estimated_tokens int,
    companies jsonb,
    products jsonb,
    regulations jsonb,
    regulatory_bodies jsonb,
    people jsonb,
    locations jsonb,
    dates jsonb,
    summary text,
    article_tags jsonb,
    total_entities int,
    created_at timestamptz,
    updated_at timestamptz,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rf.id,
        rf.chunk_id,
        rf.article_id,
        rf.text_content,
        rf.text_vector,
        rf.article_title,
        rf.published_date,
        rf.feed_name,
        rf.author,
        rf.article_link,
        rf.chunk_type,
        rf.chunk_index,
        rf.total_chunks,
        rf.text_length,
        rf.estimated_tokens,
        rf.companies,
        rf.products,
        rf.regulations,
        rf.regulatory_bodies,
        rf.people,
        rf.locations,
        rf.dates,
        rf.summary,
        rf.article_tags,
        rf.total_entities,
        rf.created_at,
        rf.updated_at,
        1 - (rf.text_vector <=> query_embedding) as similarity
    FROM public.rss_feeds rf
    WHERE 1 - (rf.text_vector <=> query_embedding) > match_threshold
    ORDER BY rf.text_vector <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS warning_letters_vectors_text_vector_idx 
ON public.warning_letters_vectors USING hnsw (text_vector vector_cosine_ops);

CREATE INDEX IF NOT EXISTS rss_feeds_text_vector_idx 
ON public.rss_feeds USING hnsw (text_vector vector_cosine_ops);

-- Function to get weekly top action categories
CREATE OR REPLACE FUNCTION get_weekly_top_actions()
RETURNS TABLE (
    week_num int,
    action_category text,
    action_count bigint
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH weekly_counts AS (
        SELECT
            EXTRACT(WEEK FROM letter_date) AS week_num,
            action_category,
            COUNT(*) AS action_count
        FROM public.actions_analytics
        GROUP BY EXTRACT(WEEK FROM letter_date), action_category
    ),
    ranked AS (
        SELECT
            week_num,
            action_category,
            action_count,
            ROW_NUMBER() OVER (
                PARTITION BY week_num
                ORDER BY action_count DESC
            ) AS rn
        FROM weekly_counts
    )
    SELECT
        week_num::int,
        action_category,
        action_count
    FROM ranked
    WHERE rn <= 3
    ORDER BY week_num DESC, action_count DESC;
END;
$$;

-- Function for direct SQL execution (for fallback vector search)
CREATE OR REPLACE FUNCTION exec_sql(query text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result json;
BEGIN
    EXECUTE query INTO result;
    RETURN result;
END;
$$;

-- Grant necessary permissions
GRANT EXECUTE ON FUNCTION search_warning_letters TO authenticated;
GRANT EXECUTE ON FUNCTION search_rss_feeds TO authenticated;
GRANT EXECUTE ON FUNCTION get_weekly_top_actions TO authenticated;
GRANT EXECUTE ON FUNCTION exec_sql TO authenticated;
