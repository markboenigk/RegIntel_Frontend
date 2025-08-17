-- Populate RSS Feeds table with data from rss_feeds_gold
-- Run this in your Supabase SQL editor

-- First, let's check what's in the rss_feeds_gold table
SELECT 
    COUNT(*) as total_articles,
    MIN(article_published_date) as earliest_date,
    MAX(article_published_date) as latest_date
FROM public.rss_feeds_gold;

-- Now let's populate the RSS feeds table with the latest articles
-- We'll select the most recent articles based on article_published_date
INSERT INTO public.rss_feeds (
    article_summary,
    article_feed_name,
    article_published_date,
    created_at
)
SELECT 
    article_summary,
    article_feed_name,
    article_published_date,
    NOW() as created_at
FROM public.rss_feeds_gold
WHERE article_summary IS NOT NULL 
  AND article_feed_name IS NOT NULL 
  AND article_published_date IS NOT NULL
ORDER BY article_published_date DESC
LIMIT 100; -- Insert the latest 100 articles

-- Verify the data was inserted
SELECT 
    COUNT(*) as inserted_count,
    MIN(article_published_date) as earliest_date,
    MAX(article_published_date) as latest_date
FROM public.rss_feeds;

-- Show a sample of the latest articles
SELECT 
    article_feed_name,
    article_summary,
    article_published_date
FROM public.rss_feeds
ORDER BY article_published_date DESC
LIMIT 10; 