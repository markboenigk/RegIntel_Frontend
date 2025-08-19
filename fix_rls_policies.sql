-- Fix RLS Policies for user_queries table
-- Run this in your Supabase SQL editor to fix the authentication issue

-- First, let's check the current policies
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check 
FROM pg_policies 
WHERE tablename = 'user_queries';

-- Drop existing policies
DROP POLICY IF EXISTS "Users can view own queries" ON public.user_queries;
DROP POLICY IF EXISTS "Users can insert own queries" ON public.user_queries;
DROP POLICY IF EXISTS "Users can update own queries" ON public.user_queries;
DROP POLICY IF EXISTS "Users can delete own queries" ON public.user_queries;

-- Create new, simplified policies that should work
CREATE POLICY "Users can view own queries" ON public.user_queries
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own queries" ON public.user_queries
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own queries" ON public.user_queries
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own queries" ON public.user_queries
    FOR DELETE USING (auth.uid() = user_id);

-- Verify the policies were created
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check 
FROM pg_policies 
WHERE tablename = 'user_queries';

-- Test the policies by checking if they're working
-- This should return the current user's ID
SELECT auth.uid() as current_user_id; 