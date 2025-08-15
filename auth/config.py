"""
Supabase configuration and client setup
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseConfig:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        
        if not self.supabase_url or not self.supabase_anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_anon_key)
    
    def get_client(self) -> Client:
        """Get the Supabase client instance"""
        return self.client
    
    def get_jwt_secret(self) -> str:
        """Get the JWT secret for token validation"""
        return self.supabase_jwt_secret or ""

# Global instance
supabase_config = SupabaseConfig() 