"""
Supabase configuration and client setup
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

class SupabaseConfig:
    def __init__(self):
        # Ensure .env is loaded
        load_dotenv()
        
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        
        print(f"🔧 Supabase Config Debug:")
        print(f"   SUPABASE_URL: {'✅ Set' if self.supabase_url else '❌ Not set'}")
        print(f"   SUPABASE_ANON_KEY: {'✅ Set' if self.supabase_anon_key else '❌ Not set'}")
        print(f"   SUPABASE_JWT_SECRET: {'✅ Set' if self.supabase_jwt_secret else '❌ Not set'}")
        # SECURITY: Never log actual credential content
        
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