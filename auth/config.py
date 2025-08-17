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
        
        # Load environment variables with better error handling
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        
        print(f"🔧 Supabase Config Debug:")
        print(f"   SUPABASE_URL: {'✅ Set' if self.supabase_url else '❌ Not set'}")
        print(f"   SUPABASE_ANON_KEY: {'✅ Set' if self.supabase_anon_key else '❌ Not set'}")
        print(f"   SUPABASE_JWT_SECRET: {'✅ Set' if self.supabase_jwt_secret else '❌ Not set'}")
        # SECURITY: Never log actual credential content
        
        # Check for required environment variables
        missing_vars = []
        if not self.supabase_url:
            missing_vars.append("SUPABASE_URL")
        if not self.supabase_anon_key:
            missing_vars.append("SUPABASE_ANON_KEY")
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Validate Supabase URL format
        if not self.supabase_url.startswith(('http://', 'https://')):
            raise ValueError("SUPABASE_URL must be a valid HTTP/HTTPS URL")
        
        # Validate API key format (should be a long string)
        if len(self.supabase_anon_key) < 20:
            raise ValueError("SUPABASE_ANON_KEY appears to be invalid (too short)")
        
        try:
            print(f"🔧 Attempting to create Supabase client...")
            print(f"   URL: {self.supabase_url}")
            print(f"   Key length: {len(self.supabase_anon_key)}")
            print(f"   Key starts with: {self.supabase_anon_key[:20]}...")
            print(f"   Key ends with: ...{self.supabase_anon_key[-8:]}")
            
            self.client: Client = create_client(self.supabase_url, self.supabase_anon_key)
            print(f"✅ Supabase client created successfully")
        except Exception as e:
            print(f"❌ Failed to create Supabase client: {str(e)}")
            print(f"❌ Exception type: {type(e).__name__}")
            print(f"❌ Full error details: {repr(e)}")
            raise ValueError(f"Failed to create Supabase client: {str(e)}")
    
    def get_client(self) -> Client:
        """Get the Supabase client instance"""
        return self.client
    
    def get_jwt_secret(self) -> str:
        """Get the JWT secret for token validation"""
        return self.supabase_jwt_secret or ""

# Global instance - lazy initialization
supabase_config = None

def get_supabase_config():
    """Get or create the Supabase config instance"""
    global supabase_config
    if supabase_config is None:
        try:
            print(f"🔧 Creating new Supabase config instance...")
            supabase_config = SupabaseConfig()
            print(f"✅ Supabase config instance created successfully")
        except Exception as e:
            print(f"❌ Failed to create Supabase config: {str(e)}")
            print(f"🔧 Environment variables available:")
            print(f"   SUPABASE_URL: {'✅ Set' if os.getenv('SUPABASE_URL') else '❌ Not set'}")
            print(f"   SUPABASE_ANON_KEY: {'✅ Set' if os.getenv('SUPABASE_ANON_KEY') else '❌ Not set'}")
            print(f"   SUPABASE_JWT_SECRET: {'✅ Set' if os.getenv('SUPABASE_JWT_SECRET') else '❌ Not set'}")
            raise
    return supabase_config 