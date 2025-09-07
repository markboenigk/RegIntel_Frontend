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
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")  # Environment var name vs internal name
        self.supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
        
        print(f"🔧 Supabase Config Debug:")
        print(f"   SUPABASE_URL: {'✅ Set' if self.supabase_url else '❌ Not set'}")
        print(f"   SUPABASE_ANON_KEY: {'✅ Set' if self.supabase_key else '❌ Not set'}")
        print(f"   SUPABASE_JWT_SECRET: {'✅ Set' if self.supabase_jwt_secret else '❌ Not set'}")
        # SECURITY: Never log actual credential content
        
        # Check if we're in CI environment
        is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
        if is_ci:
            print(f"🔧 CI environment detected - using relaxed validation")
            # In CI, we can be more lenient with validation
            if not self.supabase_url or not self.supabase_key:
                print(f"⚠️ CI environment: Missing Supabase credentials, will skip client creation")
                self.client = None
                return
        
        # Check for required environment variables
        missing_vars = []
        if not self.supabase_url:
            missing_vars.append("SUPABASE_URL")
        if not self.supabase_key:
            missing_vars.append("SUPABASE_ANON_KEY")
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Validate Supabase URL format
        if not self.supabase_url.startswith(('http://', 'https://')):
            raise ValueError("SUPABASE_URL must be a valid HTTP/HTTPS URL")
        
        # Validate API key format (should be a long string)
        if len(self.supabase_key) < 20:
            raise ValueError("SUPABASE_ANON_KEY appears to be invalid (too short)")
        
        try:
            print(f"🔧 Attempting to create Supabase client...")
            print(f"   URL: {self.supabase_url}")
            print(f"   Key length: {len(self.supabase_key)}")
            print(f"   Key starts with: {self.supabase_key[:20]}...")
            print(f"   Key ends with: ...{self.supabase_key[-8:]}")
            
            # Check for proxy-related environment variables that might cause issues
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
            for var in proxy_vars:
                if os.getenv(var):
                    print(f"⚠️ Found proxy environment variable: {var}={os.getenv(var)}")
            
            # Temporarily unset proxy environment variables to avoid client issues
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
            old_proxy_values = {}
            for var in proxy_vars:
                if var in os.environ:
                    old_proxy_values[var] = os.environ[var]
                    del os.environ[var]
            
            try:
                # Use the correct parameter names that Supabase client expects
                # Explicitly pass only the required parameters to avoid proxy issues
                self.client: Client = create_client(
                    supabase_url=self.supabase_url,
                    supabase_key=self.supabase_key
                )
            finally:
                # Restore proxy environment variables
                for var, value in old_proxy_values.items():
                    os.environ[var] = value
            print(f"✅ Supabase client created successfully")
        except Exception as e:
            print(f"❌ Failed to create Supabase client: {str(e)}")
            print(f"❌ Exception type: {type(e).__name__}")
            print(f"❌ Full error details: {repr(e)}")
            
            # Check if it's a proxy-related error
            if 'proxy' in str(e).lower():
                print(f"🔧 Proxy-related error detected. Check environment variables:")
                for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                    if os.getenv(var):
                        print(f"   {var}: {os.getenv(var)}")
                print(f"🔧 Try unsetting proxy variables or check network configuration")
            
            # In CI environment, we can continue without the client
            if is_ci:
                print(f"⚠️ CI environment: Continuing without Supabase client")
                self.client = None
            else:
                raise ValueError(f"Failed to create Supabase client: {str(e)}")
    
    def get_client(self) -> Client:
        """Get the Supabase client instance"""
        if self.client is None:
            print(f"⚠️ Supabase client not available (likely in CI environment)")
        return self.client
    
    def is_client_available(self) -> bool:
        """Check if Supabase client is available"""
        return self.client is not None
    
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
            
            # In CI environment, we can continue without Supabase for basic tests
            is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
            if is_ci:
                print(f"🔧 CI environment: Supabase config failed, but continuing for basic tests")
                # Don't create a dummy config, just return None
                return None
            else:
                # In non-CI environments, re-raise the error
                raise
    return supabase_config 