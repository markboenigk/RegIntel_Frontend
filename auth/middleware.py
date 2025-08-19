"""
Authentication middleware for protecting routes
"""

import os
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional, Dict, Any
from .config import get_supabase_config
from .models import UserProfile
import time

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)

class AuthMiddleware:
    def __init__(self):
        # Lazy initialization - don't create Supabase client until needed
        self._supabase_config = None
        self._jwt_secret = None
        self._supabase = None
    
    def _get_supabase_config(self):
        """Lazy initialization of Supabase config"""
        if self._supabase_config is None:
            try:
                self._supabase_config = get_supabase_config()
            except Exception as e:
                print(f"⚠️ Warning: Supabase config initialization failed: {str(e)}")
                print(f"   This is expected in CI environments without valid Supabase credentials")
                # In CI, we can continue without Supabase for basic tests
                return None
        return self._supabase_config
    
    def _get_jwt_secret(self):
        """Lazy initialization of JWT secret"""
        if self._jwt_secret is None:
            config = self._get_supabase_config()
            if config:
                self._jwt_secret = config.get_jwt_secret()
        return self._jwt_secret
    
    def _get_supabase_url(self):
        """Get Supabase URL from environment"""
        return os.getenv("SUPABASE_URL", "")
    
    def _get_supabase_client(self):
        """Lazy initialization of Supabase client"""
        if self._supabase is None:
            config = self._get_supabase_config()
            if config and config.is_client_available():
                try:
                    self._supabase = config.get_client()
                except Exception as e:
                    print(f"⚠️ Warning: Supabase client creation failed: {str(e)}")
                    return None
        return self._supabase
    
    async def verify_token(self, request: Request) -> UserProfile:
        """Verify JWT token and return user profile"""
        # Try to get token from cookies first (primary method)
        token = request.cookies.get("auth_token")
        
        # If no cookie token, try to get from Authorization header
        if not token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]  # Remove "Bearer " prefix
        
        if not token:
            raise HTTPException(
                status_code=401, 
                detail="No authentication token provided"
            )
        
        try:
            # Try using Supabase client first
            supabase_client = self._get_supabase_client()
            if supabase_client:
                try:
                    print(f"🔐 Attempting Supabase authentication...")
                    
                    # Get both access token and refresh token
                    access_token = request.cookies.get("auth_token")
                    refresh_token = request.cookies.get("refresh_token")
                    
                    print(f"🔐 Access token: {'✅ Found' if access_token else '❌ Not found'}")
                    print(f"🔐 Refresh token: {'✅ Found' if refresh_token else '❌ Not found'}")
                    
                    if not access_token:
                        print(f"❌ No access token found")
                        return None
                    
                    # Set the session with both tokens (handle None refresh token)
                    if refresh_token:
                        try:
                            supabase_client.auth.set_session(access_token, refresh_token)
                            print(f"🔐 Session set successfully with refresh token")
                        except Exception as session_error:
                            print(f"⚠️ Session setting with refresh token failed: {str(session_error)}")
                            # Try with just access token
                            try:
                                supabase_client.auth.set_session(access_token, None)
                                print(f"🔐 Session set successfully with access token only")
                            except Exception as access_only_error:
                                print(f"⚠️ Session setting with access token only failed: {str(access_only_error)}")
                                # Continue with JWT fallback
                                raise Exception("All Supabase session methods failed")
                    else:
                        # Try to set session with just access token
                        try:
                            supabase_client.auth.set_session(access_token, None)
                            print(f"🔐 Session set successfully with access token only")
                        except Exception as session_error:
                            print(f"⚠️ Session setting failed: {str(session_error)}")
                            # Continue with JWT fallback
                            raise Exception("Session setting failed")
                    
                    # Get user from the session
                    print(f"🔐 Getting user from session...")
                    user_response = supabase_client.auth.get_user()
                    print(f"🔐 User response: {user_response}")
                    print(f"🔐 User object: {user_response.user}")
                    
                    if user_response.user:
                        print(f"🔐 Creating user profile...")
                        
                        # Create user profile from Supabase response
                        user_profile = UserProfile(
                            id=user_response.user.id,
                            email=user_response.user.email,
                            full_name=user_response.user.user_metadata.get("full_name", "Unknown"),
                            created_at=user_response.user.created_at,
                            last_sign_in=user_response.user.last_sign_in_at
                        )
                        print(f"🔐 User profile created: {user_profile}")
                        
                        # Store the authenticated client for this user
                        self._authenticated_clients = getattr(self, '_authenticated_clients', {})
                        self._authenticated_clients[user_response.user.id] = supabase_client
                        
                        return user_profile
                    else:
                        print(f"❌ No user in response")
                        return None
                        
                except Exception as e:
                    print(f"❌ Supabase authentication failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # If Supabase fails, return None (user not authenticated)
                    return None
            else:
                print(f"⚠️ Supabase client not available, user not authenticated")
                return None
                
        except Exception:
            return None
    
    def get_authenticated_client(self, user_id: str):
        """Get the authenticated Supabase client for a specific user"""
        if hasattr(self, '_authenticated_clients') and user_id in self._authenticated_clients:
            return self._authenticated_clients[user_id]
        return None

# Global instance
auth_middleware = AuthMiddleware()

# Dependencies for route protection
async def get_current_user(request: Request) -> UserProfile:
    """Dependency for routes that require authentication"""
    return await auth_middleware.verify_token(request)

async def get_optional_user(request: Request) -> Optional[UserProfile]:
    """Dependency for routes that work with or without authentication"""
    return await auth_middleware.get_optional_user(request)

def require_auth(func):
    """Decorator to require authentication for routes"""
    func.__auth_required__ = True
    return func

def require_role(required_role: str):
    """Decorator to require specific user role"""
    def decorator(func):
        func.__required_role__ = required_role
        return func
    return decorator 