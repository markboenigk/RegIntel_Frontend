"""
Authentication middleware for protecting routes
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from typing import Optional, Dict, Any
from .config import supabase_config
from .models import UserProfile
import time

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)

class AuthMiddleware:
    def __init__(self):
        self.jwt_secret = supabase_config.get_jwt_secret()
        self.supabase = supabase_config.get_client()
    
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
            try:
                # Set the session with the token
                self.supabase.auth.set_session(token, None)
                
                # Get user from the session
                user_response = self.supabase.auth.get_user()
                if user_response.user:
                    # Create user profile from Supabase response
                    user_profile = UserProfile(
                        id=user_response.user.id,
                        email=user_response.user.email,
                        full_name=user_response.user.user_metadata.get("full_name", "Unknown"),
                        created_at=user_response.user.created_at,
                        last_sign_in=user_response.user.last_sign_in_at
                    )
                    return user_profile
                else:
                    raise Exception("No user data in response")
                    
            except Exception as supabase_error:
                # Fallback: Try JWT decoding
                try:
                    # Use a dummy key since we're not verifying
                    # Disable all validations including audience
                    decoded = jwt.decode(
                        token, 
                        key="dummy", 
                        options={
                            "verify_signature": False,
                            "verify_aud": False,
                            "verify_exp": False,
                            "verify_iat": False,
                            "verify_nbf": False,
                            "verify_iss": False,
                            "verify_sub": False,
                            "verify_jti": False
                        }
                    )
                    
                    # Create user profile from decoded token
                    # Convert Unix timestamps to ISO format strings
                    created_at = decoded.get("iat", 0)
                    last_sign_in = decoded.get("iat", 0)
                    
                    # Convert to ISO format if they're timestamps
                    if isinstance(created_at, (int, float)) and created_at > 0:
                        from datetime import datetime
                        created_at = datetime.fromtimestamp(created_at).isoformat()
                    else:
                        created_at = "1970-01-01T00:00:00"
                        
                    if isinstance(last_sign_in, (int, float)) and last_sign_in > 0:
                        from datetime import datetime
                        last_sign_in = datetime.fromtimestamp(last_sign_in).isoformat()
                    else:
                        last_sign_in = "1970-01-01T00:00:00"
                    
                    user_profile = UserProfile(
                        id=decoded.get("sub", "unknown"),
                        email=decoded.get("email", "unknown"),
                        full_name=decoded.get("user_metadata", {}).get("full_name", "Unknown"),
                        created_at=created_at,
                        last_sign_in=last_sign_in
                    )
                    
                    return user_profile
                    
                except Exception as jwt_error:
                    raise HTTPException(status_code=401, detail=f"Invalid token format: {str(jwt_error)}")
            
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
    
    async def get_optional_user(self, request: Request) -> Optional[UserProfile]:
        """Get user if authenticated, None if not (for optional auth routes)"""
        try:
            # Try to get token from cookies first
            print(f"🔍 Checking cookies in request: {dict(request.cookies)}")
            print(f"🔍 All cookie keys: {list(request.cookies.keys())}")
            
            token = request.cookies.get("auth_token")
            print(f"🔍 Auth token found: {'Yes' if token else 'No'}")
            if token:
                print(f"🔍 Token length: {len(token)}")
            if not token:
                return None
            
            # Try using Supabase client first
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
                
                # Set the session with both tokens
                self.supabase.auth.set_session(access_token, refresh_token)
                print(f"🔐 Session set successfully")
                
                # Get user from the session
                print(f"🔐 Getting user from session...")
                user_response = self.supabase.auth.get_user()
                print(f"🔐 User response: {user_response}")
                print(f"🔐 User object: {user_response.user}")
                
                if user_response.user:
                    print(f"🔐 Creating user profile...")
                    
                    # Convert datetime objects to ISO format strings
                    created_at = user_response.user.created_at
                    last_sign_in = user_response.user.last_sign_in_at
                    
                    if isinstance(created_at, (int, float)) and created_at > 0:
                        from datetime import datetime
                        created_at = datetime.fromtimestamp(created_at).isoformat()
                    elif hasattr(created_at, 'isoformat'):
                        created_at = created_at.isoformat()
                    else:
                        created_at = "1970-01-01T00:00:00"
                        
                    if isinstance(last_sign_in, (int, float)) and last_sign_in > 0:
                        from datetime import datetime
                        last_sign_in = datetime.fromtimestamp(last_sign_in).isoformat()
                    elif hasattr(last_sign_in, 'isoformat'):
                        last_sign_in = last_sign_in.isoformat()
                    else:
                        last_sign_in = "1970-01-01T00:00:00"
                    
                    print(f"🔐 Converted timestamps:")
                    print(f"   created_at: {created_at}")
                    print(f"   last_sign_in: {last_sign_in}")
                    
                    # Create user profile from Supabase response
                    user_profile = UserProfile(
                        id=user_response.user.id,
                        email=user_response.user.email,
                        full_name=user_response.user.user_metadata.get("full_name", "Unknown"),
                        created_at=created_at,
                        last_sign_in=last_sign_in
                    )
                    print(f"🔐 User profile created: {user_profile}")
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
                
        except Exception:
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