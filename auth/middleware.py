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
    
    async def verify_token(self, request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> UserProfile:
        """Verify JWT token and return user profile"""
        # Try to get token from Authorization header
        token = None
        if credentials:
            token = credentials.credentials
        
        # If no Authorization header, try to get from cookies
        if not token:
            token = request.cookies.get("auth_token")
        
        if not token:
            raise HTTPException(
                status_code=401, 
                detail="No authentication token provided"
            )
        
        try:
            # Verify token with Supabase
            user_response = self.supabase.auth.get_user(token)
            if not user_response.user:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Create user profile
            user_profile = UserProfile(
                id=user_response.user.id,
                email=user_response.user.email,
                full_name=user_response.user.user_metadata.get("full_name", "Unknown"),
                created_at=user_response.user.created_at,
                last_sign_in=user_response.user.last_sign_in_at
            )
            
            return user_profile
            
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
    
    async def get_optional_user(self, request: Request) -> Optional[UserProfile]:
        """Get user if authenticated, None if not (for optional auth routes)"""
        try:
            return await self.verify_token(request)
        except HTTPException:
            return None

# Global instance
auth_middleware = AuthMiddleware()

# Dependencies for route protection
async def get_current_user(user: UserProfile = Depends(auth_middleware.verify_token)) -> UserProfile:
    """Dependency for routes that require authentication"""
    return user

async def get_optional_user(user: Optional[UserProfile] = Depends(auth_middleware.get_optional_user)) -> Optional[UserProfile]:
    """Dependency for routes that work with or without authentication"""
    return user

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