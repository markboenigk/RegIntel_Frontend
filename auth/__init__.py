"""
Authentication package for RegIntel Frontend
"""

from .config import get_supabase_config
from .models import UserSignUp, UserSignIn, UserProfile, AuthResponse
from .middleware import AuthMiddleware, get_current_user, get_optional_user, auth_middleware
from .routes import router as auth_router

__all__ = [
    "get_supabase_config",
    "UserSignUp", 
    "UserSignIn", 
    "UserProfile", 
    "AuthResponse",
    "AuthMiddleware",
    "get_current_user",
    "get_optional_user",
    "auth_middleware",
    "auth_router"
] 