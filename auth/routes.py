"""
Authentication routes for user management
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from .models import UserSignUp, UserSignIn, UserProfile, AuthResponse, PasswordReset, PasswordUpdate, UserUpdate
from .config import supabase_config
from .middleware import get_current_user, get_optional_user
import json

router = APIRouter(prefix="/auth", tags=["authentication"])
templates = Jinja2Templates(directory="templates")

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    # Check if template exists, if not return fallback
    try:
        # Try to get template info to see if it exists
        template = templates.get_template("auth/login.html")
        return templates.TemplateResponse("auth/login.html", {"request": request})
    except Exception:
        # Fallback to simple response if template doesn't exist
        return HTMLResponse("""
        <html>
            <head><title>Login - RegIntel</title></head>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px;">
                <h1>🔐 Login</h1>
                <p>Login page is working! Template will be added later.</p>
                <hr>
                <p><strong>Status:</strong> ✅ Authentication system is running</p>
                <p><strong>Next:</strong> Beautiful login form will be added</p>
                <a href="/" style="color: #007bff;">← Back to Main App</a>
            </body>
        </html>
        """)

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Serve the registration page"""
    # Check if template exists, if not return fallback
    try:
        # Try to get template info to see if it exists
        template = templates.get_template("auth/register.html")
        return templates.TemplateResponse("auth/register.html", {"request": request})
    except Exception:
        # Fallback to simple response if template doesn't exist
        return HTMLResponse("""
        <html>
            <head><title>Register - RegIntel</title></head>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px;">
                <h1>📝 Register</h1>
                <p>Registration page is working! Template will be added later.</p>
                <hr>
                <p><strong>Status:</strong> ✅ Authentication system is running</p>
                <p><strong>Next:</strong> Beautiful registration form will be added</p>
                <a href="/" style="color: #007bff;">← Back to Main App</a>
            </body>
        </html>
        """)

@router.post("/signup")
async def signup(request: UserSignUp, response: Response):
    """User registration endpoint"""
    try:
        supabase = supabase_config.get_client()
        
        # Create user with Supabase using v2 syntax
        auth_response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "full_name": request.full_name
                }
            }
        })
        
        # Check if user was created successfully
        if auth_response.user:
            # Set cookies for automatic authentication
            response.set_cookie(
                key="auth_token",
                value=auth_response.session.access_token,
                httponly=True,
                secure=False,  # Set to False for local development
                samesite="lax",  # More permissive for local testing
                max_age=3600,
                path="/"
            )
            
            response.set_cookie(
                key="refresh_token",
                value=auth_response.session.refresh_token,
                httponly=True,
                secure=False,  # Set to False for local development
                samesite="lax",  # More permissive for local testing
                max_age=7*24*3600,  # 7 days
                path="/auth/refresh"
            )
            
            return {
                "message": "Registration successful",
                "user": {
                    "id": auth_response.user.id,
                    "email": auth_response.user.email,
                    "full_name": request.full_name
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Registration failed")
            
    except Exception as e:
        # Log the actual error for debugging
        print(f"Registration error details: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Registration error: {str(e)}")

@router.post("/signin")
async def signin(request: UserSignIn, response: Response):
    """User sign in endpoint"""
    try:
        supabase = supabase_config.get_client()
        
        # Sign in with Supabase
        auth_response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        if auth_response.user and auth_response.session:
            # Set cookies for automatic authentication
            response.set_cookie(
                key="auth_token",
                value=auth_response.session.access_token,
                httponly=True,
                secure=False,  # Set to False for local development
                samesite="lax",  # More permissive for local testing
                max_age=3600,
                path="/"
            )
            
            response.set_cookie(
                key="refresh_token",
                value=auth_response.session.refresh_token,
                httponly=True,
                secure=False,  # Set to False for local development
                samesite="lax",  # More permissive for local testing
                max_age=7*24*3600,  # 7 days
                path="/auth/refresh"
            )
            
            return {
                "message": "Sign in successful",
                "user": {
                    "id": auth_response.user.id,
                    "email": auth_response.user.email,
                    "full_name": auth_response.user.user_metadata.get("full_name", "Unknown")
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Sign in failed")
            
    except Exception as e:
        # Log the actual error for debugging
        print(f"Sign in error details: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Sign in error: {str(e)}")

@router.post("/signout")
async def signout(response: Response):
    """User sign out endpoint"""
    try:
        # Clear authentication cookies
        response.delete_cookie("auth_token", path="/")
        response.delete_cookie("refresh_token", path="/auth/refresh")
        
        return {"message": "Sign out successful"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sign out error: {str(e)}")

@router.post("/refresh")
async def refresh_token(request: Request, response: Response):
    """Refresh access token using refresh token"""
    try:
        refresh_token = request.cookies.get("refresh_token")
        if not refresh_token:
            raise HTTPException(status_code=401, detail="No refresh token")
        
        supabase = supabase_config.get_client()
        auth_response = supabase.auth.refresh_session(refresh_token)
        
        if auth_response.session:
            # Update access token cookie
            response.set_cookie(
                key="auth_token",
                value=auth_response.session.access_token,
                httponly=True,
                secure=True,
                samesite="strict",
                max_age=3600,
                path="/"
            )
            
            return {"message": "Token refreshed successfully"}
        else:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
            
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token refresh error: {str(e)}")

@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: UserProfile = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@router.put("/profile", response_model=UserProfile)
async def update_profile(
    updates: UserUpdate, 
    current_user: UserProfile = Depends(get_current_user)
):
    """Update user profile"""
    try:
        supabase = supabase_config.get_client()
        
        # Update user metadata
        update_data = {}
        if updates.full_name:
            update_data["full_name"] = updates.full_name
        if updates.email:
            update_data["email"] = updates.email
        
        if update_data:
            # Update user metadata in Supabase
            user_response = supabase.auth.update_user({
                "data": update_data
            })
            
            if user_response.user:
                # Return updated profile
                return UserProfile(
                    id=user_response.user.id,
                    email=user_response.user.email,
                    full_name=user_response.user.user_metadata.get("full_name", "Unknown"),
                    created_at=user_response.user.created_at,
                    last_sign_in=user_response.user.last_sign_in_at
                )
        
        return current_user
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profile update error: {str(e)}")

@router.post("/password-reset")
async def request_password_reset(request: PasswordReset):
    """Request password reset email"""
    try:
        supabase = supabase_config.get_client()
        
        # Send password reset email
        supabase.auth.reset_password_email(request.email)
        
        return {"message": "Password reset email sent"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Password reset error: {str(e)}")

@router.post("/password-update")
async def update_password(
    request: PasswordUpdate,
    current_user: UserProfile = Depends(get_current_user)
):
    """Update user password"""
    try:
        supabase = supabase_config.get_client()
        
        # Update password
        user_response = supabase.auth.update_user({
            "password": request.new_password
        })
        
        if user_response.user:
            return {"message": "Password updated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Password update failed")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Password update error: {str(e)}")

@router.get("/me")
async def get_current_user_info(current_user: UserProfile = Depends(get_current_user)):
    """Get current user information (for API clients)"""
    return current_user 