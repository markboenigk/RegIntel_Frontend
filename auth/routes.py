"""
Authentication routes for user management
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from .models import UserSignUp, UserSignIn, UserProfile, AuthResponse, PasswordReset, PasswordUpdate, UserUpdate, UserQuery, UserQueryCreate, UserQueryResponse
from .config import supabase_config
from .middleware import get_current_user, get_optional_user
import json
from datetime import datetime
from typing import List

router = APIRouter(prefix="/auth", tags=["authentication"])
templates = Jinja2Templates(directory="templates")

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    return templates.TemplateResponse("auth/login.html", {"request": request})

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Serve the registration page"""
    return templates.TemplateResponse("auth/register.html", {"request": request})

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
                httponly=True,  # Back to secure HttpOnly
                secure=False,  # Set to False for local development
                samesite=None,  # Remove SameSite restriction for localhost
                max_age=3600,
                path="/",
                domain=None  # Explicitly set to None for localhost
            )
            
            response.set_cookie(
                key="refresh_token",
                value=auth_response.session.refresh_token,
                httponly=True,
                secure=False,  # Set to False for local development
                samesite="lax",  # More permissive for local testing
                max_age=7*24*3600,  # 7 days
                path="/",
                domain=None  # Explicitly set to None for localhost
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
        print(f"🔐 Signin attempt for email: {request.email}")
        supabase = supabase_config.get_client()
        
        # Sign in with Supabase
        print(f"🔐 Calling Supabase auth.sign_in_with_password...")
        auth_response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        print(f"🔐 Supabase response received")
        print(f"🔐 User object: {auth_response.user}")
        print(f"🔐 Session object: {auth_response.session}")
        print(f"🔐 Auth response type: {type(auth_response)}")
        
        if auth_response.user and auth_response.session:
            print(f"🔐 Setting cookies for user: {auth_response.user.email}")
            print(f"🔐 Access token length: {len(auth_response.session.access_token)}")
            
            # Set cookies for automatic authentication
            response.set_cookie(
                key="auth_token",
                value=auth_response.session.access_token,
                httponly=True,  # Back to secure HttpOnly
                secure=False,  # Set to False for local development
                samesite=None,  # Remove SameSite restriction for localhost
                max_age=3600,
                path="/",
                domain=None  # Explicitly set to None for localhost
            )
            
            response.set_cookie(
                key="refresh_token",
                value=auth_response.session.refresh_token,
                httponly=True,
                secure=False,  # Set to False for local development
                samesite=None,  # Remove SameSite restriction for localhost
                max_age=7*24*3600,  # 7 days
                path="/",  # Set to root path so middleware can find it
                domain=None  # Explicitly set to None for localhost
            )
            
            print(f"✅ Cookies set successfully")
            print(f"🔍 Response headers: {dict(response.headers)}")
            print(f"🔍 Set-Cookie header: {response.headers.get('set-cookie', 'Not found')}")
            
            # Test: Try to get the cookie back from the response
            test_cookie = response.headers.get('set-cookie')
            if test_cookie:
                print(f"🔍 Cookie parsing test:")
                print(f"   Raw cookie: {test_cookie}")
                print(f"   Contains 'auth_token': {'auth_token' in test_cookie}")
                print(f"   Contains 'HttpOnly': {'HttpOnly' in test_cookie}")
                print(f"   Contains 'Path=/': {'Path=/' in test_cookie}")
            else:
                print(f"❌ No Set-Cookie header found!")
            
            return {
                "message": "Sign in successful",
                "user": {
                    "id": auth_response.user.id,
                    "email": auth_response.user.email,
                    "full_name": auth_response.user.user_metadata.get("full_name", "Unknown")
                }
            }
        else:
            print(f"❌ Sign in failed - user or session missing")
            print(f"❌ User exists: {bool(auth_response.user)}")
            print(f"❌ Session exists: {bool(auth_response.session)}")
            raise HTTPException(status_code=400, detail="Sign in failed")
            
    except Exception as e:
        # Log the actual error for debugging
        print(f"Sign in error details: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Sign in error: {str(e)}")

@router.post("/signout")
async def signout(response: Response):
    """User sign out endpoint"""
    try:
        print("🔐 Signout request received")
        
        # Clear authentication cookies with more comprehensive clearing
        response.delete_cookie("auth_token", path="/", domain=None)
        response.delete_cookie("refresh_token", path="/", domain=None)
        
        # Also try clearing with different path variations
        response.delete_cookie("auth_token", path="/", domain=None)
        response.delete_cookie("refresh_token", path="/", domain=None)
        
        # Clear any other potential auth-related cookies
        response.delete_cookie("supabase-auth-token", path="/", domain=None)
        response.delete_cookie("sb-access-token", path="/", domain=None)
        response.delete_cookie("sb-refresh-token", path="/", domain=None)
        
        print("🔐 Cookies cleared successfully")
        
        # Return a JSON response instead of redirect to avoid cookie issues
        return {"message": "Sign out successful", "redirect": "/"}
        
    except Exception as e:
        print(f"❌ Signout error: {str(e)}")
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
    current_user: UserProfile = Depends(get_current_user),
    request: Request = None
):
    """Update user profile"""
    try:
        print(f"🔐 Profile update attempt for user: {current_user.email}")
        print(f"🔐 Request cookies: {dict(request.cookies) if request else 'No request object'}")
        
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
                
                # Return updated profile
                return UserProfile(
                    id=user_response.user.id,
                    email=user_response.user.email,
                    full_name=user_response.user.user_metadata.get("full_name", "Unknown"),
                    created_at=created_at,
                    last_sign_in=last_sign_in
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

@router.post("/change-password")
async def change_password(
    request: PasswordUpdate,
    current_user: UserProfile = Depends(get_current_user)
):
    """Change user password (alias for password-update)"""
    return await update_password(request, current_user)

@router.post("/delete-account")
async def delete_account(current_user: UserProfile = Depends(get_current_user)):
    """Delete user account"""
    try:
        supabase = supabase_config.get_client()
        
        # Delete user account
        user_response = supabase.auth.admin.delete_user(current_user.id)
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Account deletion error: {str(e)}")

@router.get("/me")
async def get_current_user_info(current_user: UserProfile = Depends(get_current_user)):
    """Get current user information (for API clients)"""
    return current_user

@router.get("/profile-page", response_class=HTMLResponse)
async def profile_page(request: Request, current_user: UserProfile = Depends(get_current_user)):
    """Serve the profile page"""
    return templates.TemplateResponse("auth/profile.html", {"request": request}) 

@router.post("/queries", response_model=UserQuery)
async def save_user_query(
    query_data: UserQueryCreate,
    current_user: UserProfile = Depends(get_current_user)
):
    """Save a user's search query"""
    try:
        supabase = supabase_config.get_client()
        
        # Create query record
        query_record = {
            "user_id": current_user.id,
            "query_text": query_data.query_text,
            "collection_name": query_data.collection_name,
            "timestamp": datetime.utcnow().isoformat(),
            "response_length": query_data.response_length,
            "sources_count": query_data.sources_count
        }
        
        # Insert into user_queries table
        result = supabase.table("user_queries").insert(query_record).execute()
        
        if result.data:
            saved_query = result.data[0]
            return UserQuery(
                id=saved_query.get("id"),
                user_id=saved_query["user_id"],
                query_text=saved_query["query_text"],
                collection_name=saved_query["collection_name"],
                timestamp=datetime.fromisoformat(saved_query["timestamp"]),
                response_length=saved_query.get("response_length"),
                sources_count=saved_query.get("sources_count")
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save query")
            
    except Exception as e:
        print(f"Error saving user query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save query: {str(e)}")

@router.get("/queries", response_model=UserQueryResponse)
async def get_user_queries(
    current_user: UserProfile = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Get user's search query history"""
    try:
        supabase = supabase_config.get_client()
        
        # Query user_queries table
        result = supabase.table("user_queries")\
            .select("*")\
            .eq("user_id", current_user.id)\
            .order("timestamp", desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()
        
        if result.data is not None:
            queries = []
            for query_data in result.data:
                query = UserQuery(
                    id=query_data.get("id"),
                    user_id=query_data["user_id"],
                    query_text=query_data["query_text"],
                    collection_name=query_data["collection_name"],
                    timestamp=datetime.fromisoformat(query_data["timestamp"]),
                    response_length=query_data.get("response_length"),
                    sources_count=query_data.get("sources_count")
                )
                queries.append(query)
            
            # Get total count
            count_result = supabase.table("user_queries")\
                .select("id", count="exact")\
                .eq("user_id", current_user.id)\
                .execute()
            
            total_count = count_result.count if count_result.count is not None else len(queries)
            
            return UserQueryResponse(
                queries=queries,
                total_count=total_count
            )
        else:
            return UserQueryResponse(
                queries=[],
                total_count=0
            )
            
    except Exception as e:
        print(f"Error retrieving user queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve queries: {str(e)}")

@router.delete("/queries/{query_id}")
async def delete_user_query(
    query_id: str,
    current_user: UserProfile = Depends(get_current_user)
):
    """Delete a specific user query"""
    try:
        supabase = supabase_config.get_client()
        
        # Verify ownership and delete
        result = supabase.table("user_queries")\
            .delete()\
            .eq("id", query_id)\
            .eq("user_id", current_user.id)\
            .execute()
        
        if result.data:
            return {"message": "Query deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Query not found or not owned by user")
            
    except Exception as e:
        print(f"Error deleting user query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete query: {str(e)}")

@router.delete("/queries")
async def clear_user_queries(current_user: UserProfile = Depends(get_current_user)):
    """Clear all user queries"""
    try:
        supabase = supabase_config.get_client()
        
        # Delete all queries for the user
        result = supabase.table("user_queries")\
            .delete()\
            .eq("user_id", current_user.id)\
            .execute()
        
        return {"message": f"Cleared {len(result.data) if result.data else 0} queries"}
        
    except Exception as e:
        print(f"Error clearing user queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear queries: {str(e)}") 