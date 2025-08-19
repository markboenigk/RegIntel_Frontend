"""
Authentication routes for RegIntel Frontend
"""

from fastapi import APIRouter, HTTPException, Response, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from .models import UserSignUp, UserSignIn, UserProfile, AuthResponse, PasswordReset, PasswordUpdate, UserUpdate, UserQuery, UserQueryCreate, UserQueryResponse
from .config import get_supabase_config
from .middleware import get_current_user, get_optional_user
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional

# Initialize router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Templates for HTML responses
templates = Jinja2Templates(directory="templates")

@router.post("/signup")
async def signup(request: UserSignUp, response: Response):
    """User registration endpoint"""
    try:
        supabase = get_supabase_config().get_client()
        
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
        supabase = get_supabase_config().get_client()
        
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
                samesite="lax",  # Use lax for local development
                max_age=3600,
                path="/",
                domain=None  # Explicitly set to None for localhost
            )
            
            response.set_cookie(
                key="refresh_token",
                value=auth_response.session.refresh_token,
                httponly=True,
                secure=False,  # Set to False for local development
                samesite="lax",  # Use lax for local development
                max_age=7*24*3600,  # 7 days
                path="/",  # Set to root path so middleware can find it
                domain=None  # Explicitly set to None for localhost
            )
            
            # Return user profile
            user_profile = UserProfile(
                id=auth_response.user.id,
                email=auth_response.user.email,
                full_name=auth_response.user.user_metadata.get("full_name", "Unknown"),
                created_at=auth_response.user.created_at,
                last_sign_in=auth_response.user.last_sign_in_at
            )
            
            return AuthResponse(
                message="Sign in successful",
                user=user_profile,
                access_token=auth_response.session.access_token,
                refresh_token=auth_response.session.refresh_token
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid credentials")
            
    except Exception as e:
        print(f"Signin error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Signin failed: {str(e)}")

@router.post("/signout")
async def signout(response: Response):
    """User sign out endpoint"""
    try:
        # Clear authentication cookies
        response.delete_cookie(key="auth_token", path="/")
        response.delete_cookie(key="refresh_token", path="/")
        
        return {"message": "Sign out successful"}
        
    except Exception as e:
        print(f"Signout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Sign out failed")

@router.post("/refresh")
async def refresh_token(request: Request, response: Response):
    """Refresh access token using refresh token"""
    try:
        refresh_token = request.cookies.get("refresh_token")
        if not refresh_token:
            raise HTTPException(status_code=401, detail="No refresh token")
        
        supabase = get_supabase_config().get_client()
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
        
        supabase = get_supabase_config().get_client()
        
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
        supabase = get_supabase_config().get_client()
        
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
        supabase = get_supabase_config().get_client()
        
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
        supabase = get_supabase_config().get_client()
        
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
    current_user: UserProfile = Depends(get_current_user),
    request: Request = None
):
    """Save a user's search query"""
    try:
        print(f"🔍 DEBUG: Starting to save query for user {current_user.id}")
        print(f"🔍 DEBUG: Query data: {query_data}")
        
        # Get the authenticated Supabase client with user context
        supabase = get_supabase_config().get_client()
        
        # Get the auth token from the request
        auth_token = request.cookies.get("auth_token") if request else None
        if not auth_token:
            print(f"❌ DEBUG: No auth token found in cookies")
            raise HTTPException(status_code=401, detail="No authentication token found")
        
        print(f"🔍 DEBUG: Auth token found, setting session...")
        
        # Set the user's session in the Supabase client
        try:
            supabase.auth.set_session(auth_token, None)
            print(f"✅ DEBUG: Supabase session set successfully")
        except Exception as session_error:
            print(f"❌ DEBUG: Failed to set Supabase session: {str(session_error)}")
            raise HTTPException(status_code=401, detail="Failed to authenticate with database")
        
        print(f"🔍 DEBUG: Supabase client created successfully")
        
        # Create query record
        query_record = {
            "user_id": current_user.id,
            "query_text": query_data.query_text,
            "collection_name": query_data.collection_name,
            "timestamp": datetime.utcnow().isoformat(),
            "response_length": query_data.response_length,
            "sources_count": query_data.sources_count
        }
        
        print(f"🔍 DEBUG: Query record prepared: {query_record}")
        
        # Insert into user_queries table with proper RLS handling
        try:
            print(f"🔍 DEBUG: Attempting to insert into user_queries table...")
            result = supabase.table("user_queries").insert(query_record).execute()
            print(f"🔍 DEBUG: Insert result: {result}")
            
            if result.data:
                print(f"✅ DEBUG: Query saved successfully with ID: {result.data[0].get('id')}")
            else:
                print(f"⚠️ DEBUG: Insert succeeded but no data returned")
                
        except Exception as insert_error:
            print(f"❌ DEBUG: Insert failed with error: {str(insert_error)}")
            print(f"❌ DEBUG: Error type: {type(insert_error).__name__}")
            print(f"⚠️ Warning: Could not save user query due to RLS policy: {str(insert_error)}")
            # Return a mock query object instead of failing
            return UserQuery(
                id="temp_id",
                user_id=current_user.id,
                query_text=query_data.query_text,
                collection_name=query_data.collection_name,
                timestamp=datetime.now(),
                response_length=query_data.response_length,
                sources_count=query_data.sources_count
            )
        
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
            print(f"❌ DEBUG: No data in result, raising error")
            raise HTTPException(status_code=500, detail="Failed to save query")
            
    except Exception as e:
        print(f"❌ DEBUG: Outer exception in save_user_query: {str(e)}")
        print(f"❌ DEBUG: Exception type: {type(e).__name__}")
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
        supabase = get_supabase_config().get_client()
        
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
        supabase = get_supabase_config().get_client()
        
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
        supabase = get_supabase_config().get_client()
        
        # Delete all queries for the user
        result = supabase.table("user_queries")\
            .delete()\
            .eq("user_id", current_user.id)\
            .execute()
        
        return {"message": f"Cleared {len(result.data) if result.data else 0} queries"}
        
    except Exception as e:
        print(f"Error clearing user queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear queries: {str(e)}") 

@router.get("/debug/table-check")
async def check_user_queries_table(
    current_user: UserProfile = Depends(get_current_user),
    request: Request = None
):
    """Debug endpoint to check if user_queries table exists and is accessible"""
    try:
        supabase = get_supabase_config().get_client()
        
        # Get the auth token from the request
        auth_token = request.cookies.get("auth_token") if request else None
        if not auth_token:
            return {
                "table_exists": False,
                "accessible": False,
                "error": "No authentication token found",
                "error_type": "AuthError",
                "user_id": str(current_user.id)
            }
        
        # Set the user's session in the Supabase client
        try:
            supabase.auth.set_session(auth_token, None)
            print(f"✅ DEBUG: Supabase session set successfully for table check")
        except Exception as session_error:
            return {
                "table_exists": False,
                "accessible": False,
                "error": f"Failed to set Supabase session: {str(session_error)}",
                "error_type": "SessionError",
                "user_id": str(current_user.id)
            }
        
        # Try to describe the table
        result = supabase.table("user_queries").select("id").limit(1).execute()
        
        return {
            "table_exists": True,
            "accessible": True,
            "user_id": str(current_user.id),
            "test_result": result
        }
    except Exception as e:
        return {
            "table_exists": False,
            "accessible": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "user_id": str(current_user.id)
        } 