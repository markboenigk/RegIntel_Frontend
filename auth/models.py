"""
Authentication models and schemas
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserSignUp(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")
    full_name: str = Field(..., description="User's full name")
    
class UserSignIn(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    
class UserProfile(BaseModel):
    id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email")
    full_name: str = Field(..., description="User's full name")
    created_at: str = Field(..., description="Account creation date")
    last_sign_in: Optional[str] = Field(None, description="Last sign in date")
    
class AuthResponse(BaseModel):
    user: UserProfile
    access_token: str
    refresh_token: str
    message: str = "Authentication successful"
    
class PasswordReset(BaseModel):
    email: EmailStr = Field(..., description="Email to send reset link to")
    
class PasswordUpdate(BaseModel):
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")
    
class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, description="Updated full name")
    email: Optional[EmailStr] = Field(None, description="Updated email address")

# New models for user queries
class UserQuery(BaseModel):
    id: Optional[str] = Field(None, description="Query ID")
    user_id: str = Field(..., description="User ID")
    query_text: str = Field(..., description="The search query text")
    collection_name: str = Field(..., description="Collection searched (e.g., rss_feeds, fda_warning_letters)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the query was made")
    response_length: Optional[int] = Field(None, description="Length of the AI response")
    sources_count: Optional[int] = Field(None, description="Number of sources found")
    
class UserQueryCreate(BaseModel):
    query_text: str = Field(..., description="The search query text")
    collection_name: str = Field(..., description="Collection searched")
    response_length: Optional[int] = Field(None, description="Length of the AI response")
    sources_count: Optional[int] = Field(None, description="Number of sources found")
    
class UserQueryResponse(BaseModel):
    queries: List[UserQuery]
    total_count: int
    message: str = "Queries retrieved successfully" 