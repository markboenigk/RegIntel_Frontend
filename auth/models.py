"""
Authentication models and schemas
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

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