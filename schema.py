from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from datetime import datetime
from models import User
class UserCreate(BaseModel):
    username : str = Field(min_length=3, max_length=50)  
    email : EmailStr
    password: str = Field(min_length=6, max_length=255)

class UserResponse(BaseModel):
    id: int
    username: str

    class Config:
        from_attributes = True
        
class ConfessionCreate(BaseModel):
    content: str

class ConfessionResponse(BaseModel):
    id: int
    content: str
    created_at: datetime
    mentions: List

    class Config:
        orm_mode = True
        from_attributes = True

class CommentResponse(BaseModel):
    id: int
    content: str
    created_at: datetime
    user: UserResponse

    class Config:
        from_attributes = True

class CommentCreate(BaseModel):
    content: str