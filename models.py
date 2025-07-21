from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table, func, Text, Boolean, ARRAY, JSON
from sqlalchemy.orm import relationship
from database import Base
from sqlalchemy.sql import func


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), unique=False, nullable=False)
    hashedpassword = Column(String(255), nullable=False)
    profile_pic = Column(String(255), nullable=False, default="images/profile/def.jpg")
    relationship_status = Column(String(10), default="Single", nullable=False)
    