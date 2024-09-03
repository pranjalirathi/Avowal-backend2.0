from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table, func, Text, Boolean, ARRAY, JSON
from sqlalchemy.orm import relationship
from database import Base
from sqlalchemy.sql import func

# Association table for the many-to-many relationship between Confession and User
confession_mentions = Table(
    'confession_mentions',
    Base.metadata,
    Column('confession_id', Integer, ForeignKey('confessions.id'), primary_key=True),
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True)
)

    
class Confession(Base):
    __tablename__ = "confessions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    mentions = relationship("User", secondary=confession_mentions, back_populates="mentioned_in")
    comments=relationship("Comment", back_populates="confessions")
    

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), unique=False, nullable=False)
    hashedpassword = Column(String(255), nullable=False)
    profile_pic = Column(String(255), nullable=False, default="images/profile/def.jpg")
    unread_confessions = Column(ARRAY(Integer), default=[], nullable=False)
    relationship_status = Column(String(10), default="Single", nullable=False)
    mentioned_in = relationship("Confession", secondary=confession_mentions, back_populates="mentions")
    comments = relationship("Comment", back_populates="user")
   

class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey('users.id'))
    confession_id = Column(Integer, ForeignKey('confessions.id'))

    user = relationship("User", back_populates="comments")
    confessions = relationship("Confession", back_populates="comments")