from typing import List, Optional
from sqlmodel import Field, Relationship, SQLModel, ARRAY, Column, Integer, String
from datetime import datetime

class ConfessionMentionLink(SQLModel, table=True):
    confession_id: Optional[int] = Field(
        default=None, foreign_key="confession.id", primary_key=True
    )
    user_id: Optional[int] = Field(
        default=None, foreign_key="user.id", primary_key=True
    )


class Comment(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, index=True)
    content: str = Field(sa_column=Column(String))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    confession_id: Optional[int] = Field(default=None, foreign_key="confession.id")

    user: Optional["User"] = Relationship(back_populates="comments")
    confession: Optional["Confession"] = Relationship(back_populates="comments")


class Confession(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, index=True)
    content: str = Field(nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    mentions: List["User"] = Relationship(
        back_populates="mentioned_in", link_model=ConfessionMentionLink
    )
    comments: List[Comment] = Relationship(back_populates="confession")


class User(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True, index=True)
    username: str = Field(max_length=50, unique=True, nullable=False, index=True)
    email: str = Field(max_length=255, unique=True, nullable=False)
    name: str = Field(max_length=255, nullable=False)
    hashedpassword: str = Field(max_length=255, nullable=False)
    profile_pic: str = Field(
        max_length=255, nullable=False, default="images/profile/def.jpg"
    )
    unread_confessions: List[int] = Field(
        default=[], sa_column=Column(ARRAY(Integer))
    )
    relationship_status: str = Field(max_length=10, default="Single", nullable=False)

    mentioned_in: List[Confession] = Relationship(
        back_populates="mentions", link_model=ConfessionMentionLink
    )
    comments: List[Comment] = Relationship(back_populates="user")