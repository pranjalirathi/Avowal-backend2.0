# from sqlalchemy import create_engine, event
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base

# from dotenv import load_dotenv
# import os

# load_dotenv(".env")

# SQLALCHEMY_DATABASE_URL = os.getenv(key="DATABASE_URI")
# # SQLALCHEMY_DATABASE_URL = "sqlite:///./site.db"
# print(SQLALCHEMY_DATABASE_URL)
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base()

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from models import *
from config import DATABASE_URL
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={"server_settings": {"jit": "off"}, 'statement_cache_size': 0, 'prepared_statement_name_func': lambda: str(uuid.uuid4())}
)
# Async session maker
async_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False,
)

async def init_db():
    async with engine.begin() as conn:
        logger.info("Creating database tables...")
        # Will create all tables, if not present.
        await conn.run_sync(SQLModel.metadata.create_all)


# await conn.run_sync(SQLModel.metadata.drop_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for request-scoped sessions"""
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Production-ready context manager for database sessions.
    Use this for all business logic operations.
    """
    session = None
    try:
        session = async_session()
        yield session
        await session.commit()
    except Exception as e:
        if session:
            await session.rollback()
        logger.error(f"Database session error: {e}")
        raise e
    finally:
        if session:
            await session.close()