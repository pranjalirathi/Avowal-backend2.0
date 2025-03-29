from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from dotenv import load_dotenv
import os


# SQLALCHEMY_DATABASE_URL = os.getenv(key="DATABASE_URI")
SQLALCHEMY_DATABASE_URL = "sqlite:///site.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
