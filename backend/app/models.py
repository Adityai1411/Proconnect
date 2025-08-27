from sqlalchemy import Column, Integer, String, ForeignKey, Text, Float, DateTime
from sqlalchemy.orm import relationship
from .db import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable allows uploads without email
    filename = Column(String(255))
    content = Column(Text)

    user = relationship("User")
    analysis = relationship("ResumeAnalysis", uselist=False, back_populates="resume")

class ResumeAnalysis(Base):
    __tablename__ = "resume_analysis"
    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    keywords = Column(Text)
    entities = Column(Text)
    nouns = Column(Text)
    sentence_to_platform = Column(Text)
    score = Column(Float, default=0.0)

    resume = relationship("Resume", back_populates="analysis")
