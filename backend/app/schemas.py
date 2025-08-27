from pydantic import BaseModel
from typing import List, Dict, Optional

class UserCreate(BaseModel):
    email: str
    name: Optional[str] = None

class UserOut(BaseModel):
    id: int
    email: str
    name: Optional[str]
    class Config:
        from_attributes = True

class ResumeOut(BaseModel):
    id: int
    filename: str
    content: str
    class Config:
        from_attributes = True

class AnalysisOut(BaseModel):
    resume_id: int
    keywords: List[str]
    entities: List[tuple]
    nouns: List[str]
    sentence_to_platform: Dict[str, str]
    score: int

class ChatIn(BaseModel):
    message: str
    resume_context: Optional[str] = None

class ChatOut(BaseModel):
    response: str
