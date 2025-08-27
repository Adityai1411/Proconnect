from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..db import get_db
from .. import models
from ..config import settings
from groq import Groq

router = APIRouter(prefix="/chat", tags=["chat"])

# Request schema
class ChatReq(BaseModel):
    message: str
    resume_id: int | None = None  # Optional resume context

@router.post("/")  # ensure slash is present
def chat(req: ChatReq, db: Session = Depends(get_db)):
    # Ensure GROQ key exists
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing in backend .env")

    # Retrieve resume content if resume_id is provided
    resume_context = ""
    if req.resume_id is not None:
        r = db.query(models.Resume).filter(models.Resume.id == req.resume_id).first()
        if r:
            resume_context = r.content

    # Initialize Groq client
    client = Groq(api_key=settings.GROQ_API_KEY)

    # Prepare messages
    messages = []
    if resume_context:
        messages.append({
            "role": "system",
            "content": f"Here is the user's resume:\n{resume_context}"
        })
    messages.append({"role": "user", "content": req.message})

    # Call Groq API
    try:
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            stream=False
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    # Store chat message (optional, minimal)
    try:
        cm = models.ChatMessage(user_id=None, role="assistant", content=reply)
        db.add(cm)
        db.commit()
    except Exception:
        db.rollback()  # prevent DB crash if table missing

    return {"response": reply}
