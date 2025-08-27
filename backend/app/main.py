from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .db import Base, engine
from .routers import resume, chat

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Resume Analyzer API", version="1.0")

# Allow frontend origin
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(resume.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"status": "ok", "service": "resume-analyzer-api"}
