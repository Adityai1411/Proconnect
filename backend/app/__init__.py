"""
backend/app package initializer.

Expose common objects (settings, DB engine/session/base) so other modules can import them:
    from app import settings, engine, SessionLocal, Base
Also sets up a package logger (NullHandler by default to avoid double-logs).
"""
from .config import settings
from .db import engine, SessionLocal, Base

import logging

logger = logging.getLogger("resume_analyzer")
# Avoid configuring handlers here to prevent duplicate handlers when the package is imported multiple times.
logger.addHandler(logging.NullHandler())

__all__ = ["settings", "engine", "SessionLocal", "Base", "logger"]
