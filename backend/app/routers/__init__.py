"""
routers package initializer.

This imports the router modules so that:
    from .routers import resume, chat
works and resume.router / chat.router are accessible.
"""
from . import resume, chat

__all__ = ["resume", "chat"]
