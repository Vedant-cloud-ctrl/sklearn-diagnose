"""
Server module for the sklearn-diagnose chatbot.

This module provides a FastAPI-based server that enables interactive
conversations about diagnosis reports through a web interface.
"""

from skdiagnose.server.app import app
from skdiagnose.server.chat_agent import ChatAgent

__all__ = ["app", "ChatAgent"]
