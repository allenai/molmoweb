"""SQLAlchemy models for session persistence."""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Boolean, String, Text, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column

from demo.database import Base


class SessionModel(Base):
    """Database model for user sessions."""
    
    __tablename__ = "sessions"
    
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=True)  # X-Auth-Request-Email or fallback; nullable for migration
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    start_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running, paused, completed, stopped
    share_token: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    steps: Mapped[List["StepModel"]] = relationship(
        "StepModel", back_populates="session", cascade="all, delete-orphan"
    )
    messages: Mapped[List["ChatMessageModel"]] = relationship(
        "ChatMessageModel", back_populates="session", cascade="all, delete-orphan"
    )


class StepModel(Base):
    """Database model for steps within a session."""
    
    __tablename__ = "steps"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    session_id: Mapped[str] = mapped_column(String(16), ForeignKey("sessions.id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    thought: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    action: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    action_str: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    action_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    screenshot_base64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_base64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="steps")


class ChatMessageModel(Base):
    """Database model for chat messages."""
    
    __tablename__ = "chat_messages"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(16), ForeignKey("sessions.id"))
    type: Mapped[str] = mapped_column(String(20), nullable=False)  # user, agent, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    step_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    is_answer: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    is_final: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    is_safe: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session: Mapped["SessionModel"] = relationship("SessionModel", back_populates="messages")


class SessionEventModel(Base):
    """Append-only log of session events (take_control, resume, etc.)."""

    __tablename__ = "session_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(16), ForeignKey("sessions.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserConsentModel(Base):
    """Tracks per-user terms acceptance and open-science consent."""

    __tablename__ = "user_consent"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    terms_accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    open_science_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
