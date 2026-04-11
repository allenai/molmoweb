"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field
import uuid


# ============================================================================
# Session Models
# ============================================================================

class SessionCreate(BaseModel):
    """Request to create a new session."""
    goal: str = Field(..., description="The task/goal for the agent")
    start_url: str = Field(default="https://www.google.com", description="Starting URL")


class SessionResponse(BaseModel):
    """Response containing session details."""
    id: str
    goal: str
    start_url: str
    status: Literal["running", "paused", "completed", "stopped"]
    step_count: int
    created_at: datetime
    updated_at: datetime


class SessionList(BaseModel):
    """List of sessions."""
    sessions: list[SessionResponse]


# ============================================================================
# Step Models
# ============================================================================

class ActionData(BaseModel):
    """Action data from the model."""
    name: str
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    url: Optional[str] = None
    key: Optional[str] = None
    delta_x: Optional[float] = None
    delta_y: Optional[float] = None
    button: Optional[str] = None
    msg: Optional[str] = None


class StepSummary(BaseModel):
    """Summary of a step (without full screenshot)."""
    id: int
    status: Literal["running", "completed", "failed"]
    thought: Optional[str] = None
    action_preview: Optional[str] = None  # e.g., "click(x=450, y=320)"
    action_description: Optional[str] = None
    url: Optional[str] = None
    created_at: datetime


class StepDetail(BaseModel):
    """Full step details including screenshot."""
    id: int
    status: Literal["running", "completed", "failed"]
    thought: Optional[str] = None
    action: Optional[ActionData] = None
    action_str: Optional[str] = None
    action_description: Optional[str] = None
    url: Optional[str] = None
    screenshot_base64: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime


# ============================================================================
# Chat/Message Models
# ============================================================================

class ChatMessage(BaseModel):
    """A message in the chat (user, agent, or system)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["user", "agent", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # For agent messages, may include associated steps
    step_ids: Optional[list[int]] = None
