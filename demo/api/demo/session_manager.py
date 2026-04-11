"""Session manager for handling agent sessions with PostgreSQL persistence."""

import uuid
from datetime import datetime
from typing import Any, Optional, Literal
from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import joinedload

from demo.config import settings
from demo.database import get_db
from demo.models import SessionModel, StepModel, ChatMessageModel, SessionEventModel


# =============================================================================
# Dataclasses (kept for API compatibility)
# =============================================================================

@dataclass
class Step:
    """A single step in a session."""
    id: int
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    thought: Optional[str] = None
    action: Optional[dict[str, Any]] = None
    action_str: Optional[str] = None
    action_description: Optional[str] = None
    url: Optional[str] = None
    screenshot_base64: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChatMessage:
    """A message in the chat history."""
    id: str
    type: Literal["user", "agent", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step_ids: list[int] = field(default_factory=list)
    is_answer: bool = False
    is_final: bool = False


@dataclass
class Session:
    """A user session with the agent."""
    id: str
    goal: str
    start_url: str
    status: Literal["running", "paused", "completed", "stopped"] = "running"
    steps: list[Step] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""
    
    _step_count: Optional[int] = None

    @property
    def step_count(self) -> int:
        if self._step_count is not None:
            return self._step_count
        return len(self.steps)
    
    def get_step(self, step_id: int) -> Optional[Step]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def add_message(self, msg_type: Literal["user", "agent", "system"], content: str, step_ids: list[int] = None, is_answer: bool = False, is_final: bool = False) -> ChatMessage:
        """Add a message to the chat history."""
        msg = ChatMessage(
            id=str(uuid.uuid4()),
            type=msg_type,
            content=content,
            step_ids=step_ids or [],
            is_answer=is_answer,
            is_final=is_final,
        )
        self.messages.append(msg)
        self.updated_at = datetime.utcnow()
        return msg
    
    def get_conversation_history(self) -> list[dict[str, Any]]:
        """
        Get conversation history for memory context.
        Returns a list of previous tasks with their key outcomes.
        Uses step_ids on agent messages to properly group steps by query.
        """
        history = []
        
        # Build step lookup by id
        step_lookup = {step.id: step for step in self.steps}
        
        current_query = None
        current_result = None
        current_step_ids = []
        
        for msg in self.messages:
            if msg.type == "user":
                # Save previous query if exists
                if current_query:
                    steps = [step_lookup[sid] for sid in current_step_ids if sid in step_lookup]
                    actions = [s.action_description for s in steps if s.action_description]
                    thoughts = [s.thought for s in steps if s.thought]
                    history.append({
                        "query": current_query,
                        "key_actions": actions[-3:],  # Last 3 actions
                        "last_thought": thoughts[-1] if thoughts else None,
                        "result": current_result,
                    })
                # Start new query
                current_query = msg.content
                current_result = None
                current_step_ids = []
            elif msg.type == "agent":
                current_result = msg.content
                current_step_ids = msg.step_ids or []
        
        # Add final query if exists
        if current_query:
            steps = [step_lookup[sid] for sid in current_step_ids if sid in step_lookup]
            actions = [s.action_description for s in steps if s.action_description]
            thoughts = [s.thought for s in steps if s.thought]
            if actions or current_result:
                history.append({
                    "query": current_query,
                    "key_actions": actions[-3:],
                    "last_thought": thoughts[-1] if thoughts else None,
                    "result": current_result,
                })
        
        return history
    
    def get_followup_context(self, new_query: str) -> dict[str, Any]:
        """
        Get context for follow-up queries including last screenshot.
        Conversation history is passed separately for the template to render.
        """
        last_step = self.steps[-1] if self.steps else None
        
        return {
            "goal": new_query,
            "last_screenshot_base64": last_step.screenshot_base64 if last_step else None,
            "last_url": last_step.url if last_step else None,
            "last_thought": last_step.thought if last_step else None,
            "conversation_history": self.get_conversation_history(),
        }


# =============================================================================
# Conversion helpers (DB model <-> Dataclass)
# =============================================================================

def _step_from_model(model: StepModel) -> Step:
    """Convert StepModel to Step dataclass."""
    return Step(
        id=model.id,
        status=model.status,
        thought=model.thought,
        action=model.action,
        action_str=model.action_str,
        action_description=model.action_description,
        url=model.url,
        screenshot_base64=model.screenshot_base64,
        thumbnail_base64=model.thumbnail_base64,
        error=model.error,
        created_at=model.created_at,
    )


def _message_from_model(model: ChatMessageModel) -> ChatMessage:
    """Convert ChatMessageModel to ChatMessage dataclass."""
    return ChatMessage(
        id=model.id,
        type=model.type,
        content=model.content,
        timestamp=model.timestamp,
        step_ids=model.step_ids or [],
        is_answer=getattr(model, "is_answer", False) or False,
        is_final=getattr(model, "is_final", False) or False,
    )


def _session_from_model(model: SessionModel) -> Session:
    """Convert SessionModel to Session dataclass."""
    steps = [_step_from_model(s) for s in sorted(model.steps, key=lambda x: x.id)]
    messages = [_message_from_model(m) for m in sorted(model.messages, key=lambda x: x.timestamp)]
    
    return Session(
        id=model.id,
        goal=model.goal,
        start_url=model.start_url,
        status=model.status,
        steps=steps,
        messages=messages,
        created_at=model.created_at,
        updated_at=model.updated_at,
        user_id=getattr(model, "user_id", "") or "",
    )


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """
    Manages all active sessions with PostgreSQL persistence.
    """
    
    def __init__(self):
        pass  # No longer storing in-memory state
    
    def create_session(
        self, goal: str, start_url: str = "https://www.google.com", user_id: str = ""
    ) -> Session:
        """Create a new session for the given user."""
        session_id = str(uuid.uuid4())[:8]  # Short ID for readability
        
        with get_db() as db:
            # Create session model
            session_model = SessionModel(
                id=session_id,
                user_id=user_id,
                goal=goal,
                start_url=start_url,
                status="running",
            )
            db.add(session_model)
            
            # Add initial user message
            msg = ChatMessageModel(
                id=str(uuid.uuid4()),
                session_id=session_id,
                type="user",
                content=goal,
            )
            db.add(msg)
            db.flush()
            
            # Return as dataclass
            session = Session(
                id=session_id,
                goal=goal,
                start_url=start_url,
                user_id=user_id,
            )
            session.add_message("user", goal)
            return session
    
    def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        """Get a session by ID. If user_id is provided, return only if session belongs to that user."""
        with get_db() as db:
            q = db.query(SessionModel).options(
                joinedload(SessionModel.steps),
                joinedload(SessionModel.messages),
            ).filter(SessionModel.id == session_id)
            if user_id is not None:
                q = q.filter(SessionModel.user_id == user_id)
            model = q.first()
            
            if not model:
                return None
            
            return _session_from_model(model)
    
    def get_session_light(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        """Lightweight session fetch: sessions table only, no steps or messages loaded."""
        with get_db() as db:
            q = db.query(SessionModel).filter(SessionModel.id == session_id)
            if user_id is not None:
                q = q.filter(SessionModel.user_id == user_id)
            model = q.first()
            if not model:
                return None
            return Session(
                id=model.id,
                goal=model.goal,
                start_url=model.start_url,
                status=model.status,
                created_at=model.created_at,
                updated_at=model.updated_at,
                user_id=getattr(model, "user_id", "") or "",
            )

    def get_session_lock(self, session_id: str):
        """
        Get a lock for a session.
        With database persistence, we rely on database transactions for consistency.
        Returns a dummy context manager for API compatibility.
        """
        from contextlib import nullcontext
        return nullcontext()
    
    def list_sessions(self, user_id: str) -> list[Session]:
        """List sessions for the given user, newest first. No steps/messages loaded."""
        with get_db() as db:
            step_count_sub = (
                db.query(func.count(StepModel.id))
                .filter(StepModel.session_id == SessionModel.id)
                .correlate(SessionModel)
                .scalar_subquery()
            )
            rows = (
                db.query(SessionModel, step_count_sub.label("step_count"))
                .filter(SessionModel.user_id == user_id)
                .order_by(SessionModel.updated_at.desc())
                .all()
            )
            sessions = []
            for model, count in rows:
                s = Session(
                    id=model.id,
                    goal=model.goal,
                    start_url=model.start_url,
                    status=model.status,
                    created_at=model.created_at,
                    updated_at=model.updated_at,
                    user_id=getattr(model, "user_id", "") or "",
                    _step_count=count or 0,
                )
                sessions.append(s)
            return sessions
    
    def delete_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a session. If user_id is provided, delete only if session belongs to that user."""
        with get_db() as db:
            q = db.query(SessionModel).filter(SessionModel.id == session_id)
            if user_id is not None:
                q = q.filter(SessionModel.user_id == user_id)
            model = q.first()
            if model:
                db.delete(model)
                return True
            return False
    
    def delete_all_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user. Returns count deleted."""
        with get_db() as db:
            models = db.query(SessionModel).filter(
                SessionModel.user_id == user_id
            ).all()
            count = len(models)
            for model in models:
                db.delete(model)
            return count
    
    def add_step(self, session_id: str) -> Optional[Step]:
        """Add a new step to a session."""
        with get_db() as db:
            session_model = db.query(SessionModel).filter(
                SessionModel.id == session_id
            ).first()
            
            if not session_model:
                return None
            
            # Get current step count
            step_count = db.query(StepModel).filter(
                StepModel.session_id == session_id,
            ).count()
            
            step_id = step_count + 1
            
            step_model = StepModel(
                id=step_id,
                session_id=session_id,
                status="running",
            )
            db.add(step_model)
            
            session_model.updated_at = datetime.utcnow()
            db.flush()
            
            return Step(id=step_id, status="running")
    
    def update_step(
        self,
        session_id: str,
        step_id: int,
        **kwargs,
    ) -> Optional[Step]:
        """Update a step with new data."""
        with get_db() as db:
            session_model = db.query(SessionModel).filter(
                SessionModel.id == session_id
            ).first()
            
            if not session_model:
                return None
            
            step_model = db.query(StepModel).filter(
                StepModel.id == step_id,
                StepModel.session_id == session_id,
            ).first()
            
            if not step_model:
                return None
            
            for key, value in kwargs.items():
                if hasattr(step_model, key):
                    setattr(step_model, key, value)
            
            session_model.updated_at = datetime.utcnow()
            db.flush()
            
            return _step_from_model(step_model)
    
    def update_session_status(
        self,
        session_id: str,
        status: Literal["running", "paused", "completed", "stopped"],
    ) -> bool:
        """Update session status. Returns True if session was found and updated."""
        with get_db() as db:
            session_model = db.query(SessionModel).filter(
                SessionModel.id == session_id
            ).first()

            if not session_model:
                return False

            session_model.status = status
            session_model.updated_at = datetime.utcnow()
            db.flush()

            return True
    
    def share_session(self, session_id: str, user_id: str) -> Optional[str]:
        """Generate a share token for a session. Returns the token, or None if session not found."""
        with get_db() as db:
            model = db.query(SessionModel).filter(
                SessionModel.id == session_id,
                SessionModel.user_id == user_id,
            ).first()
            if not model:
                return None
            if model.share_token:
                return model.share_token
            token = str(uuid.uuid4())
            model.share_token = token
            db.flush()
            return token

    def unshare_session(self, session_id: str, user_id: str) -> bool:
        """Remove the share token from a session. Returns True if session was found."""
        with get_db() as db:
            model = db.query(SessionModel).filter(
                SessionModel.id == session_id,
                SessionModel.user_id == user_id,
            ).first()
            if not model:
                return False
            model.share_token = None
            db.flush()
            return True

    def get_shared_session(self, share_token: str) -> Optional[Session]:
        """Fetch a full session by share token (no user_id check)."""
        with get_db() as db:
            model = (
                db.query(SessionModel)
                .options(
                    joinedload(SessionModel.steps),
                    joinedload(SessionModel.messages),
                )
                .filter(SessionModel.share_token == share_token)
                .first()
            )
            if not model:
                return None
            return _session_from_model(model)

    def get_share_token(self, session_id: str, user_id: str) -> Optional[str]:
        """Get the existing share token for a session, or None."""
        with get_db() as db:
            model = db.query(SessionModel).filter(
                SessionModel.id == session_id,
                SessionModel.user_id == user_id,
            ).first()
            if not model:
                return None
            return model.share_token

    def add_message(
        self,
        session_id: str,
        msg_type: Literal["user", "agent", "system"],
        content: str,
        step_ids: list[int] = None,
        is_answer: bool = False,
        is_final: bool = False,
    ) -> Optional[ChatMessage]:
        """Add a message to the session."""
        with get_db() as db:
            session_model = db.query(SessionModel).filter(
                SessionModel.id == session_id
            ).first()
            
            if not session_model:
                return None
            
            msg_model = ChatMessageModel(
                id=str(uuid.uuid4()),
                session_id=session_id,
                type=msg_type,
                content=content,
                step_ids=step_ids or [],
                is_answer=is_answer,
                is_final=is_final,
            )
            db.add(msg_model)
            
            session_model.updated_at = datetime.utcnow()
            db.flush()
            
            return _message_from_model(msg_model)

    def log_session_event(self, session_id: str, event_type: str) -> None:
        """Append an event to the session_events log."""
        with get_db() as db:
            db.add(SessionEventModel(session_id=session_id, event_type=event_type))

# Global session manager instance
session_manager = SessionManager()
