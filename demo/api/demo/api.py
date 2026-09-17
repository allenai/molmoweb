"""REST API routes for the WebOlmo demo."""

from datetime import datetime

from fastapi import APIRouter, Request, Response
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from demo.session_manager import session_manager
from demo.schemas import SessionCreate
from demo.config import settings
from demo.database import get_db
from demo.models import UserConsentModel

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def get_current_user_id(request: Request) -> str:
    """Current user from OAuth header; fallback for local dev."""
    email = request.headers.get("X-Auth-Request-Email")
    return email or "demo.user@allenai.org"


def session_to_response(session) -> dict:
    """Convert a Session object to response dict."""
    return {
        "id": session.id,
        "goal": session.goal,
        "start_url": session.start_url,
        "status": session.status,
        "step_count": session.step_count,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


# ========================================================================
# Health Check
# ========================================================================

@router.get("/")
def index():
    """Health check endpoint for Kubernetes."""
    return Response(status_code=204)


@router.get("/api/health")
def health():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "modal_endpoint": settings.MODAL_ENDPOINT or None,
    }


@router.get("/api/user")
def get_user(request: Request):
    """Get the authenticated user info from OAuth headers."""
    email = request.headers.get("X-Auth-Request-Email")
    if not email:
        email = "demo.user@allenai.org"
    return {
        "authenticated": True,
        "email": email,
    }


# ========================================================================
# Session Routes
# ========================================================================

@router.post("/api/sessions", status_code=201)
def create_session(body: SessionCreate, request: Request):
    """Create a new session for the current user."""
    if not body.goal or len(body.goal.strip()) == 0:
        raise HTTPException(status_code=400, detail="Please provide a goal")

    user_id = get_current_user_id(request)

    session = session_manager.create_session(
        goal=body.goal,
        start_url=body.start_url,
        user_id=user_id,
    )

    logger.info({
        "message": "Session created",
        "event": "session_created",
        "session_id": session.id,
        "goal": body.goal,
    })

    return session_to_response(session)


@router.get("/api/sessions")
def list_sessions(request: Request):
    """List sessions for the current user."""
    user_id = get_current_user_id(request)
    sessions = session_manager.list_sessions(user_id=user_id)
    return {
        "sessions": [session_to_response(s) for s in sessions]
    }


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    """Get a session by ID (must belong to current user)."""
    user_id = get_current_user_id(request)
    session = session_manager.get_session_light(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_to_response(session)


@router.get("/api/sessions/{session_id}/full")
def get_session_full(session_id: str, request: Request):
    """Get full session with messages and steps for loading a conversation."""
    user_id = get_current_user_id(request)
    session = session_manager.get_session(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Clean up stale "running" steps left by previous crashes.
    # Import here to avoid circular import at module level.
    from demo.socket_events import active_agent_threads
    has_active_agent = (
        session_id in active_agent_threads
        and active_agent_threads[session_id].is_alive()
    )
    if not has_active_agent:
        stale_steps = [s for s in session.steps if s.status == "running"]
        for s in stale_steps:
            session_manager.update_step(
                session_id=session_id, step_id=s.id, status="cancelled"
            )
            s.status = "cancelled"
        if session.status == "running" and stale_steps:
            session_manager.update_session_status(session_id, "stopped")
            session.status = "stopped"

    out = session_to_response(session)
    share_token = session_manager.get_share_token(session_id, user_id)
    out["share_token"] = share_token
    out["messages"] = [
        {
            "id": m.id,
            "type": m.type,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
            "step_ids": getattr(m, "step_ids", []) or [],
            "is_answer": getattr(m, "is_answer", False) or False,
            "is_final": getattr(m, "is_final", False) or False,
        }
        for m in sorted(session.messages, key=lambda x: x.timestamp)
    ]
    out["steps"] = [
        {
            "id": s.id,
            "status": s.status,
            "thought": s.thought,
            "action_str": s.action_str,
            "action_description": s.action_description,
            "url": s.url,
            "screenshot_base64": s.screenshot_base64,
            "thumbnail_base64": s.thumbnail_base64,
            "error": s.error,
            "created_at": s.created_at.isoformat(),
        }
        for s in sorted(session.steps, key=lambda x: x.id)
    ]
    return out


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    """Delete a session (must belong to current user)."""
    user_id = get_current_user_id(request)
    if not session_manager.delete_session(session_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.delete("/api/sessions")
def delete_all_sessions(request: Request):
    """Delete all sessions for the current user."""
    user_id = get_current_user_id(request)
    count = session_manager.delete_all_sessions(user_id=user_id)
    return {"status": "deleted", "count": count}


@router.post("/api/sessions/{session_id}/pause")
def pause_session(session_id: str, request: Request):
    """Pause a running session."""
    user_id = get_current_user_id(request)
    session = session_manager.get_session_light(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session_manager.update_session_status(session_id, "paused")
    return {"status": "paused", "session_id": session_id}


@router.post("/api/sessions/{session_id}/resume")
def resume_session(session_id: str, request: Request):
    """Resume a paused session."""
    user_id = get_current_user_id(request)
    session = session_manager.get_session_light(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "paused":
        raise HTTPException(status_code=400, detail="Session is not paused")
    session_manager.update_session_status(session_id, "running")
    return {"status": "running", "session_id": session_id}


# ========================================================================
# Share Routes
# ========================================================================

@router.post("/api/sessions/{session_id}/share")
def share_session(session_id: str, request: Request):
    """Generate a shareable link for a session."""
    user_id = get_current_user_id(request)
    token = session_manager.share_session(session_id, user_id)
    if token is None:
        raise HTTPException(status_code=404, detail="Session not found")
    share_url = f"{request.base_url}shared/{token}".replace("http://", "https://", 1)
    return {"share_token": token, "share_url": share_url}


@router.delete("/api/sessions/{session_id}/share")
def unshare_session(session_id: str, request: Request):
    """Remove sharing from a session."""
    user_id = get_current_user_id(request)
    if not session_manager.unshare_session(session_id, user_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "unshared", "session_id": session_id}


@router.get("/api/shared/{share_token}")
def get_shared_session(share_token: str):
    """Public endpoint: view a shared session (no auth required)."""
    session = session_manager.get_shared_session(share_token)
    if not session:
        raise HTTPException(status_code=404, detail="Shared session not found")
    out = session_to_response(session)
    out["messages"] = [
        {
            "id": m.id,
            "type": m.type,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
            "step_ids": getattr(m, "step_ids", []) or [],
            "is_answer": getattr(m, "is_answer", False) or False,
            "is_final": getattr(m, "is_final", False) or False,
        }
        for m in sorted(session.messages, key=lambda x: x.timestamp)
    ]
    out["steps"] = [
        {
            "id": s.id,
            "status": s.status,
            "thought": s.thought,
            "action_str": s.action_str,
            "action_description": s.action_description,
            "url": s.url,
            "screenshot_base64": s.screenshot_base64,
            "thumbnail_base64": s.thumbnail_base64,
            "error": s.error,
            "created_at": s.created_at.isoformat(),
        }
        for s in sorted(session.steps, key=lambda x: x.id)
    ]
    return out


# ========================================================================
# Step Routes
# ========================================================================

@router.get("/api/sessions/{session_id}/steps")
def get_steps(session_id: str, request: Request):
    """Get all steps for a session."""
    user_id = get_current_user_id(request)
    session = session_manager.get_session(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    steps = [
        {
            "id": s.id,
            "status": s.status,
            "thought": s.thought,
            "action_preview": s.action_str,
            "action_description": s.action_description,
            "url": s.url,
            "created_at": s.created_at.isoformat(),
        }
        for s in session.steps
    ]

    return {"steps": steps}


@router.get("/api/sessions/{session_id}/steps/{step_id}")
def get_step(session_id: str, step_id: int, request: Request):
    """Get full step details including screenshot."""
    user_id = get_current_user_id(request)
    session = session_manager.get_session(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    step = session.get_step(step_id)
    if not step:
        raise HTTPException(status_code=404, detail="Step not found")

    action_data = None
    if step.action:
        action_data = step.action

    return {
        "id": step.id,
        "status": step.status,
        "thought": step.thought,
        "action": action_data,
        "action_str": step.action_str,
        "action_description": step.action_description,
        "url": step.url,
        "screenshot_base64": step.screenshot_base64,
        "error": step.error,
        "created_at": step.created_at.isoformat(),
    }


@router.get("/api/whitelist")
def get_whitelist():
    """Local demo: no URL whitelist; kept for API compatibility."""
    return {"hosts": []}


# ========================================================================
# User Consent Routes
# ========================================================================

class ConsentBody(BaseModel):
    open_science_consent: bool = False


@router.post("/api/consent")
def save_consent(body: ConsentBody, request: Request):
    """Record the user's terms acceptance and open-science consent choice."""
    user_id = get_current_user_id(request)
    with get_db() as db:
        existing = db.query(UserConsentModel).filter_by(user_id=user_id).first()
        if existing:
            existing.terms_accepted = True
            existing.open_science_consent = body.open_science_consent
            existing.updated_at = datetime.utcnow()
        else:
            db.add(UserConsentModel(
                user_id=user_id,
                terms_accepted=True,
                open_science_consent=body.open_science_consent,
            ))
    return {"status": "ok"}


@router.get("/api/consent")
def get_consent(request: Request):
    """Get the current user's consent status."""
    user_id = get_current_user_id(request)
    with get_db() as db:
        row = db.query(UserConsentModel).filter_by(user_id=user_id).first()
        if not row:
            return {"terms_accepted": False, "open_science_consent": False}
        return {
            "terms_accepted": row.terms_accepted,
            "open_science_consent": row.open_science_consent,
        }
