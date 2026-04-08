"""FastAPI application with Socket.IO support."""

from contextlib import asynccontextmanager
import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

from demo import glog
from demo.api import router
from demo.error import handle as http_exception_handler
from demo.socket_events import sio_asgi_app
from demo.config import settings
from demo.database import engine, Base
from demo import models  # Import models to register them with Base
from demo.mem_util import log_memory
from demo.migrations.add_user_id_to_sessions import run as run_user_id_migration
from demo.migrations.add_answer_flags_to_chat_messages import run as run_answer_flags_migration
from demo.migrations.add_share_token_to_sessions import run as run_share_token_migration
from demo.migrations.add_is_safe_to_chat_messages import run as run_is_safe_migration
from demo.migrations.add_user_consent_table import run as run_user_consent_migration


# Configure logging at module level (before app creation)
fmt = os.getenv("LOG_FORMAT")
handlers = [glog.Handler()] if fmt == "google:json" else []
level = os.environ.get("LOG_LEVEL", default=logging.INFO)
logging.basicConfig(level=level, handlers=handlers)

# Silence verbose loggers
logging.getLogger('engineio.server').setLevel(logging.WARNING)
logging.getLogger('engineio.client').setLevel(logging.WARNING)
logging.getLogger('socketio.server').setLevel(logging.WARNING)
logging.getLogger('socketio.client').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('browserbase').setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # Initialize database tables
    Base.metadata.create_all(bind=engine)
    # Run migrations
    try:
        run_user_id_migration()
    except Exception as e:
        logging.warning("Migration add_user_id_to_sessions: %s", e)
    try:
        run_answer_flags_migration()
    except Exception as e:
        logging.warning("Migration add_answer_flags_to_chat_messages: %s", e)
    try:
        run_share_token_migration()
    except Exception as e:
        logging.warning("Migration add_share_token_to_sessions: %s", e)
    try:
        run_is_safe_migration()
    except Exception as e:
        logging.warning("Migration add_is_safe_to_chat_messages: %s", e)
    try:
        run_user_consent_migration()
    except Exception as e:
        logging.warning("Migration add_user_consent_table: %s", e)
    logging.info("Database tables initialized")

    logging.info("FastAPI app initialized with Socket.IO support")
    logging.info(f"CORS origins: {settings.CORS_ORIGINS}")
    log_memory("app_startup_complete")

    yield  # App is running

    # Shutdown: close all active browsers and cancel timers
    from demo.socket_events import (
        active_browsers,
        active_agent_threads,
        stop_flags,
        cleanup_timers,
        grace_timers,
        followup_contexts,
        pending_new_task_per_session,
        cancel_cleanup_timer,
    )
    log_memory("app_shutdown_start")
    for sid in list(active_browsers.keys()):
        try:
            active_browsers[sid].close()
        except Exception:
            pass
    active_browsers.clear()
    for sid in list(cleanup_timers.keys()) + list(grace_timers.keys()):
        cancel_cleanup_timer(sid)
    active_agent_threads.clear()
    stop_flags.clear()
    followup_contexts.clear()
    pending_new_task_per_session.clear()
    log_memory("app_shutdown_complete")


app = FastAPI(title="WebOlmo Demo", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
app.add_exception_handler(HTTPException, http_exception_handler)

# REST API routes
app.include_router(router)

# Mount Socket.IO at /socket.io (the default path)
app.mount("/socket.io", sio_asgi_app)


if __name__ == "__main__":
    import uvicorn

    print(f"""
╔════════════════════════════════════════════════════════════╗
║                WebOlmo Demo - Backend Server               ║
╠════════════════════════════════════════════════════════════╣
║  API:           http://localhost:{settings.PORT}                        ║
║  Socket.IO:     ws://localhost:{settings.PORT}/socket.io                ║
║  Modal:         {((settings.MODAL_ENDPOINT or "(not set)")[:40])}...  ║
╚════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
