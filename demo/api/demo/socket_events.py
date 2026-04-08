"""Socket.IO event handlers for real-time agent communication.

This module uses python-socketio (ASGI) for Socket.IO events with FastAPI.
Background threads call _emit() which schedules the coroutine on the main
event loop.
"""

import asyncio
import logging
from threading import Thread, Timer
import time

import socketio as sio_module

from demo.session_manager import session_manager
from demo.browser_service import BrowserService
from demo.config import settings
from demo.mem_util import log_memory, force_gc_and_log

logger = logging.getLogger(__name__)

# Store active browser services per session
active_browsers: dict[str, BrowserService] = {}
active_agent_threads: dict[str, Thread] = {}
stop_flags: dict[str, bool] = {}
keep_browser_on_stop: dict[str, bool] = {}  # Stop agent only; do not close browser
cleanup_timers: dict[str, Timer] = {}
grace_timers: dict[str, Timer] = {}
followup_contexts: dict[str, dict] = {}  # Store follow-up context per session
# When user sends a message while paused, store (goal, context) for the agent loop to apply
pending_new_task_per_session: dict[str, tuple[str, dict]] = {}
sid_to_session_id: dict[str, str] = {}
# Map socket sid → user_id (captured at connect from HTTP headers)
sid_to_user_id: dict[str, str] = {}

# Inactivity timeout in seconds (warning fires after this, then grace period before close)
BROWSER_INACTIVITY_TIMEOUT = 60
BROWSER_GRACE_PERIOD = 30

# Create an AsyncServer for ASGI compatibility with FastAPI/uvicorn
sio = sio_module.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.CORS_ORIGINS,
    logger=False,
    engineio_logger=False,
)

# ASGI app to mount in FastAPI
sio_asgi_app = sio_module.ASGIApp(sio)

# Reference to the running event loop, captured at first connect
_loop: asyncio.AbstractEventLoop | None = None


def _emit(*args, **kwargs):
    """Thread-safe emit: schedule sio.emit() on the async event loop.

    Used by background threads (agent loop, cleanup timers) that cannot await.
    """
    global _loop
    if _loop is None or _loop.is_closed():
        logger.warning("No event loop available for emit")
        return
    asyncio.run_coroutine_threadsafe(sio.emit(*args, **kwargs), _loop)


def _get_user_id_for_sid(sid: str) -> str:
    """Get user_id stored at connect time for the given socket sid."""
    return sid_to_user_id.get(sid, "demo.user@allenai.org")


def _session_for_user(session_id: str, sid: str):
    """Get session only if it belongs to the user behind this socket; otherwise None."""
    user_id = _get_user_id_for_sid(sid)
    return session_manager.get_session_light(session_id, user_id=user_id)


# ============================================================================
# Socket.IO event handlers (async, run on the main event loop)
# ============================================================================

@sio.event
async def connect(sid, environ, auth=None):
    """Handle client connection."""
    global _loop
    if _loop is None:
        _loop = asyncio.get_running_loop()

    # Extract user from ASGI scope headers (list of [name, value] byte pairs)
    scope = environ.get("asgi.scope", {})
    raw_headers = scope.get("headers", [])
    email = None
    for name, value in raw_headers:
        header_name = name.decode() if isinstance(name, bytes) else name
        if header_name.lower() == "x-auth-request-email":
            email = value.decode() if isinstance(value, bytes) else value
            break
    if not email:
        # Fallback: WSGI-style keys set by some adapters
        email = environ.get("HTTP_X_AUTH_REQUEST_EMAIL") or "demo.user@allenai.org"
    sid_to_user_id[sid] = email
    logger.info(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    """Handle client disconnection - cleanup resources."""
    logger.info(f"Client disconnected: {sid}")
    sid_to_user_id.pop(sid, None)
    session_id = sid_to_session_id.pop(sid, None)
    if session_id:
        stop_session_and_browser(session_id, reason="tab_closed")


@sio.event
async def join_session(sid, data):
    """
    Client joins a session room for receiving updates.

    Expected data: { session_id: string }
    """
    session_id = data.get('session_id')
    if not session_id:
        await sio.emit('error', {'message': 'session_id required'}, to=sid)
        return

    session = _session_for_user(session_id, sid)
    if not session:
        await sio.emit('error', {'message': 'Session not found'}, to=sid)
        return

    # Join the session room and track this socket for cleanup on disconnect
    await sio.enter_room(sid, session_id)
    sid_to_session_id[sid] = session_id
    logger.info(f"Client {sid} joined session {session_id}")

    # If session is running and no browser yet, start the agent
    if session.status == "running" and session_id not in active_browsers:
        start_agent_loop(session_id)

    await sio.emit('joined', {'session_id': session_id, 'status': session.status}, to=sid)


@sio.event
async def leave_session(sid, data):
    """Client leaves a session room (stops agent and closes browser for that session)."""
    session_id = data.get('session_id')
    if session_id:
        if not _session_for_user(session_id, sid):
            logger.warning(f"Client {sid} tried to leave session {session_id} (not owner)")
            return
        if sid_to_session_id.get(sid) == session_id:
            del sid_to_session_id[sid]
        await sio.leave_room(sid, session_id)
        stop_session_and_browser(session_id, reason="user")
        logger.info(f"Client {sid} left session {session_id}")


@sio.event
async def extend_session(sid, data):
    """User clicked 'Continue Session' to dismiss the inactivity warning."""
    session_id = data.get('session_id')
    if not session_id or not _session_for_user(session_id, sid):
        return

    cancel_cleanup_timer(session_id)

    if session_id in active_browsers:
        start_cleanup_timer(session_id)

    await sio.emit('inactivity_warning_cancelled', {}, room=session_id)
    logger.info(f"Session {session_id} extended by user")


@sio.event
async def user_message(sid, data):
    """
    Handle user sending a message/task.

    Expected data: { session_id: string, text: string }
    """
    session_id = data.get('session_id')
    text = data.get('text', '')

    if not session_id or not text:
        await sio.emit('error', {'message': 'session_id and text required'}, to=sid)
        return

    session = _session_for_user(session_id, sid)
    if not session:
        await sio.emit('error', {'message': 'Session not found'}, to=sid)
        return

    # Cancel any pending cleanup timer - user is active
    had_grace_timer = session_id in grace_timers
    cancel_cleanup_timer(session_id)
    if had_grace_timer:
        await sio.emit('inactivity_warning_cancelled', {}, room=session_id)

    # Check if this is a follow-up (session was completed/stopped)
    is_followup = session.status in ("completed", "stopped")

    # Add user message to chat
    session_manager.add_message(session_id, "user", text)

    if is_followup:
        # Need full session with steps for followup context (last screenshot)
        full_session = session_manager.get_session(session_id)
        context = full_session.get_followup_context(text) if full_session else {"goal": text}

        # Update session status in database
        session_manager.update_session_status(session_id, "running")

        # Check if we can reuse existing browser
        if session_id in active_browsers:
            browser = active_browsers[session_id]
            if browser.is_running and browser.thread and browser.thread.is_alive():
                # Reuse existing browser session
                logger.info(f"Reusing existing browser for follow-up in session {session_id}")
                result = browser.start_new_task(context["goal"], context)

                if result.get("error"):
                    logger.error(f"Failed to start new task on existing browser: {result.get('error')}")
                    # Fall through to create new browser
                else:
                    # Browser ready, emit state and start agent loop with existing browser
                    await sio.emit('browser_ready', {
                        'url': result.get('url'),
                        'screenshot_base64': result.get('screenshot_base64'),
                    }, to=sid)

                    # Start agent loop with existing browser
                    start_agent_loop(session_id, existing_browser=browser, goal=context["goal"])
                    return

        # Store context for new browser service if we couldn't reuse
        followup_contexts[session_id] = context
        logger.info(f"Follow-up query for session {session_id} (with context screenshot)")

    # User sent a message while paused (Take Control): queue new goal for agent loop and resume
    session = _session_for_user(session_id, sid)
    if session and session.status == "paused" and session_id in active_browsers:
        browser = active_browsers[session_id]
        if browser.is_running and browser.thread and browser.thread.is_alive():
            full_session = session_manager.get_session(session_id)
            context = full_session.get_followup_context(text) if full_session else {"goal": text}
            pending_new_task_per_session[session_id] = (context["goal"], context)
            session_manager.update_session_status(session_id, "running")
            await sio.emit('session_resumed', {}, room=session_id)
            logger.info(f"Session {session_id}: new task from chat while paused, queued for agent loop")

    # Start agent loop if not running
    if session_id not in active_agent_threads or not active_agent_threads[session_id].is_alive():
        start_agent_loop(session_id)


@sio.event
async def pause(sid, data):
    """Pause the session."""
    session_id = data.get('session_id')
    if not session_id:
        return
    if not _session_for_user(session_id, sid):
        return

    session_manager.update_session_status(session_id, "paused")
    session_manager.log_session_event(session_id, "take_control")
    await sio.emit('session_paused', {}, room=session_id)
    logger.info(f"Session {session_id} paused")


@sio.event
async def resume(sid, data):
    """Resume a paused session."""
    session_id = data.get('session_id')
    if not session_id:
        return

    session = _session_for_user(session_id, sid)
    if session and session.status == "paused":
        session_manager.update_session_status(session_id, "running")
        session_manager.log_session_event(session_id, "resume")
        await sio.emit('session_resumed', {}, room=session_id)
        logger.info(f"Session {session_id} resumed")


@sio.event
async def navigate_to_blank(sid, data):
    """
    Pause the session (quit current step like Take Control) and navigate browser to about:blank.
    """
    session_id = data.get('session_id')
    if not session_id:
        return

    session = _session_for_user(session_id, sid)
    if not session:
        return

    # Pause session so UI shows "You're in control" and next step is cancelled
    session_manager.update_session_status(session_id, "paused")
    await sio.emit('session_paused', {}, room=session_id)
    logger.info(f"Session {session_id} paused for navigate to blank")

    # Navigate browser to about:blank
    if session_id in active_browsers:
        browser = active_browsers[session_id]
        result = browser.navigate("about:blank")
        if result and not result.get("error"):
            await sio.emit(
                'browser_ready',
                {
                    'url': result.get('url', 'about:blank'),
                    'screenshot_base64': result.get('screenshot_base64'),
                },
                room=session_id,
            )
            logger.info(f"Session {session_id} navigated to about:blank")
        elif result and result.get("error"):
            logger.warning(f"Session {session_id} navigate failed: {result.get('error')}")


@sio.event
async def stop_agent(sid, data):
    """Stop the agent loop only; keep browser and conversation. No resume."""
    session_id = data.get('session_id')
    if not session_id:
        return
    session = _session_for_user(session_id, sid)
    if not session:
        return
    cancel_cleanup_timer(session_id)
    stop_flags[session_id] = True
    keep_browser_on_stop[session_id] = True
    session_manager.update_session_status(session_id, "stopped")
    logger.info(f"Session {session_id}: stop_agent (browser kept open)")


# ============================================================================
# Sync helpers (called from background threads or sync contexts)
# ============================================================================

def stop_session_and_browser(session_id: str, reason: str = "user"):
    """Stop the agent and close the browser for a session. Used by stop and disconnect."""
    cancel_cleanup_timer(session_id)
    stop_flags[session_id] = True
    session_manager.update_session_status(session_id, "stopped")
    pending_new_task_per_session.pop(session_id, None)
    followup_contexts.pop(session_id, None)
    if session_id in active_browsers:
        try:
            active_browsers[session_id].close()
        except Exception:
            pass
        active_browsers.pop(session_id, None)
    if session_id not in active_agent_threads:
        stop_flags.pop(session_id, None)
    _emit('session_stopped', {'reason': reason}, room=session_id)
    logger.info(f"Session {session_id} stopped (reason={reason})")


def cleanup_browser_session(session_id: str):
    """Clean up browser session and notify user."""
    # Clean up grace timer reference
    if session_id in grace_timers:
        del grace_timers[session_id]

    if session_id in active_browsers:
        try:
            active_browsers[session_id].close()
        except:
            pass
        active_browsers.pop(session_id, None)

        _emit('browser_closed', {
            'message': 'Browser session closed due to inactivity.',
        }, room=session_id)
        logger.info(f"Browser closed for session {session_id} due to inactivity")

    stop_flags.pop(session_id, None)

    # Clean up timer reference
    if session_id in cleanup_timers:
        del cleanup_timers[session_id]


def warn_inactivity(session_id: str):
    """Send an inactivity warning and start the grace period before closing."""
    # Clean up warning timer reference
    if session_id in cleanup_timers:
        del cleanup_timers[session_id]

    if session_id not in active_browsers:
        return

    _emit('inactivity_warning', {
        'countdown': BROWSER_GRACE_PERIOD,
    }, room=session_id)
    logger.info(f"Sent inactivity warning for session {session_id}, "
                f"{BROWSER_GRACE_PERIOD}s grace period")

    # Start grace period timer
    timer = Timer(BROWSER_GRACE_PERIOD, cleanup_browser_session, args=[session_id])
    timer.daemon = True
    timer.start()
    grace_timers[session_id] = timer


def cancel_cleanup_timer(session_id: str):
    """Cancel any pending cleanup/grace timers for a session."""
    if session_id in cleanup_timers:
        cleanup_timers[session_id].cancel()
        del cleanup_timers[session_id]
        logger.info(f"Cancelled cleanup timer for session {session_id}")
    if session_id in grace_timers:
        grace_timers[session_id].cancel()
        del grace_timers[session_id]
        logger.info(f"Cancelled grace timer for session {session_id}")


def start_cleanup_timer(session_id: str):
    """Start a timer that warns the user after inactivity, then closes after grace period."""
    cancel_cleanup_timer(session_id)

    timer = Timer(BROWSER_INACTIVITY_TIMEOUT, warn_inactivity, args=[session_id])
    timer.daemon = True
    timer.start()
    cleanup_timers[session_id] = timer
    logger.info(f"Started {BROWSER_INACTIVITY_TIMEOUT}s cleanup timer for session {session_id}")


def start_agent_loop(
    session_id: str,
    existing_browser: BrowserService | None = None,
    goal: str | None = None,
):
    """Start the agent loop in a background thread.

    Args:
        session_id: The session ID
        existing_browser: Optional existing browser to reuse for follow-ups
        goal: Optional goal override (used when reusing browser)
    """

    # Cancel any pending cleanup timer (user is active)
    cancel_cleanup_timer(session_id)

    # Reset stop flag
    stop_flags[session_id] = False

    def run_agent():
        """Agent loop that runs in a background thread."""
        nonlocal goal

        log_memory(f"agent_loop_start:{session_id[:8]}")

        try:
            session = session_manager.get_session_light(session_id)
            if not session:
                return

            # Check if we're reusing an existing browser
            if existing_browser:
                browser = existing_browser
                # goal was already passed in
                if not goal:
                    goal = session.goal
                logger.info(f"Reusing existing browser for session {session_id}")
            else:
                # Send initializing status
                _emit('status', {
                    'status': 'initializing',
                    'message': 'Starting browser session...',
                }, room=session_id)

                # Get follow-up context if available
                context = followup_contexts.pop(session_id, None)

                # Use goal from context if available (for follow-ups), otherwise use session goal
                goal = context["goal"] if context else session.goal

                # Create and initialize browser
                browser = BrowserService(
                    browserbase_api_key=settings.BROWSERBASE_API_KEY or "",
                    browserbase_project_id=settings.BROWSERBASE_PROJECT_ID or "",
                    start_url=session.start_url,
                    goal=goal,
                    context=context,
                    session_id=session_id,
                )

                try:
                    init_result = browser.initialize()
                    active_browsers[session_id] = browser
                except Exception as e:
                    logger.error(f"Failed to initialize browser: {e}")
                    _emit('error', {
                        'message': f'Failed to start browser: {str(e)}',
                    }, room=session_id)
                    return

                # Send browser ready
                _emit('browser_ready', {
                    'url': init_result.get('url'),
                    'screenshot_base64': init_result.get('screenshot_base64'),
                    'live_view_url': init_result.get('live_view_url'),
                }, room=session_id)

            max_steps = settings.MAX_STEPS_PER_SESSION
            step_num = 1
            current_step_ids = []
            step = None

            while step_num <= max_steps:
                # Check stop flag
                if stop_flags.get(session_id, False):
                    if keep_browser_on_stop.get(session_id):
                        _emit('agent_stopped', {'reason': 'user'}, room=session_id)
                    else:
                        _emit('session_stopped', {'reason': 'user'}, room=session_id)
                    break

                # Check session status
                session = session_manager.get_session_light(session_id)
                if not session or session.status == "stopped":
                    break

                # Handle pause
                while session and session.status == "paused":
                    time.sleep(0.5)
                    session = session_manager.get_session_light(session_id)
                    if stop_flags.get(session_id, False):
                        break

                if not session or session.status == "stopped" or stop_flags.get(session_id, False):
                    if stop_flags.get(session_id, False):
                        if keep_browser_on_stop.get(session_id):
                            _emit('agent_stopped', {'reason': 'user'}, room=session_id)
                        else:
                            _emit('session_stopped', {'reason': 'user'}, room=session_id)
                    break

                # Apply pending new task from chat-while-paused (this thread waits; worker is idle here)
                pending = pending_new_task_per_session.pop(session_id, None)
                if pending:
                    goal, context = pending
                    result = browser.start_new_task(goal, context)
                    if result.get("error"):
                        logger.error(f"Failed to apply pending new task: {result.get('error')}")
                    else:
                        _emit('browser_ready', {
                            'url': result.get('url'),
                            'screenshot_base64': result.get('screenshot_base64'),
                        }, room=session_id)
                        logger.info(f"Session {session_id}: applied pending new task (context + latest input)")

                # Add new step
                step = session_manager.add_step(session_id)
                if not step:
                    break

                # Check stop before showing or running this step
                if stop_flags.get(session_id, False):
                    session_manager.update_step(
                        session_id=session_id,
                        step_id=step.id,
                        status='cancelled',
                    )
                    _emit('step_cancelled', {
                        'session_id': session_id,
                        'step_id': step.id,
                        'message': 'Stopped by user',
                    }, room=session_id)
                    if keep_browser_on_stop.get(session_id):
                        _emit('agent_stopped', {'reason': 'user'}, room=session_id)
                    else:
                        _emit('session_stopped', {'reason': 'user'}, room=session_id)
                    break

                # Send step started
                _emit('step_started', {
                    'session_id': session_id,
                    'step_id': step.id,
                    'thought': '',
                    'action_preview': '',
                }, room=session_id)

                # Update status
                _emit('status', {
                    'status': 'predicting',
                    'message': 'Predicting the next move...',
                }, room=session_id)

                # Action preview callback
                def on_action_preview(preview):
                    _emit('step_update', {
                        'session_id': session_id,
                        'step_id': step.id,
                        'thought': preview.get('thought', ''),
                        'action_str': preview.get('action_str', ''),
                        'action_name': preview.get('action_name', ''),
                        'action_description': preview.get('action_description', ''),
                    }, room=session_id)

                    if preview.get('click_coords'):
                        _emit('action_preview', {
                            'session_id': session_id,
                            'step_id': step.id,
                            'click_coords': preview.get('click_coords'),
                        }, room=session_id)

                # Check stop again right before running the step
                if stop_flags.get(session_id, False):
                    session_manager.update_step(
                        session_id=session_id,
                        step_id=step.id,
                        status='cancelled',
                    )
                    _emit('step_cancelled', {
                        'session_id': session_id,
                        'step_id': step.id,
                        'message': 'Stopped by user',
                    }, room=session_id)
                    if keep_browser_on_stop.get(session_id):
                        _emit('agent_stopped', {'reason': 'user'}, room=session_id)
                    else:
                        _emit('session_stopped', {'reason': 'user'}, room=session_id)
                    break

                # Get prediction and execute
                result = browser.predict_and_execute(on_action_preview=on_action_preview)

                # User clicked stop while we were in predict_and_execute
                if stop_flags.get(session_id, False):
                    session_manager.update_step(
                        session_id=session_id,
                        step_id=step.id,
                        status='cancelled',
                    )
                    _emit('step_cancelled', {
                        'session_id': session_id,
                        'step_id': step.id,
                        'message': 'Stopped by user',
                    }, room=session_id)
                    if keep_browser_on_stop.get(session_id):
                        _emit('agent_stopped', {'reason': 'user'}, room=session_id)
                    else:
                        _emit('session_stopped', {'reason': 'user'}, room=session_id)
                    break

                if result.get('cancelled'):
                    session_manager.update_step(
                        session_id=session_id,
                        step_id=step.id,
                        status='cancelled',
                    )
                    _emit('step_cancelled', {
                        'session_id': session_id,
                        'step_id': step.id,
                        'message': 'Paused by user',
                    }, room=session_id)
                    continue


                _emit('status', {
                    'status': 'executing',
                    'message': f'Step {step_num}: Executing action...',
                }, room=session_id)

                if result.get('error'):
                    session_manager.update_step(
                        session_id=session_id,
                        step_id=step.id,
                        status='failed',
                        error=result.get('error'),
                    )
                    _emit('step_failed', {
                        'session_id': session_id,
                        'step_id': step.id,
                        'error': result.get('error'),
                    }, room=session_id)
                    session_manager.update_session_status(session_id, 'stopped')
                    _emit('session_stopped', {'reason': 'error'}, room=session_id)
                    break

                # Update step
                session_manager.update_step(
                    session_id=session_id,
                    step_id=step.id,
                    thought=result.get('thought', ''),
                    action={'name': result.get('action_name', 'unknown')},
                    action_str=result.get('action_str', ''),
                    action_description=result.get('action_description', ''),
                    url=result.get('url', ''),
                    screenshot_base64=result.get('screenshot_base64'),
                    status='completed',
                )
                current_step_ids.append(step.id)

                # Send step completed
                step_data = {
                    'session_id': session_id,
                    'step_id': step.id,
                    'success': True,
                    'thought': result.get('thought', ''),
                    'action_str': result.get('action_str', ''),
                    'action_name': result.get('action_name', ''),
                    'action_description': result.get('action_description', ''),
                    'url': result.get('url', ''),
                    'screenshot_base64': result.get('screenshot_base64'),
                    'annotated_screenshot': result.get('annotated_screenshot'),
                    'click_coords': result.get('click_coords'),
                }
                if result.get('live_view_url'):
                    step_data['live_view_url'] = result.get('live_view_url')
                _emit('step_completed', step_data, room=session_id)
                step = None

                # Emit agent_message for all SendMsgToUser actions
                if result.get('agent_message'):
                    is_answer = result.get('is_answer', False)
                    is_final = result.get('is_final', False)
                    session_manager.add_message(session_id, 'agent', result['agent_message'], step_ids=list(current_step_ids), is_answer=is_answer, is_final=is_final)
                    session.add_message('agent', result['agent_message'], step_ids=list(current_step_ids), is_answer=is_answer, is_final=is_final)
                    current_step_ids = []
                    _emit('agent_message', {
                        'session_id': session_id,
                        'text': result['agent_message'],
                        'is_answer': result.get('is_answer', False),
                        'is_final': result.get('is_final', False),
                    }, room=session_id)

                # Check if task complete (triggered by [EXIT])
                if result.get('is_final'):
                    session_manager.update_session_status(session_id, 'completed')

                    final_message = result.get('final_message') or 'Task completed.'

                    _emit('session_complete', {
                        'session_id': session_id,
                        'summary': final_message,
                    }, room=session_id)
                    break

                step_num += 1
                time.sleep(0.1)

            if step_num > max_steps:
                # Max steps reached
                session_manager.update_session_status(session_id, 'stopped')
                _emit('session_stopped', {'reason': 'max_steps'}, room=session_id)

        except Exception as e:
            logger.exception(f"Agent loop error for session {session_id}")
            if step and step.status == "running":
                session_manager.update_step(
                    session_id=session_id,
                    step_id=step.id,
                    status='failed',
                    error=str(e),
                )
                _emit('step_failed', {
                    'session_id': session_id,
                    'step_id': step.id,
                    'error': str(e),
                }, room=session_id)
            _emit('error', {'message': str(e)}, room=session_id)
            # Close browser immediately on error
            if session_id in active_browsers:
                try:
                    active_browsers[session_id].close()
                except:
                    pass
                active_browsers.pop(session_id, None)

        finally:
            # Safety net: if session is still "running" but agent loop is ending,
            # update to "stopped" so join_session won't auto-restart a ghost loop
            try:
                sess = session_manager.get_session_light(session_id)
                if sess and sess.status == "running":
                    session_manager.update_session_status(session_id, "stopped")
                    _emit('session_stopped', {'reason': 'error'}, room=session_id)
            except Exception:
                logger.exception(f"Failed to update stale session status for {session_id}")

            # Clean up stop flag
            if session_id in stop_flags:
                was_stopped = stop_flags.get(session_id, False)
                keep_browser = keep_browser_on_stop.pop(session_id, False)
                del stop_flags[session_id]

                # If user explicitly stopped, close browser only when not keep_browser_on_stop
                if was_stopped and not keep_browser and session_id in active_browsers:
                    try:
                        active_browsers[session_id].close()
                    except:
                        pass
                    active_browsers.pop(session_id, None)

            # If browser still exists (task completed normally), start cleanup timer
            if session_id in active_browsers:
                start_cleanup_timer(session_id)

            active_agent_threads.pop(session_id, None)
            force_gc_and_log(f"agent_loop_end:{session_id[:8]}")

    # Start the agent thread
    thread = Thread(target=run_agent, daemon=True)
    active_agent_threads[session_id] = thread
    thread.start()
