/**
 * TypeScript types for the WebOlmo demo.
 */

// Session status
export type SessionStatus = 'idle' | 'running' | 'paused' | 'completed' | 'stopped';

// Step
export interface Step {
    id: number;
    status: 'running' | 'completed' | 'failed' | 'cancelled';
    thought?: string;
    actionStr?: string;
    actionName?: string;
    actionDescription?: string;
    url?: string;
    screenshotBase64?: string;
    annotatedScreenshot?: string; // Screenshot with click indicator
    thumbnailBase64?: string;
    clickCoords?: { x: number; y: number };
    error?: string;
    createdAt: string;
}

// Chat message types
export type MessageType = 'user' | 'agent' | 'system' | 'step';

export interface ChatMessage {
    id: string;
    type: MessageType;
    content: string;
    timestamp: string;
    // For step messages
    step?: Step;
    // For final answer messages
    isFinal?: boolean;
    // For [ANSWER] messages
    isAnswer?: boolean;
}

// Session
export interface Session {
    id: string;
    goal: string;
    startUrl: string;
    status: SessionStatus;
    messages: ChatMessage[];
    createdAt: string;
    updatedAt: string;
    shareToken?: string | null;
}

// Session list item (from GET /api/sessions)
export interface SessionListItem {
    id: string;
    goal: string;
    start_url: string;
    status: string;
    step_count: number;
    created_at: string;
    updated_at: string;
}

// WebSocket message types (server -> client)
export interface WSStatus {
    type: 'status';
    status: string;
    message: string;
}

export interface WSBrowserReady {
    type: 'browser_ready';
    url?: string;
    screenshot_base64?: string;
    live_view_url?: string;
}

export interface WSAgentMessage {
    type: 'agent_message';
    text: string;
    is_final?: boolean;
    is_answer?: boolean;
}

export interface WSStepStarted {
    type: 'step_started';
    step_id: number;
    thought: string;
    action_preview: string;
}

export interface WSActionPreview {
    type: 'action_preview';
    step_id: number;
    click_coords?: { x: number; y: number };
}

export interface WSStepCompleted {
    type: 'step_completed';
    step_id: number;
    success: boolean;
    thought?: string;
    action_str?: string;
    action_name?: string;
    action_description?: string;
    url?: string;
    screenshot_base64?: string;
    annotated_screenshot?: string; // Screenshot with click indicator
    thumbnail_base64?: string;
    click_coords?: { x: number; y: number };
}

export interface WSStepUpdate {
    type: 'step_update';
    step_id: number;
    branch: string;
    thought?: string;
    action_str?: string;
    action_name?: string;
    action_description?: string;
}

export interface WSStepFailed {
    type: 'step_failed';
    step_id: number;
    error: string;
}

export interface WSStepCancelled {
    type: 'step_cancelled';
    step_id: number;
    message?: string;
}

export interface WSSessionPaused {
    type: 'session_paused';
}

export interface WSSessionResumed {
    type: 'session_resumed';
}

export interface WSSessionComplete {
    type: 'session_complete';
    summary: string;
}

export interface WSSessionStopped {
    type: 'session_stopped';
    reason: 'user' | 'error' | 'max_steps';
}

export interface WSAgentStopped {
    type: 'agent_stopped';
    reason: string;
}

export interface WSError {
    type: 'error';
    message: string;
    details?: string;
}

export type WSServerMessage =
    | WSStatus
    | WSBrowserReady
    | WSAgentMessage
    | WSStepStarted
    | WSActionPreview
    | WSStepCompleted
    | WSStepFailed
    | WSStepCancelled
    | WSStepUpdate
    | WSSessionPaused
    | WSSessionResumed
    | WSSessionComplete
    | WSSessionStopped
    | WSAgentStopped
    | WSError;

// WebSocket message types (client -> server)
export interface WSUserMessagePayload {
    type: 'user_message';
    session_id: string;
    text: string;
}

export interface WSPausePayload {
    type: 'pause';
    session_id: string;
}

export interface WSResumePayload {
    type: 'resume';
    session_id: string;
}

export interface WSStopAgentPayload {
    type: 'stop_agent';
    session_id: string;
}

export type WSClientMessage =
    | WSUserMessagePayload
    | WSPausePayload
    | WSResumePayload
    | WSStopAgentPayload;
