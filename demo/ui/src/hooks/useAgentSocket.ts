/**
 * Socket.IO hook for agent communication with auto-reconnection.
 *
 * This replaces the native WebSocket implementation from demo-prod
 * with Socket.IO client to work with the FastAPI + Socket.IO backend.
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

import { useSessionStore } from '@/stores/sessionStore';
import type { WSServerMessage, Step } from '@/types';

// Socket.IO connects to the same origin by default (through nginx proxy)
const SOCKET_URL = process.env.NEXT_PUBLIC_SOCKET_URL || '';

interface UseAgentSocketOptions {
    sessionId: string | null;
    onMessage?: (message: WSServerMessage) => void;
}

export function useAgentSocket({ sessionId, onMessage }: UseAgentSocketOptions) {
    const socketRef = useRef<Socket | null>(null);

    const {
        setConnected,
        setConnecting,
        addMessage,
        addStep,
        updateStep,
        updateSessionStatus,
        setCurrentStepId,
        setCurrentScreenshot,
        setCurrentUrl,
        setLiveViewUrl,
        setStatusMessage,
        setLastClickCoords,
        setBrowserClosedMessage,
        setInactivityWarning,
    } = useSessionStore();

    const [error, setError] = useState<string | null>(null);

    const connect = useCallback(() => {
        if (!sessionId) return;

        setConnecting(true);
        setError(null);

        // Create Socket.IO connection
        const socket = io(SOCKET_URL, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 30000,
        });

        socketRef.current = socket;

        socket.on('connect', () => {
            console.log('Socket.IO connected');
            setConnected(true);
            setConnecting(false);

            // Join the session room
            socket.emit('join_session', { session_id: sessionId });
        });

        socket.on('disconnect', () => {
            console.log('Socket.IO disconnected');
            setConnected(false);
            setConnecting(false);
        });

        socket.on('connect_error', (err) => {
            console.error('Socket.IO connection error:', err);
            setError('Connection error');
            setConnecting(false);
        });

        // Handle joined confirmation
        socket.on('joined', (data: { session_id: string; status: string }) => {
            console.log('Joined session:', data.session_id);
        });

        // Register all event handlers
        socket.on('status', (data: { status: string; message: string }) => {
            const status = useSessionStore.getState().session?.status;
            if (status !== 'paused' && status !== 'stopped') {
                setStatusMessage(data.message);
            }
            onMessage?.({ type: 'status', ...data });
        });

        socket.on(
            'browser_ready',
            (data: { url?: string; screenshot_base64?: string; live_view_url?: string }) => {
                setStatusMessage(null);
                // Clear any browser closed message when browser is ready
                setBrowserClosedMessage(null);
                if (data.screenshot_base64) {
                    setCurrentScreenshot(data.screenshot_base64);
                }
                if (data.url) {
                    setCurrentUrl(data.url);
                }
                if (data.live_view_url) {
                    setLiveViewUrl(data.live_view_url + '&navbar=false');
                }
                onMessage?.({ type: 'browser_ready', ...data });
            }
        );

        socket.on(
            'agent_message',
            (data: { branch: string; text: string; is_final?: boolean; is_answer?: boolean }) => {
                addMessage({
                    id: `agent-${Date.now()}`,
                    type: 'agent',
                    content: data.text,
                    timestamp: new Date().toISOString(),
                    isFinal: data.is_final || false,
                    isAnswer: data.is_answer || false,
                });
                onMessage?.({ type: 'agent_message', ...data });
            }
        );

        socket.on(
            'step_started',
            (data: {
                branch: string;
                step_id: number;
                thought: string;
                action_preview: string;
            }) => {
                const status = useSessionStore.getState().session?.status;
                if (status !== 'paused' && status !== 'stopped') {
                    setStatusMessage('Predicting the next move...');
                }
                setLastClickCoords(null);
                onMessage?.({ type: 'step_started', ...data });
            }
        );

        socket.on(
            'action_preview',
            (data: {
                branch: string;
                step_id: number;
                click_coords?: { x: number; y: number };
            }) => {
                const status = useSessionStore.getState().session?.status;
                if (status === 'paused' || status === 'stopped') return;
                if (data.click_coords) {
                    setLastClickCoords(data.click_coords);
                }
                onMessage?.({ type: 'action_preview', ...data });
            }
        );

        // creates step card with action details
        socket.on(
            'step_update',
            (data: {
                branch: string;
                step_id: number;
                thought?: string;
                action_str?: string;
                action_name?: string;
                action_description?: string;
            }) => {
                const status = useSessionStore.getState().session?.status;
                if (status === 'paused' || status === 'stopped') return;
                setStatusMessage(null);
                const newStep: Step = {
                    id: data.step_id,
                    status: 'running',
                    thought: data.thought,
                    actionStr: data.action_str,
                    actionName: data.action_name,
                    actionDescription: data.action_description,
                    createdAt: new Date().toISOString(),
                };
                addStep(newStep);
                onMessage?.({ type: 'step_update', ...data });
            }
        );

        socket.on(
            'step_completed',
            (data: {
                branch: string;
                step_id: number;
                success: boolean;
                thought?: string;
                action_str?: string;
                action_name?: string;
                action_description?: string;
                url?: string;
                screenshot_base64?: string;
                annotated_screenshot?: string;
                click_coords?: { x: number; y: number };
                live_view_url?: string;
            }) => {
                // Always update step status (terminal event), but skip
                // side effects (screenshot, status message) when paused/stopped.
                updateStep(data.step_id, {
                    status: 'completed',
                    thought: data.thought,
                    actionStr: data.action_str,
                    actionName: data.action_name,
                    actionDescription: data.action_description,
                    screenshotBase64: data.screenshot_base64,
                    annotatedScreenshot: data.annotated_screenshot,
                    thumbnailBase64: data.screenshot_base64,
                    url: data.url,
                    clickCoords: data.click_coords,
                });
                setCurrentStepId(null);

                const status = useSessionStore.getState().session?.status;
                if (status !== 'paused' && status !== 'stopped') {
                    setStatusMessage('Predicting the next move...');
                    if (data.screenshot_base64) {
                        setCurrentScreenshot(data.screenshot_base64);
                    }
                    if (data.url) {
                        setCurrentUrl(data.url);
                    }
                    if (data.live_view_url) {
                        setLiveViewUrl(data.live_view_url + '&navbar=false');
                    }
                }
                onMessage?.({ type: 'step_completed', ...data });
            }
        );

        socket.on('step_failed', (data: { branch: string; step_id: number; error: string }) => {
            updateStep(data.step_id, {
                status: 'failed',
                error: data.error,
            });
            setCurrentStepId(null);
            const status = useSessionStore.getState().session?.status;
            if (status !== 'paused' && status !== 'stopped') {
                setStatusMessage(null);
            }
            onMessage?.({ type: 'step_failed', ...data });
        });

        socket.on(
            'step_cancelled',
            (data: { branch?: string; step_id: number; message?: string }) => {
                updateStep(data.step_id, { status: 'cancelled' });
                setCurrentStepId(null);
                onMessage?.({ type: 'step_cancelled', ...data });
            }
        );

        socket.on('session_paused', () => {
            setStatusMessage('User is taking control');
            updateSessionStatus('paused');
            onMessage?.({ type: 'session_paused' });
        });

        socket.on('session_resumed', () => {
            setStatusMessage(null);
            updateSessionStatus('running');
            onMessage?.({ type: 'session_resumed' });
        });

        socket.on('session_complete', (data: { branch: string; summary: string }) => {
            setStatusMessage(null);
            updateSessionStatus('completed');
            onMessage?.({ type: 'session_complete', ...data });
        });

        socket.on('session_stopped', (data: { reason: 'user' | 'error' | 'max_steps' }) => {
            setStatusMessage(null);
            updateSessionStatus('stopped');
            setLiveViewUrl(null);
            setCurrentScreenshot(null);
            setCurrentUrl('');
            addMessage({
                id: `system-${Date.now()}`,
                type: 'system',
                content: `Session stopped: ${data.reason}`,
                timestamp: new Date().toISOString(),
            });
            onMessage?.({ type: 'session_stopped', ...data });
        });

        socket.on('agent_stopped', (data: { reason: string }) => {
            setStatusMessage('Agent stopped. Send a new message to continue.');
            updateSessionStatus('stopped');
            onMessage?.({ type: 'agent_stopped', ...data });
        });

        socket.on('error', (data: { message: string; details?: string }) => {
            setStatusMessage(null);
            setError(data.message);
            addMessage({
                id: `error-${Date.now()}`,
                type: 'system',
                content: `Error: ${data.message}`,
                timestamp: new Date().toISOString(),
            });
            onMessage?.({ type: 'error', ...data });
        });

        socket.on('browser_closed', (data: { message: string }) => {
            setBrowserClosedMessage(data.message);
            setInactivityWarning(null);
            setCurrentScreenshot(null);
            addMessage({
                id: `system-${Date.now()}`,
                type: 'system',
                content: data.message,
                timestamp: new Date().toISOString(),
            });
        });

        socket.on('inactivity_warning', (data: { countdown: number }) => {
            setInactivityWarning(data.countdown);
        });

        socket.on('inactivity_warning_cancelled', () => {
            setInactivityWarning(null);
        });
    }, [
        sessionId,
        setConnected,
        setConnecting,
        onMessage,
        addMessage,
        addStep,
        updateStep,
        updateSessionStatus,
        setCurrentStepId,
        setCurrentScreenshot,
        setCurrentUrl,
        setLiveViewUrl,
        setStatusMessage,
        setLastClickCoords,
        setBrowserClosedMessage,
        setInactivityWarning,
    ]);

    const disconnect = useCallback(() => {
        if (socketRef.current) {
            socketRef.current.disconnect();
            socketRef.current = null;
        }
    }, []);

    // Connect when sessionId changes
    useEffect(() => {
        if (sessionId) {
            connect();
        }

        return () => {
            disconnect();
        };
    }, [sessionId, connect, disconnect]);

    // Send message and optimistically add to chat
    const sendMessage = useCallback(
        (text: string) => {
            if (!sessionId) return;

            // Add user message to chat immediately
            addMessage({
                id: `user-${Date.now()}`,
                type: 'user',
                content: text,
                timestamp: new Date().toISOString(),
            });
            // Update status to running (for follow-up messages)
            updateSessionStatus('running');
            // Send to server
            socketRef.current?.emit('user_message', { session_id: sessionId, text });
        },
        [sessionId, addMessage, updateSessionStatus]
    );

    const pause = useCallback(() => {
        if (!sessionId) return;
        socketRef.current?.emit('pause', { session_id: sessionId });
    }, [sessionId]);

    const resume = useCallback(() => {
        if (!sessionId) return;
        socketRef.current?.emit('resume', { session_id: sessionId });
    }, [sessionId]);

    const navigateToBlank = useCallback(() => {
        if (!sessionId) return;
        socketRef.current?.emit('navigate_to_blank', { session_id: sessionId });
    }, [sessionId]);

    const stopAgent = useCallback(() => {
        if (!sessionId) return;
        updateSessionStatus('stopped');
        setStatusMessage('The agent is gracefully stopping.');
        socketRef.current?.emit('stop_agent', { session_id: sessionId });
    }, [sessionId, updateSessionStatus, setStatusMessage]);

    const extendSession = useCallback(() => {
        if (!sessionId) return;
        socketRef.current?.emit('extend_session', { session_id: sessionId });
    }, [sessionId]);

    /** Leave a session (stops agent and closes browser for that session). Call before switching conversation. */
    const leaveSession = useCallback((id: string) => {
        if (id && socketRef.current) {
            socketRef.current.emit('leave_session', { session_id: id });
        }
    }, []);

    return {
        disconnect,
        error,
        sendMessage,
        pause,
        resume,
        stopAgent,
        navigateToBlank,
        leaveSession,
        extendSession,
    };
}
