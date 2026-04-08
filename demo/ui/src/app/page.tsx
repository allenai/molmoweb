'use client';

import { useCallback, useEffect, useState } from 'react';
import { Box, Alert, Snackbar, Stack } from '@mui/material';

import { useSessionStore } from '@/stores/sessionStore';
import { useAgentSocket } from '@/hooks/useAgentSocket';
import { BrowserView } from './components/BrowserView';
import { ChatPanel } from './components/ChatPanel';
import { ConversationDrawer } from './components/ConversationDrawer';
import { FAQView } from './components/FAQView';
import Header from './components/Header';
import { TermsModal } from './components/TermsModal';
import { fullSessionToSession } from './utils/sessionUtils';
import type { Session, SessionListItem } from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const CACHE_KEY = 'conversations_cache';

function getCachedConversations(): SessionListItem[] {
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        return cached ? JSON.parse(cached) : [];
    } catch {
        return [];
    }
}

function setCachedConversations(sessions: SessionListItem[]) {
    try {
        localStorage.setItem(CACHE_KEY, JSON.stringify(sessions));
    } catch {
        /* ignore quota errors */
    }
}

export default function Home() {
    const { session, setSession, addMessage, reset } = useSessionStore();
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [conversations, setConversations] = useState<SessionListItem[]>(() =>
        getCachedConversations()
    );
    const [conversationsLoading, setConversationsLoading] = useState(
        () => getCachedConversations().length === 0
    );
    const [showFaq, setShowFaq] = useState(false);

    const { sendMessage, pause, resume, stopAgent, leaveSession, extendSession, error } =
        useAgentSocket({
            sessionId,
        });

    const fetchConversations = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/api/sessions`);
            if (!response.ok) return;
            const data = await response.json();
            const sessions = data.sessions || [];
            setConversations(sessions);
            setCachedConversations(sessions);
        } catch (err) {
            console.error('Failed to fetch conversations:', err);
        } finally {
            setConversationsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchConversations();
    }, [fetchConversations]);

    const createSession = useCallback(
        async (goal: string) => {
            try {
                const response = await fetch(`${API_URL}/api/sessions`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ goal, start_url: 'about:blank' }),
                });

                if (!response.ok) {
                    const errBody = await response.json().catch(() => null);
                    const detail = errBody?.detail || 'Failed to create session';
                    addMessage({
                        id: `system-${Date.now()}`,
                        type: 'system',
                        content: detail,
                        timestamp: new Date().toISOString(),
                    });
                    return;
                }

                const data = await response.json();

                const messages: Session['messages'] = [
                    {
                        id: 'initial-user',
                        type: 'user',
                        content: goal,
                        timestamp: new Date().toISOString(),
                    },
                ];

                const newSession: Session = {
                    id: data.id,
                    goal: data.goal,
                    startUrl: data.start_url,
                    status: 'running',
                    messages,
                    createdAt: data.created_at,
                    updatedAt: data.updated_at,
                };

                setSession(newSession);
                setSessionId(data.id);
                setConversations((prev) => {
                    const updated = [data, ...prev];
                    setCachedConversations(updated);
                    return updated;
                });
            } catch (err) {
                console.error('Failed to create session:', err);
            }
        },
        [setSession, addMessage]
    );

    const handleNewChat = useCallback(() => {
        if (sessionId) {
            leaveSession(sessionId);
        }
        reset();
        setSessionId(null);
    }, [sessionId, leaveSession, reset]);

    const handleSelectSession = useCallback(
        async (id: string) => {
            if (id === sessionId) return;
            if (sessionId) {
                leaveSession(sessionId);
            }
            try {
                const response = await fetch(`${API_URL}/api/sessions/${id}/full`);
                if (!response.ok) throw new Error('Failed to load session');
                const data = await response.json();
                const loaded = fullSessionToSession(data);
                setSession(loaded);
                setSessionId(id);
            } catch (err) {
                console.error('Failed to load session:', err);
            }
        },
        [sessionId, leaveSession, setSession]
    );

    const handleDeleteSession = useCallback(
        async (id: string) => {
            try {
                const response = await fetch(`${API_URL}/api/sessions/${id}`, {
                    method: 'DELETE',
                });
                if (!response.ok) return;
                setConversations((prev) => {
                    const updated = prev.filter((s) => s.id !== id);
                    setCachedConversations(updated);
                    return updated;
                });
                if (sessionId === id) {
                    handleNewChat();
                }
            } catch (err) {
                console.error('Failed to delete session:', err);
            }
        },
        [sessionId, handleNewChat]
    );

    const handleDeleteAllSessions = useCallback(async () => {
        const previous = conversations;
        setConversations([]);
        setCachedConversations([]);
        if (sessionId) {
            handleNewChat();
        }
        try {
            const response = await fetch(`${API_URL}/api/sessions`, {
                method: 'DELETE',
            });
            if (!response.ok) {
                setConversations(previous);
                setCachedConversations(previous);
            }
        } catch (err) {
            console.error('Failed to delete all sessions:', err);
            setConversations(previous);
            setCachedConversations(previous);
        }
    }, [conversations, sessionId, handleNewChat]);

    const handleSendMessage = useCallback(
        (text: string) => {
            if (!sessionId) {
                createSession(text);
            } else {
                sendMessage(text);
            }
        },
        [sessionId, createSession, sendMessage]
    );

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0, overflow: 'hidden' }}>
            <Header onFaqToggle={() => setShowFaq((v) => !v)} />
            <Stack direction="row" flex={1} overflow="hidden" spacing={0} minHeight={0}>
                <ConversationDrawer
                    sessions={conversations}
                    currentSessionId={sessionId}
                    onNewChat={handleNewChat}
                    onSelectSession={handleSelectSession}
                    onDeleteSession={handleDeleteSession}
                    onDeleteAllSessions={handleDeleteAllSessions}
                    loading={conversationsLoading}
                />

                {showFaq ? (
                    <FAQView onBack={() => setShowFaq(false)} />
                ) : (
                    <>
                        <Stack flex={3} px={1} py={3} minWidth={0} minHeight={0}>
                            <BrowserView onTakeControl={pause} onResumeAgent={resume} />
                            <Box
                                sx={{
                                    fontSize: 12,
                                    textAlign: 'center',
                                    maxWidth: '1000px',
                                    alignSelf: 'center',
                                    color: 'grey.500',
                                    '& a': { color: 'grey.500' },
                                }}>
                                Molmo Web is a free scientific and educational tool and by using it
                                you agree to Ai2&apos;s{' '}
                                <a href="https://allenai.org/terms">Terms of use</a>, and{' '}
                                <a href="https://allenai.org/responsible-use">
                                    Responsible Use Guidelines
                                </a>
                                , and have read Ai2&apos;s{' '}
                                <a href="https://allenai.org/privacy-policy">Privacy policy</a>.
                                This site is protected by reCAPTCHA and the Google{' '}
                                <a href="https://policies.google.com/privacy">Privacy Policy</a>{' '}
                                and{' '}
                                <a href="https://policies.google.com/terms">Terms of Service</a>{' '}
                                apply.
                            </Box>
                        </Stack>
                        <Stack
                            sx={{
                                flex: 1,
                                minWidth: 260,
                                height: '100%',
                                minHeight: 0,
                                overflow: 'hidden',
                            }}>
                            <ChatPanel
                                onSendMessage={handleSendMessage}
                                onStopAgent={stopAgent}
                                onExtendSession={extendSession}
                            />
                        </Stack>
                    </>
                )}
            </Stack>

            <Snackbar
                open={!!error}
                autoHideDuration={6000}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
                <Alert severity="error">{error}</Alert>
            </Snackbar>

            <TermsModal />
        </div>
    );
}
