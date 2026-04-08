'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import {
    Box,
    Typography,
    Chip,
    CircularProgress,
    Fab,
    Button,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    ChatBubbleOutline as MessageIcon,
    Language as GlobeIcon,
    Pause as PauseIcon,
    KeyboardArrowDown as ArrowDownIcon,
    WarningAmber as WarningIcon,
    Share as ShareIcon,
    Check as CheckIcon,
} from '@mui/icons-material';

import { useSessionStore } from '@/stores/sessionStore';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';

interface ChatPanelProps {
    onSendMessage: (text: string) => void;
    onStopAgent?: () => void;
    onExtendSession?: () => void;
}

const SUGGESTIONS = [
    "On semanticscholar.org, search for Yann Lecun, open the author profile, click on 'Co-Authors', and tell me the top 3 names.",
    'On bing maps, find coffee shops near Pike Market, Seattle and tell me the names and ratings of the top 3 results.',
    "On barnesandnoble.com, search for 'Project Hail Mary', open the product, and add to cart.",
    "On apartments.com, search for 'condo in seattle', filter for 2 rooms and tell me price of the first 2 results.",
    'Go to github.com/trending, set language to Python and tell me the name, description, and stars of the top 2 repos.',
    'On nationalgeographic.com, go to the Science & Nature section, find the first article about Climate Change, and tell me about its title.',
    'Find SatlasPretrain: A Large-Scale Dataset for Remote Sensing Image Understanding on arxiv and tell me the names of the authors.',
    'Go to foodnetwork.com, open Recipes, then open Lunch recipes, find Top lunch recipes section and tell me the names of the top 2.',
] as const;

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const SCROLL_THRESHOLD = 80;

export function ChatPanel({ onSendMessage, onStopAgent, onExtendSession }: ChatPanelProps) {
    const { session, setSession, statusMessage, inactivityWarning } = useSessionStore();
    const scrollRef = useRef<HTMLDivElement>(null);
    const isNearBottom = useRef(true);
    const [showScrollBtn, setShowScrollBtn] = useState(false);
    const [prefillForInput, setPrefillForInput] = useState('');
    const [countdown, setCountdown] = useState<number | null>(null);
    const [shareCopied, setShareCopied] = useState(false);

    useEffect(() => {
        if (inactivityWarning == null) {
            setCountdown(null);
            return;
        }
        setCountdown(inactivityWarning);
        const id = setInterval(() => {
            setCountdown((prev) => (prev != null && prev > 0 ? prev - 1 : prev));
        }, 1000);
        return () => clearInterval(id);
    }, [inactivityWarning]);

    const copyToClipboard = useCallback(async (text: string): Promise<boolean> => {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch {
            // Fallback for non-HTTPS / unfocused contexts
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            try {
                document.execCommand('copy');
                return true;
            } catch {
                return false;
            } finally {
                document.body.removeChild(textarea);
            }
        }
    }, []);

    const handleShare = useCallback(async () => {
        if (!session) return;
        try {
            const res = await fetch(`${API_URL}/api/sessions/${session.id}/share`, {
                method: 'POST',
            });
            if (!res.ok) {
                console.error('Share endpoint returned', res.status);
                return;
            }
            const data = await res.json();
            const shareUrl = `${window.location.origin}/shared/${data.share_token}`;
            const copied = await copyToClipboard(shareUrl);
            if (copied) {
                setShareCopied(true);
                setTimeout(() => setShareCopied(false), 2000);
            }
            if (!session.shareToken) {
                setSession({ ...session, shareToken: data.share_token });
            }
        } catch (err) {
            console.error('Failed to share session:', err);
        }
    }, [session, setSession, copyToClipboard]);

    const handleSuggestionClick = useCallback((text: string) => {
        setPrefillForInput(text);
    }, []);

    const clearPrefill = useCallback(() => setPrefillForInput(''), []);

    const handleScroll = useCallback(() => {
        const el = scrollRef.current;
        if (!el) return;
        const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_THRESHOLD;
        isNearBottom.current = nearBottom;
        if (nearBottom) setShowScrollBtn(false);
    }, []);

    const scrollToBottom = useCallback(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTo({
                top: scrollRef.current.scrollHeight,
                behavior: 'smooth',
            });
        }
        isNearBottom.current = true;
        setShowScrollBtn(false);
    }, []);

    useEffect(() => {
        if (scrollRef.current && isNearBottom.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        } else if (!isNearBottom.current) {
            setShowScrollBtn(true);
        }
    }, [session?.messages.length, statusMessage]);

    const stepCount = session?.messages.filter((m) => m.type === 'step').length || 0;

    return (
        <Box
            sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                bgcolor: 'background.paper',
                margin: 0,
                borderLeft: 3,
                borderColor: 'grey.100',
            }}>
            {/* Header - only show when there are messages */}
            {session?.messages && session.messages.length > 0 && (
                <Box
                    sx={{
                        height: 48,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        px: 2,
                        borderBottom: 1,
                        borderColor: 'grey.200',
                    }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <MessageIcon sx={{ fontSize: 18, color: 'grey.600' }} />
                        <Typography variant="body2" fontWeight={500} color="text.secondary">
                            Agent Activity
                        </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        {stepCount > 0 && (
                            <Chip
                                label={`${stepCount} ${stepCount === 1 ? 'step' : 'steps'}`}
                                size="small"
                                sx={{
                                    height: 20,
                                    fontSize: '0.7rem',
                                    bgcolor: 'grey.100',
                                    color: 'grey.600',
                                }}
                            />
                        )}
                        <Tooltip title={shareCopied ? 'Link copied!' : 'Copy share link'}>
                            <IconButton
                                size="small"
                                onClick={handleShare}
                                sx={{ color: shareCopied ? 'success.main' : 'grey.500' }}>
                                {shareCopied ? (
                                    <CheckIcon sx={{ fontSize: 16 }} />
                                ) : (
                                    <ShareIcon sx={{ fontSize: 16 }} />
                                )}
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>
            )}

            {/* Messages */}
            <Box sx={{ flex: 1, position: 'relative', minHeight: 0 }}>
                <Box
                    ref={scrollRef}
                    onScroll={handleScroll}
                    sx={{
                        height: '100%',
                        overflowY: 'auto',
                        scrollbarGutter: 'stable',
                    }}>
                    {!session?.messages || session.messages.length === 0 ? (
                        <Box
                            sx={{
                                px: 3,
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'flex-start',
                                pt: 6,
                                gap: 2,
                            }}>
                            <img src="/ai2-logo.svg" alt="AI2 Logo" style={{ height: 53 }} />
                            <Box sx={{ textAlign: 'center' }}>
                                <Typography variant="body2" color="text.primary" sx={{ mb: 0.5 }}>
                                    Give MolmoWeb a task to begin browsing
                                </Typography>
                            </Box>
                            <Box
                                sx={{
                                    width: '100%',
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                                    gap: 1,
                                }}>
                                {SUGGESTIONS.map((label) => (
                                    <Box
                                        key={label}
                                        component="button"
                                        type="button"
                                        onClick={() => handleSuggestionClick(label)}
                                        sx={{
                                            border: 'none',
                                            cursor: 'pointer',
                                            textAlign: 'left',
                                            font: 'inherit',
                                            borderRadius: 3,
                                            bgcolor: 'grey.100',
                                            px: 2,
                                            py: 2,
                                            fontSize: '0.8rem',
                                            color: 'text.secondary',
                                            '&:hover': { bgcolor: 'grey.200' },
                                        }}>
                                        {label}
                                    </Box>
                                ))}
                            </Box>
                        </Box>
                    ) : (
                        <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            {session.messages.map((message) => (
                                <ChatMessage key={message.id} message={message} />
                            ))}

                            {statusMessage && (
                                <Box
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 0.75,
                                        ml: 0.5,
                                        py: 0.5,
                                    }}>
                                    {session?.status === 'running' ? (
                                        <CircularProgress size={12} thickness={5} />
                                    ) : session?.status === 'paused' ? (
                                        <PauseIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
                                    ) : null}
                                    <Typography
                                        variant="caption"
                                        color="text.disabled"
                                        sx={{ fontSize: '0.7rem' }}>
                                        {statusMessage}
                                    </Typography>
                                </Box>
                            )}
                        </Box>
                    )}
                </Box>

                {/* Scroll-to-bottom button */}
                {showScrollBtn && (
                    <Fab
                        size="small"
                        onClick={scrollToBottom}
                        sx={{
                            position: 'absolute',
                            bottom: 12,
                            right: 20,
                            width: 32,
                            height: 32,
                            minHeight: 32,
                            bgcolor: 'background.paper',
                            color: 'text.secondary',
                            boxShadow: 2,
                            '&:hover': { bgcolor: 'grey.100' },
                        }}>
                        <ArrowDownIcon sx={{ fontSize: 20 }} />
                    </Fab>
                )}
            </Box>

            {/* Inactivity warning */}
            {countdown != null && (
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        px: 2,
                        py: 1,
                        bgcolor: (t) => `${t.palette.primary.main}0A`,
                        borderTop: 1,
                        borderColor: 'primary.light',
                    }}>
                    <WarningIcon sx={{ fontSize: 18, color: 'primary.main' }} />
                    <Typography variant="caption" sx={{ flex: 1, color: 'primary.dark' }}>
                        Browser closing in {countdown}s due to inactivity
                    </Typography>
                    <Button
                        size="small"
                        variant="outlined"
                        color="primary"
                        onClick={onExtendSession}
                        sx={{
                            textTransform: 'none',
                            fontSize: '0.7rem',
                            py: 0.25,
                            minWidth: 0,
                        }}>
                        Continue
                    </Button>
                </Box>
            )}

            {/* Input */}
            <ChatInput
                status={session?.status || 'idle'}
                onSend={onSendMessage}
                onStopAgent={onStopAgent}
                prefill={prefillForInput}
                onPrefillApplied={clearPrefill}
            />
        </Box>
    );
}
