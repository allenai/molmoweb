'use client';

import { useState, useEffect, KeyboardEvent } from 'react';
import { Box, TextField, IconButton, Typography, InputAdornment } from '@mui/material';
import { Send as SendIcon, Stop as StopIcon } from '@mui/icons-material';

import type { SessionStatus } from '@/types';

interface ChatInputProps {
    status: SessionStatus;
    onSend: (text: string) => void;
    onStopAgent?: () => void;
    prefill?: string;
    onPrefillApplied?: () => void;
}

export function ChatInput({
    status,
    onSend,
    onStopAgent,
    prefill,
    onPrefillApplied,
}: ChatInputProps) {
    const [input, setInput] = useState('');
    const [ready, setReady] = useState(false);

    useEffect(() => {
        setReady(true);
    }, []);

    useEffect(() => {
        if (prefill) {
            setInput(prefill);
            onPrefillApplied?.();
        }
    }, [prefill, onPrefillApplied]);

    const isRunning = status === 'running';
    const isIdle = status === 'idle';
    const canSend = input.trim().length > 0 && !isRunning;

    const handleSubmit = () => {
        if (canSend) {
            onSend(input.trim());
            setInput('');
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <Box
            sx={{
                px: 0,
                pb: 0,
                m: 1,
                bgcolor: 'transparent',
                opacity: ready ? 1 : 0,
            }}>
            <TextField
                fullWidth
                size="small"
                placeholder={
                    isIdle
                        ? 'Describe a task'
                        : isRunning
                          ? 'Agent is working...'
                          : status === 'stopped'
                            ? 'Send a new message to continue'
                            : 'Send a follow-up message...'
                }
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isRunning}
                multiline
                minRows={1}
                maxRows={6}
                InputProps={{
                    endAdornment: (
                        <InputAdornment position="end">
                            {isRunning ? (
                                <IconButton
                                    onClick={() => onStopAgent?.()}
                                    edge="end"
                                    aria-label="Stop agent"
                                    sx={{
                                        color: '#d6559c',
                                    }}>
                                    <StopIcon />
                                </IconButton>
                            ) : (
                                <IconButton
                                    onClick={handleSubmit}
                                    disabled={!canSend}
                                    edge="end"
                                    aria-label="Send"
                                    sx={{
                                        color: '#d6559c',
                                        '&.Mui-disabled': {
                                            color: '#d6559c',
                                            opacity: 0.5,
                                        },
                                    }}>
                                    <SendIcon />
                                </IconButton>
                            )}
                        </InputAdornment>
                    ),
                }}
                sx={{
                    '& .MuiOutlinedInput-root': {
                        bgcolor: 'white',
                        borderRadius: 2.5,
                        py: 1,
                        pl: 2.5,
                        pr: 1.5,
                    },
                    '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'grey.200',
                    },
                    '& .MuiInputAdornment-root': {
                        alignSelf: 'center',
                    },
                }}
            />
            <Typography
                variant="caption"
                color="text.disabled"
                sx={{ display: 'block', textAlign: 'center', mt: 1, fontSize: '0.65rem' }}>
                Always fact-check your results.
            </Typography>
        </Box>
    );
}
