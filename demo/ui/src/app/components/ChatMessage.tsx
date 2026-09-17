'use client';

import { Box, Typography, Avatar, Link } from '@mui/material';
import { Person as PersonIcon, Info as InfoIcon } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';

import type { ChatMessage as ChatMessageType } from '@/types';
import { StepCard } from './StepCard';

interface ChatMessageProps {
    message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
    const isUser = message.type === 'user';
    // const isAgent = message.type === 'agent';
    const isSystem = message.type === 'system';
    const isStep = message.type === 'step';

    // Step messages render as StepCard
    if (isStep && message.step) {
        return <StepCard step={message.step} />;
    }

    // System messages
    if (isSystem) {
        return (
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    my: 1,
                }}>
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 0.5,
                        px: 1.5,
                        py: 0.5,
                        bgcolor: 'grey.100',
                        borderRadius: 2,
                    }}>
                    <InfoIcon sx={{ fontSize: 14, color: 'grey.600' }} />
                    <Typography variant="caption" color="text.secondary">
                        {message.content}
                    </Typography>
                </Box>
            </Box>
        );
    }

    // User and agent messages
    return (
        <Box
            sx={{
                display: 'flex',
                gap: 1.5,
                flexDirection: isUser ? 'row-reverse' : 'row',
            }}>
            {/* Avatar */}
            <Avatar
                src={isUser ? undefined : '/agent-icon.png'}
                sx={{
                    width: 32,
                    height: 32,
                    bgcolor: isUser ? 'primary.main' : 'transparent',
                }}>
                {isUser && <PersonIcon sx={{ fontSize: 18 }} />}
            </Avatar>

            {/* Message bubble */}
            <Box
                sx={{
                    maxWidth: '80%',
                    px: 2,
                    py: 1.5,
                    borderRadius: 2,
                    bgcolor: isUser ? 'primary.main' : 'grey.100',
                    color: isUser ? 'white' : 'text.primary',
                    ...(message.isFinal && {
                        bgcolor: '#FCE4EC',
                        border: 1,
                        borderColor: '#F48FB1',
                    }),
                    ...(message.isAnswer && {
                        bgcolor: '#E8F5E9',
                        border: 1,
                        borderColor: '#81C784',
                    }),
                }}>
                <Box
                    sx={{
                        wordBreak: 'break-word',
                        '& p': { m: 0 },
                        '& p + p': { mt: 1 },
                        '& ul, & ol': { m: 0, pl: 2.5 },
                        '& pre': {
                            bgcolor: 'rgba(0,0,0,0.06)',
                            p: 1,
                            borderRadius: 1,
                            overflowX: 'auto',
                            fontSize: '0.8rem',
                        },
                        '& code': {
                            bgcolor: 'rgba(0,0,0,0.06)',
                            px: 0.5,
                            borderRadius: 0.5,
                            fontSize: '0.85em',
                            fontFamily: 'monospace',
                        },
                        '& pre code': {
                            bgcolor: 'transparent',
                            px: 0,
                        },
                        '& blockquote': {
                            m: 0,
                            pl: 1.5,
                            borderLeft: '3px solid',
                            borderColor: 'grey.400',
                            color: 'text.secondary',
                        },
                    }}>
                    <ReactMarkdown
                        components={{
                            p: ({ children }) => (
                                <Typography variant="body2" component="p">
                                    {children}
                                </Typography>
                            ),
                            li: ({ children }) => (
                                <Typography variant="body2" component="li">
                                    {children}
                                </Typography>
                            ),
                            a: ({ href, children }) => (
                                <Link href={href} target="_blank" rel="noopener noreferrer">
                                    {children}
                                </Link>
                            ),
                        }}>
                        {message.content}
                    </ReactMarkdown>
                </Box>
            </Box>
        </Box>
    );
}
