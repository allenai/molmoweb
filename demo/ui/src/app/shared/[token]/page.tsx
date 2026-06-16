'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { Box, Typography, CircularProgress, Alert } from '@mui/material';

import { ChatMessage } from '../../components/ChatMessage';
import { buildMessagesFromFullSession } from '../../utils/sessionUtils';
import type { ChatMessage as ChatMessageType } from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

export default function SharedTrajectoryPage() {
    const { token } = useParams<{ token: string }>();
    const [messages, setMessages] = useState<ChatMessageType[]>([]);
    const [goal, setGoal] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!token) return;
        (async () => {
            try {
                const res = await fetch(`${API_URL}/api/shared/${token}`);
                if (!res.ok) {
                    setError(res.status === 404 ? 'This shared trajectory was not found or has been removed.' : 'Failed to load trajectory.');
                    return;
                }
                const data = await res.json();
                setGoal(data.goal || '');
                setMessages(buildMessagesFromFullSession({ messages: data.messages, steps: data.steps }));
            } catch {
                setError('Failed to load trajectory.');
            } finally {
                setLoading(false);
            }
        })();
    }, [token]);

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box sx={{ maxWidth: 600, mx: 'auto', mt: 8, px: 2 }}>
                <Alert severity="warning">{error}</Alert>
            </Box>
        );
    }

    return (
        <Box sx={{ maxWidth: 700, mx: 'auto', py: 4, px: 2 }}>
            <Typography variant="h6" sx={{ mb: 0.5 }}>
                Shared Trajectory
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                {goal}
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                {messages.map((message) => (
                    <ChatMessage key={message.id} message={message} />
                ))}
            </Box>
        </Box>
    );
}
