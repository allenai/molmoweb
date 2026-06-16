'use client';

import { useState } from 'react';
import {
    Box,
    Card,
    CardContent,
    Collapse,
    Typography,
    Chip,
    CircularProgress,
} from '@mui/material';
import {
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    CheckCircle as CheckIcon,
    Error as ErrorIcon,
    Pause as PauseIcon,
} from '@mui/icons-material';

import type { Step } from '@/types';

interface StepCardProps {
    step: Step;
}

export function StepCard({ step }: StepCardProps) {
    const [expanded, setExpanded] = useState(false);

    const isRunning = step.status === 'running';
    const isCompleted = step.status === 'completed';
    const isFailed = step.status === 'failed';
    const isCancelled = step.status === 'cancelled';

    const getStatusIcon = () => {
        if (isRunning) {
            return <CircularProgress size={16} />;
        }
        if (isCompleted) {
            return <CheckIcon sx={{ fontSize: 16, color: 'success.main' }} />;
        }
        if (isFailed) {
            return <ErrorIcon sx={{ fontSize: 16, color: 'error.main' }} />;
        }
        if (isCancelled) {
            return <PauseIcon sx={{ fontSize: 16, color: 'grey.600' }} />;
        }
        return null;
    };

    const hasExpandableContent = !!(step.thought || step.screenshotBase64 || step.annotatedScreenshot);

    const getActionLabel = () => {
        if (step.actionName) {
            return step.actionName;
        }
        if (step.actionStr) {
            // Extract action name from action string like "click(x=100, y=200)"
            const match = step.actionStr.match(/^(\w+)\(/);
            return match ? match[1] : 'action';
        }
        return 'processing';
    };

    return (
        <Card
            variant="outlined"
            sx={{
                bgcolor: isRunning ? 'action.hover' : isCancelled ? 'grey.50' : 'background.paper',
                transition: 'all 0.2s ease',
            }}>
            <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
                {/* Header row */}
                <Box
                    onClick={hasExpandableContent ? () => setExpanded(!expanded) : undefined}
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: 1,
                        cursor: hasExpandableContent ? 'pointer' : 'default',
                        userSelect: 'none',
                    }}>
                    <Box
                        sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            flex: 1,
                            minWidth: 0,
                        }}>
                        {getStatusIcon()}
                        <Chip
                            label={getActionLabel()}
                            size="small"
                            sx={{
                                height: 20,
                                fontSize: '0.7rem',
                                bgcolor: isRunning
                                    ? 'primary.100'
                                    : isCompleted
                                      ? 'success.100'
                                      : isCancelled
                                        ? 'grey.200'
                                        : 'error.100',
                                color: isRunning
                                    ? 'primary.dark'
                                    : isCompleted
                                      ? 'success.dark'
                                      : isCancelled
                                        ? 'grey.700'
                                        : 'error.dark',
                            }}
                        />
                        {/* <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{
                                flex: 1,
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap',
                            }}>
                            {step.actionDescription || step.actionStr || 'Executing...'}
                        </Typography> */}
                    </Box>

                    {hasExpandableContent && (
                        <Box sx={{ display: 'flex', alignItems: 'center', p: 0.5 }}>
                            {expanded ? (
                                <ExpandLessIcon sx={{ fontSize: 16, color: 'action.active' }} />
                            ) : (
                                <ExpandMoreIcon sx={{ fontSize: 16, color: 'action.active' }} />
                            )}
                        </Box>
                    )}
                </Box>

                {/* Expandable content */}
                <Collapse in={expanded}>
                    <Box sx={{ mt: 1.5, pt: 1.5, borderTop: 1, borderColor: 'grey.200' }}>
                        {/* Action Description */}
                        {step.actionDescription && (
                            <Box sx={{ mb: 1.5 }}>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ fontWeight: 600 }}>
                                    Action Description
                                </Typography>
                                <Typography variant="body2" sx={{ mt: 0.5, fontSize: '0.8rem' }}>
                                    {step.actionDescription}
                                </Typography>
                            </Box>
                        )}

                        {/* Thought */}
                        {step.thought && (
                            <Box sx={{ mb: 1.5 }}>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ fontWeight: 600 }}>
                                    Thought
                                </Typography>
                                <Typography variant="body2" sx={{ mt: 0.5, fontSize: '0.8rem' }}>
                                    {step.thought}
                                </Typography>
                            </Box>
                        )}

                        {/* Screenshot */}
                        {(step.annotatedScreenshot || step.screenshotBase64) && (
                            <Box>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ fontWeight: 600 }}>
                                    Screenshot
                                </Typography>
                                <Box
                                    component="img"
                                    src={`data:image/png;base64,${step.annotatedScreenshot || step.screenshotBase64}`}
                                    alt={`Step ${step.id} screenshot`}
                                    sx={{
                                        width: '100%',
                                        mt: 0.5,
                                        borderRadius: 1,
                                        border: 1,
                                        borderColor: 'grey.200',
                                    }}
                                />
                            </Box>
                        )}

                        {/* Error */}
                        {step.error && (
                            <Box sx={{ mt: 1.5 }}>
                                <Typography
                                    variant="caption"
                                    sx={{ color: 'error.main', fontWeight: 600 }}>
                                    Error
                                </Typography>
                                <Typography
                                    variant="body2"
                                    sx={{ mt: 0.5, fontSize: '0.8rem', color: 'error.main' }}>
                                    {step.error}
                                </Typography>
                            </Box>
                        )}

                        {/* URL */}
                        {step.url && (
                            <Box sx={{ mb: 1.5 }}>
                                <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ fontWeight: 600 }}>
                                    URL
                                </Typography>
                                <Typography variant="body2" sx={{ mt: 0.5, fontSize: '0.8rem' }}>
                                    {step.url}
                                </Typography>
                            </Box>
                        )}
                    </Box>
                </Collapse>
            </CardContent>
        </Card>
    );
}
