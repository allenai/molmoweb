'use client';

import { useState, useRef, useEffect } from 'react';
import { Box, Card, Typography, Button, Switch, keyframes } from '@mui/material';
import {
    Language as GlobeIcon,
    PanTool as HandIcon,
    PlayArrow as PlayIcon,
    Lock as LockIcon,
} from '@mui/icons-material';

import { useSessionStore } from '@/stores/sessionStore';

// Backend browser viewport dimensions
const VIEWPORT_WIDTH = 1280;
const VIEWPORT_HEIGHT = 720;
// const VIEWPORT_ASPECT = VIEWPORT_WIDTH / VIEWPORT_HEIGHT;

const TOP_BAR_HEIGHT = 44;
const BOTTOM_BAR_HEIGHT = 30;

const SHADOW_INSET_PX = 16;

const pulseRing = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(240, 82, 156, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(240, 82, 156, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(240, 82, 156, 0);
  }
`;

interface BrowserViewProps {
    onTakeControl?: () => void;
    onResumeAgent?: () => void;
}

export function BrowserView({ onTakeControl, onResumeAgent }: BrowserViewProps) {
    const { session, currentScreenshot, liveViewUrl, lastClickCoords, currentUrl } =
        useSessionStore();
    const [showLive] = useState(true);
    const [showIndicator, setShowIndicator] = useState(true);
    const cardRef = useRef<HTMLDivElement>(null);
    const [scale, setScale] = useState(0);

    const isAutomating = session?.status === 'running';
    const isPaused = session?.status === 'paused';
    const isStoppedWithBrowser =
        (session?.status === 'stopped' || session?.status === 'completed') &&
        !!(currentScreenshot || liveViewUrl);
    const hasSession = !!session;

    useEffect(() => {
        const el = cardRef.current;
        if (!el) return;
        const updateScale = () => {
            const w = Math.max(0, el.clientWidth - 2 * SHADOW_INSET_PX);
            const h = Math.max(0, el.clientHeight - 2 * SHADOW_INSET_PX);
            const viewportHeightAvailable = h - (TOP_BAR_HEIGHT + BOTTOM_BAR_HEIGHT);
            const s =
                viewportHeightAvailable > 0
                    ? Math.min(w / VIEWPORT_WIDTH, viewportHeightAvailable / VIEWPORT_HEIGHT)
                    : 0;
            setScale(Math.max(0, s));
        };
        const ro = new ResizeObserver(updateScale);
        ro.observe(el);
        updateScale();
        return () => ro.disconnect();
    }, []);

    const scaledWidth = VIEWPORT_WIDTH * scale;
    const scaledViewportHeight = VIEWPORT_HEIGHT * scale;
    const totalHeight = TOP_BAR_HEIGHT + scaledViewportHeight;

    return (
        <Card
            ref={cardRef}
            sx={{
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                marginInline: 0.5,
                bgcolor: 'transparent',
                border: 'none',
            }}>
            <Box
                sx={{
                    width: scaledWidth,
                    height: totalHeight,
                    flexShrink: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    minWidth: 0,
                    minHeight: 0,
                    borderRadius: '10px',
                    overflow: 'hidden',
                    boxShadow: '0 4px 12px #0000000d',
                    opacity: scale > 0 ? 1 : 0,
                    ...(isAutomating && {
                        boxShadow: '0 0 0 5px #F0529C, 0 2px 16px rgba(0,0,0,0.15)',
                        animation: `${pulseRing} 2s infinite`,
                    }),
                }}>
                {/* Browser chrome */}
                <Box
                    sx={{
                        flex: `0 0 ${TOP_BAR_HEIGHT}px`,
                        height: TOP_BAR_HEIGHT,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        px: 1.5,
                        bgcolor: '#105257',
                        borderBottom: 1,
                        borderColor: 'grey.300',
                    }}>
                    {/* Window dots */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.6, mr: 0.5 }}>
                        <Box
                            sx={{
                                width: 12,
                                height: 12,
                                borderRadius: '50%',
                                bgcolor: '#FFFFFF',
                            }}
                        />
                        <Box
                            sx={{
                                width: 12,
                                height: 12,
                                borderRadius: '50%',
                                bgcolor: '#FFFFFF',
                            }}
                        />
                        <Box
                            sx={{
                                width: 12,
                                height: 12,
                                borderRadius: '50%',
                                bgcolor: '#FFFFFF',
                            }}
                        />
                    </Box>

                    {/* Address bar */}
                    <Box
                        sx={{
                            flex: 1,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 0.75,
                            bgcolor: 'rgba(255,255,255,0.08)',
                            borderRadius: '8px',
                            px: 1.5,
                            py: 0.5,
                            height: 30,
                            minWidth: 0,
                            cursor: 'not-allowed',
                        }}>
                        <LockIcon sx={{ fontSize: 14, color: 'rgba(255,255,255,0.3)', flexShrink: 0 }} />
                        <Typography
                            variant="caption"
                            noWrap
                            sx={{
                                color: 'rgba(255,255,255,0.45)',
                                fontSize: 13,
                                lineHeight: 1,
                                minWidth: 0,
                                cursor: 'not-allowed',
                                userSelect: 'none',
                            }}>
                            {currentUrl || 'about:blank'}
                        </Typography>
                    </Box>

                    {/* Indicator toggle */}
                    <Box
                        sx={{
                            display: 'flex',
                            alignItems: 'center',
                            flexShrink: 0,
                            ml: 0.5,
                        }}>
                        <Typography variant="caption" sx={{ fontSize: 11, color: 'white' }}>
                            Click indicator
                        </Typography>
                        <Switch
                            size="small"
                            checked={showIndicator}
                            onChange={(e) => setShowIndicator(e.target.checked)}
                            sx={{
                                '& .MuiSwitch-switchBase.Mui-checked': { color: '#F0529C' },
                                '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                    bgcolor: 'rgba(240,82,156,0.5)',
                                },
                                '& .MuiSwitch-switchBase': { color: 'rgba(255,255,255,0.4)' },
                                '& .MuiSwitch-track': { bgcolor: 'rgba(255,255,255,0.2)' },
                            }}
                        />
                    </Box>
                </Box>

                {/* Viewport: 1280×720 scaled to fit; height = 720*scale */}
                <Box
                    sx={{
                        flex: `0 0 ${scaledViewportHeight}px`,
                        height: scaledViewportHeight,
                        minHeight: 0,
                        position: 'relative',
                        width: '100%',
                        overflow: 'hidden',
                        bgcolor: 'white',
                    }}>
                    <Box
                        sx={{
                            width: VIEWPORT_WIDTH,
                            height: VIEWPORT_HEIGHT,
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            transform: `scale(${scale})`,
                            transformOrigin: '0 0',
                        }}>
                        {/* Live view iframe */}
                        {liveViewUrl && showLive ? (
                            <>
                                <Box
                                    component="iframe"
                                    src={liveViewUrl}
                                    sx={{
                                        width: '100%',
                                        height: '100%',
                                        border: 'none',
                                        position: 'absolute',
                                        inset: 0,
                                        zIndex: 0,
                                        ...((isAutomating || isStoppedWithBrowser) && {
                                            pointerEvents: 'none',
                                        }),
                                    }}
                                    title="Live browser view"
                                    sandbox="allow-same-origin allow-scripts"
                                />
                                {/* Click blocker overlay when agent is automating */}
                                {(isAutomating || isStoppedWithBrowser) && (
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            inset: 0,
                                            zIndex: 10,
                                            cursor: 'not-allowed',
                                        }}
                                        onClick={(e) => e.preventDefault()}
                                        title="Click 'Take Control' below to interact with the browser"
                                    />
                                )}
                            </>
                        ) : currentScreenshot ? (
                            <Box
                                component="img"
                                src={`data:image/png;base64,${currentScreenshot}`}
                                alt="Browser view"
                                sx={{
                                    width: '100%',
                                    height: '100%',
                                    objectFit: 'contain',
                                }}
                            />
                        ) : (
                            <Box
                                sx={{
                                    position: 'absolute',
                                    inset: 0,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                }}>
                                <Box sx={{ textAlign: 'center' }}>
                                    <Typography variant="body2" color="text.secondary">
                                        {session && session.status === 'running'
                                            ? 'Starting browser session...'
                                            : 'Give MolmoWeb a task to begin browsing'}
                                    </Typography>
                                </Box>
                            </Box>
                        )}

                        {/* Click indicator overlay */}
                        {showIndicator && lastClickCoords && (
                            <Box
                                sx={{
                                    position: 'absolute',
                                    inset: 0,
                                    pointerEvents: 'none',
                                    zIndex: 9999,
                                }}>
                                <Box
                                    sx={{
                                        position: 'absolute',
                                        left: `${(lastClickCoords.x / VIEWPORT_WIDTH) * 100}%`,
                                        top: `${(lastClickCoords.y / VIEWPORT_HEIGHT) * 100}%`,
                                        transform: 'translate(-50%, -50%)',
                                    }}>
                                    {/* Outer ring - pulsing */}
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            width: 48,
                                            height: 48,
                                            left: -24,
                                            top: -24,
                                            borderRadius: '50%',
                                            border: '4px solid',
                                            borderColor: 'error.main',
                                            animation:
                                                'ping 1s cubic-bezier(0, 0, 0.2, 1) infinite',
                                            '@keyframes ping': {
                                                '75%, 100%': {
                                                    transform: 'scale(2)',
                                                    opacity: 0,
                                                },
                                            },
                                        }}
                                    />
                                    {/* Middle ring */}
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            width: 32,
                                            height: 32,
                                            left: -16,
                                            top: -16,
                                            borderRadius: '50%',
                                            border: '2px solid',
                                            borderColor: 'error.main',
                                            bgcolor: 'rgba(239, 68, 68, 0.2)',
                                        }}
                                    />
                                    {/* Inner dot */}
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            width: 12,
                                            height: 12,
                                            left: -6,
                                            top: -6,
                                            borderRadius: '50%',
                                            bgcolor: 'error.main',
                                        }}
                                    />
                                </Box>
                            </Box>
                        )}
                    </Box>
                </Box>
            </Box>

            {/* Take Control / Resume button */}
            <Button
                variant="outlined"
                size="medium"
                startIcon={isPaused ? <PlayIcon /> : <HandIcon />}
                disabled={!hasSession || (!isAutomating && !isPaused) || isStoppedWithBrowser}
                onClick={isPaused ? onResumeAgent : onTakeControl}
                sx={{
                    textTransform: 'none',
                    mt: 1.5,
                    px: 3,
                    py: 1,
                    fontSize: '0.9375rem',
                    fontWeight: 600,
                    backgroundColor: 'white',
                    borderWidth: 2,
                    '&:hover': { borderWidth: 2 },
                }}>
                {isPaused ? 'Resume Agent' : 'Take Control'}
            </Button>
        </Card>
    );
}
