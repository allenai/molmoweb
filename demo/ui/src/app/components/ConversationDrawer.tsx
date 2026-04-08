'use client';

import { useEffect, useState } from 'react';
import {
    Avatar,
    Box,
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogContentText,
    DialogTitle,
    Drawer,
    IconButton,
    List,
    ListItemButton,
    ListItemText,
    Tooltip,
    Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import LogoutIcon from '@mui/icons-material/Logout';

import type { SessionListItem } from '@/types';

interface User {
    authenticated: boolean;
    email: string | null;
}

function getInitials(email: string): string {
    const name = email.split('@')[0];
    if (name.length >= 2) {
        return name.substring(0, 2).toUpperCase();
    }
    return name.toUpperCase();
}

const GOAL_MAX_LEN = 44;
const DRAWER_WIDTH = 260;
const STRIP_WIDTH = 36;
const WARM_BG = '#faf7f2';
const WARM_BORDER = '#e8e4de';
const WARM_HOVER = 'rgba(0,0,0,0.04)';
const WARM_SELECTED = 'rgba(0,0,0,0.06)';

function formatDate(iso: string): string {
    try {
        const d = new Date(iso);
        const now = new Date();
        const sameDay = d.toDateString() === now.toDateString();
        if (sameDay) return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    } catch {
        return '';
    }
}

function truncateGoal(goal: string): string {
    const t = (goal || '').trim();
    if (!t) return 'New chat';
    if (t.length <= GOAL_MAX_LEN) return t;
    return t.slice(0, GOAL_MAX_LEN - 1) + '…';
}

export interface ConversationDrawerProps {
    sessions: SessionListItem[];
    currentSessionId: string | null;
    onNewChat: () => void;
    onSelectSession: (id: string) => void;
    onDeleteSession?: (id: string) => void;
    onDeleteAllSessions?: () => void;
    loading?: boolean;
}

export function ConversationDrawer({
    sessions,
    currentSessionId,
    onNewChat,
    onSelectSession,
    onDeleteSession,
    onDeleteAllSessions,
    loading,
}: ConversationDrawerProps) {
    const [hoveredId, setHoveredId] = useState<string | null>(null);
    const [open, setOpen] = useState(false);
    const [confirmOpen, setConfirmOpen] = useState(false);
    const [user, setUser] = useState<User | null>(null);

    useEffect(() => {
        const fetchUser = async () => {
            try {
                const response = await fetch('/api/user');
                if (response.ok) {
                    const data = await response.json();
                    setUser(data);
                }
            } catch (error) {
                console.error('Failed to fetch user:', error);
            }
        };
        fetchUser();
    }, []);
    const toggleDrawer = (open: boolean) => () => setOpen(open);

    const iconButtonSx = {
        color: 'text.secondary',
        '&:hover': { bgcolor: WARM_HOVER, color: 'text.primary' },
        transition: 'background-color 0.2s, color 0.2s',
    } as const;

    const drawerList = (
        <Box
            sx={{
                width: DRAWER_WIDTH,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                bgcolor: WARM_BG,
            }}>
            {user?.authenticated && user.email && (
                <Box
                    sx={{
                        px: 2,
                        pt: 2,
                        pb: 1.5,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        borderBottom: `1px solid ${WARM_BORDER}`,
                    }}>
                    <Avatar
                        sx={{
                            width: 28,
                            height: 28,
                            bgcolor: 'primary.main',
                            fontSize: '0.75rem',
                        }}>
                        {getInitials(user.email)}
                    </Avatar>
                    <Typography
                        variant="body2"
                        sx={{
                            flex: 1,
                            minWidth: 0,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                        }}>
                        {user.email}
                    </Typography>
                    <Tooltip title="Sign out">
                        <IconButton
                            size="small"
                            onClick={() => { window.location.href = 'https://google.login.allen.ai/oauth2/sign_out?rd=https://allenai.org/blog/molmoweb'; }}
                            sx={{ color: 'text.secondary', '&:hover': { color: 'text.primary' } }}>
                            <LogoutIcon sx={{ fontSize: 18 }} />
                        </IconButton>
                    </Tooltip>
                </Box>
            )}

            <Box sx={{ px: 2, pt: 2, pb: 1.5 }}>
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        mb: 1.5,
                    }}>
                    <Box
                        component="button"
                        onClick={() => {
                            onNewChat();
                            setOpen(false);
                        }}
                        disabled={loading}
                        sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            border: 0,
                            background: 'none',
                            padding: 0,
                            cursor: loading ? 'default' : 'pointer',
                            color: 'text.primary',
                            fontSize: '0.9375rem',
                            fontFamily: 'inherit',
                            '&:hover': loading ? {} : { color: 'primary.main' },
                            transition: 'color 0.2s',
                        }}>
                        <AddIcon sx={{ fontSize: 18 }} />
                        <span>New chat</span>
                    </Box>
                    <Tooltip title="Close" placement="left">
                        <IconButton onClick={toggleDrawer(false)} sx={iconButtonSx}>
                            <ChevronLeftIcon sx={{ fontSize: 20 }} />
                        </IconButton>
                    </Tooltip>
                </Box>
                {/* <InputBase
                    placeholder="Search chats"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    startAdornment={
                        <InputAdornment position="start" sx={{ mr: 0.5 }}>
                            <SearchIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                        </InputAdornment>
                    }
                    sx={{
                        width: '100%',
                        py: 0.75,
                        px: 1,
                        fontSize: '0.875rem',
                        borderRadius: 1,
                        bgcolor: 'background.paper',
                        border: `1px solid ${WARM_BORDER}`,
                        '&.Mui-focused': { borderColor: 'text.secondary', outline: 'none' },
                        '& input': { py: 0.5 },
                    }}
                /> */}
            </Box>

            <Box sx={{ flex: 1, overflow: 'auto', minHeight: 0, px: 1.5 }}>
                {sessions.length === 0 && !loading && (
                    <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ py: 3, textAlign: 'center', px: 1 }}>
                        No conversations yet
                    </Typography>
                )}
                <List disablePadding sx={{ py: 0.5 }}>
                    {sessions.map((s) => {
                        const isSelected = currentSessionId === s.id;
                        const showMenu = (isSelected || hoveredId === s.id) && onDeleteSession;
                        return (
                            <ListItemButton
                                key={s.id}
                                selected={isSelected}
                                onClick={() => {
                                    onSelectSession(s.id);
                                    setOpen(false);
                                }}
                                onMouseEnter={() => setHoveredId(s.id)}
                                onMouseLeave={() => setHoveredId(null)}
                                disableRipple
                                sx={{
                                    borderRadius: 1,
                                    py: 0.875,
                                    px: 1.25,
                                    mb: 0.25,
                                    '&.Mui-selected': {
                                        bgcolor: WARM_SELECTED,
                                        '&:hover': { bgcolor: WARM_SELECTED },
                                    },
                                    '&:hover': { bgcolor: WARM_HOVER },
                                }}>
                                <ListItemText
                                    primary={truncateGoal(s.goal)}
                                    secondary={formatDate(s.updated_at)}
                                    primaryTypographyProps={{
                                        noWrap: true,
                                        variant: 'body2',
                                        sx: { fontWeight: isSelected ? 500 : 400 },
                                    }}
                                    secondaryTypographyProps={{
                                        variant: 'caption',
                                        sx: { mt: 0.15 },
                                    }}
                                    sx={{ flex: 1, minWidth: 0 }}
                                />
                                {showMenu && (
                                    <IconButton
                                        size="small"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDeleteSession!(s.id);
                                        }}
                                        sx={{
                                            ml: 0.5,
                                            color: 'text.secondary',
                                            '&:hover': { bgcolor: WARM_HOVER, color: 'error.main' },
                                        }}>
                                        <DeleteOutlineIcon sx={{ fontSize: 18 }} />
                                    </IconButton>
                                )}
                            </ListItemButton>
                        );
                    })}
                </List>
            </Box>

            {onDeleteAllSessions && (
                <Box
                    sx={{
                        px: 1.5,
                        py: 1,
                        borderTop: `1px solid ${WARM_BORDER}`,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 0.5,
                    }}>
                    <Tooltip title="Delete all conversations" placement="right">
                        <span>
                            <IconButton
                                onClick={() => setConfirmOpen(true)}
                                disabled={sessions.length === 0}
                                sx={iconButtonSx}>
                                <DeleteForeverIcon sx={{ fontSize: 20 }} />
                            </IconButton>
                        </span>
                    </Tooltip>
                </Box>
            )}
        </Box>
    );

    return (
        <>
            <Box
                sx={{
                    width: STRIP_WIDTH,
                    minWidth: STRIP_WIDTH,
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    pt: 1.5,
                    pb: 1.5,
                    borderRight: `1px solid ${WARM_BORDER}`,
                    bgcolor: WARM_BG,
                }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                    <Tooltip title="Open drawer" placement="right">
                        <IconButton onClick={toggleDrawer(true)} sx={iconButtonSx}>
                            <ChevronRightIcon sx={{ fontSize: 18 }} />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="New chat" placement="right">
                        <IconButton onClick={onNewChat} disabled={loading} sx={iconButtonSx}>
                            <AddIcon sx={{ fontSize: 18 }} />
                        </IconButton>
                    </Tooltip>
                </Box>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                    {onDeleteAllSessions && (
                        <Tooltip title="Delete all conversations" placement="right">
                            <IconButton onClick={toggleDrawer(true)} sx={iconButtonSx}>
                                <DeleteForeverIcon sx={{ fontSize: 18 }} />
                            </IconButton>
                        </Tooltip>
                    )}
                </Box>
            </Box>
            <Drawer
                open={open}
                onClose={toggleDrawer(false)}
                anchor="left"
                PaperProps={{
                    sx: {
                        width: DRAWER_WIDTH,
                        bgcolor: WARM_BG,
                        borderRight: `1px solid ${WARM_BORDER}`,
                    },
                }}>
                {drawerList}
            </Drawer>

            <Dialog
                open={confirmOpen}
                onClose={() => setConfirmOpen(false)}
                PaperProps={{
                    sx: {
                        bgcolor: WARM_BG,
                        borderRadius: 2,
                        maxWidth: 400,
                    },
                }}>
                <DialogTitle sx={{ fontSize: '1rem', fontWeight: 600 }}>
                    Delete all conversations?
                </DialogTitle>
                <DialogContent>
                    <DialogContentText sx={{ fontSize: '0.875rem' }}>
                        This will permanently delete all your conversations and their data.
                        This action cannot be undone.
                    </DialogContentText>
                </DialogContent>
                <DialogActions sx={{ px: 3, pb: 2 }}>
                    <Button
                        onClick={() => setConfirmOpen(false)}
                        sx={{ color: 'text.secondary', textTransform: 'none' }}>
                        Cancel
                    </Button>
                    <Button
                        onClick={() => {
                            setConfirmOpen(false);
                            setOpen(false);
                            onDeleteAllSessions?.();
                        }}
                        variant="contained"
                        color="error"
                        sx={{ textTransform: 'none' }}>
                        Delete all
                    </Button>
                </DialogActions>
            </Dialog>
        </>
    );
}
