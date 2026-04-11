'use client';

import { useState, useEffect } from 'react';
import {
    Box,
    Button,
    Checkbox,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Divider,
    FormControlLabel,
    Link,
    Typography,
    useMediaQuery,
    useTheme,
} from '@mui/material';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const TERMS_ACCEPTED_KEY = 'molmo_web_terms_accepted';

export function TermsModal() {
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const [open, setOpen] = useState(false);
    const [openScienceConsent, setOpenScienceConsent] = useState(false);

    useEffect(() => {
        const localAccepted = localStorage.getItem(TERMS_ACCEPTED_KEY);
        if (!localAccepted) {
            setOpen(true);
        }
    }, []);

    const handleAccept = async () => {
        try {
            await fetch(`${API_URL}/api/consent`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ open_science_consent: openScienceConsent }),
            });
        } catch (err) {
            console.error('Failed to save consent:', err);
        }
        localStorage.setItem(TERMS_ACCEPTED_KEY, 'true');
        setOpen(false);
    };

    return (
        <Dialog
            open={open}
            maxWidth="md"
            fullWidth
            fullScreen={isMobile}
            disableEscapeKeyDown
            slotProps={{
                backdrop: { sx: { backgroundColor: 'rgba(0,0,0,0.5)' } },
            }}
            PaperProps={{
                sx: isMobile
                    ? {}
                    : {
                          borderRadius: 3,
                          maxHeight: 'none',
                          overflow: 'visible',
                          my: 4,
                      },
            }}>
            <DialogTitle sx={{ pt: 4, pb: 0, px: 5 }}>
                <Typography
                    component="span"
                    sx={{
                        fontSize: '1.75rem',
                        fontWeight: 700,
                        lineHeight: 1.2,
                        display: 'block',
                    }}>
                    Terms of Use &amp; Data Consent
                </Typography>
            </DialogTitle>

            <DialogContent sx={{ px: 5, pt: 3, pb: 1, ...(!isMobile && { overflow: 'visible' }) }}>
                <Box
                    sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2,
                        fontSize: '1rem',
                        lineHeight: 1.7,
                        color: 'text.primary',
                    }}>
                    <Typography sx={{ fontSize: 'inherit', lineHeight: 'inherit' }}>
                        By accepting these terms and using MolmoWeb, you confirm that you have read
                        and agree to Ai2&apos;s full{' '}
                        <Link href="https://allenai.org/terms" target="_blank" rel="noopener">
                            Terms of Use
                        </Link>{' '}
                        and{' '}
                        <Link
                            href="https://allenai.org/responsible-use"
                            target="_blank"
                            rel="noopener">
                            Responsible Use Guidelines
                        </Link>{' '}
                        and acknowledge Ai2&apos;s{' '}
                        <Link
                            href="https://allenai.org/privacy-policy"
                            target="_blank"
                            rel="noopener">
                            Privacy Policy
                        </Link>
                        .
                    </Typography>

                    <Typography sx={{ fontSize: 'inherit', lineHeight: 'inherit' }}>
                        You agree not to submit any personal, sensitive, proprietary, or confidential information in your queries. 
                    </Typography>

                    <Typography sx={{ fontSize: 'inherit', lineHeight: 'inherit' }}>
                        You agree that Ai2 may use your queries and responses for AI training and
                        scientific research (e.g. requests, actions, input/output text, trajectories, and demo screenshots).
                    </Typography>
                </Box>

                <Divider sx={{ my: 3 }} />

                <Box>
                    <Typography
                        sx={{ fontWeight: 700, mb: 0.5, fontSize: '1.1rem' }}>
                        Optional Consent for Open Science
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.6 }}>
                        <strong>To use MolmoWeb you only need to agree to the terms above.</strong> To help us
                        improve this technology and accelerate scientific discovery, please consider
                        granting this optional permission:
                    </Typography>

                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={openScienceConsent}
                                onChange={(e) => setOpenScienceConsent(e.target.checked)}
                                sx={{ alignSelf: 'flex-start', mt: -0.25 }}
                            />
                        }
                        label={
                            <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
                                <strong>Help improve open science!</strong> I consent to the
                                inclusion of my de-identified interactions with MolmoWeb in a public research dataset.
                            </Typography>
                        }
                        sx={{ alignItems: 'flex-start', mx: 0 }}
                    />
                </Box>
            </DialogContent>

            <DialogActions sx={{ px: 5, py: 3, justifyContent: 'flex-end' }}>
                <Button
                    variant="contained"
                    onClick={handleAccept}
                    size="large"
                    sx={{
                        textTransform: 'none',
                        fontWeight: 600,
                        borderRadius: 2,
                        px: 4,
                        py: 1,
                    }}>
                    Agree &amp; Use MolmoWeb
                </Button>
            </DialogActions>
        </Dialog>
    );
}
