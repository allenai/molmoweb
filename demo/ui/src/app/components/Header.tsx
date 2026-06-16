'use client';

import Image from 'next/image';
import { Button, Typography } from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

import { HeaderLogo, HeaderMenuColumn, VarnishHeader } from './VarnishHeader';

interface HeaderProps {
    onFaqToggle?: () => void;
}

export default function Header({ onFaqToggle }: HeaderProps) {
    return (
        <VarnishHeader columns="auto 1fr auto">
            <HeaderLogo
                label={
                    <Typography
                        component="span"
                        variant="h3"
                        sx={{
                            color: '#F0529C',
                        }}>
                        MolmoWeb
                    </Typography>
                }>
                <Image src="/ai2-logo.svg" alt="AI2 Logo" width={40} height={40} />
            </HeaderLogo>

            {onFaqToggle && (
                <HeaderMenuColumn>
                    <Button
                        onClick={onFaqToggle}
                        size="large"
                        startIcon={<HelpOutlineIcon sx={{ fontSize: 16 }} />}
                        sx={{
                            textTransform: 'none',
                            color: 'text.secondary',
                            fontSize: '1rem',
                            '& .MuiButton-startIcon': { mr: 0 },
                        }}>
                        FAQs
                    </Button>
                </HeaderMenuColumn>
            )}
        </VarnishHeader>
    );
}
