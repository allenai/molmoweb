'use client';

import { ReactNode } from 'react';

import { VarnishApp } from '@allenai/varnish2/components';

interface ThemeProviderProps {
    children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
    return <VarnishApp layout="left-aligned">{children}</VarnishApp>;
}
