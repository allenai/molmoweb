import { Content } from '@allenai/varnish2/components';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v14-appRouter';
import type { Metadata } from 'next';
import type { ReactNode } from 'react';

import { ThemeProvider } from './components/ThemeProvider';

import '@fontsource/lato/300-italic.css';
import '@fontsource/lato/300.css';
import '@fontsource/lato/400-italic.css';
import '@fontsource/lato/400.css';
import '@fontsource/lato/700-italic.css';
import '@fontsource/lato/700.css';
import './globals.css';

export const metadata: Metadata = {
    title: 'Molmo Web',
    description: 'Molmo Web Agent Demo',
};

// This layout will be applied to every page in the app.
// To learn more about layouts in NextJS, see their docs: https://nextjs.org/docs/app/building-your-application/routing/pages-and-layouts#layouts
export default function RootLayout({ children }: { children: ReactNode }) {
    return (
        <html lang="en">
            <body style={{ backgroundColor: '#faf7f2' }}>
                <AppRouterCacheProvider>
                    <ThemeProvider>
                        <div
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                flex: 1,
                                minHeight: 0,
                            }}>
                            <Content main className="main-fill-height">
                                {children}
                            </Content>
                            {/* <Footer /> */}
                        </div>
                    </ThemeProvider>
                </AppRouterCacheProvider>
            </body>
        </html>
    );
}
