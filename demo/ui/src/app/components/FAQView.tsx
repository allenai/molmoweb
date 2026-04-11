'use client';

import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Box,
    Button,
    Chip,
    Typography,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface FAQViewProps {
    onBack: () => void;
}

const accordionSx = {
    bgcolor: 'transparent',
    boxShadow: 'none',
    '&:before': { display: 'none' },
    '&.Mui-expanded': { my: 0 },
} as const;

const summarySx = {
    px: 0,
    fontWeight: 600,
    fontSize: '1.1rem',
    '&.Mui-expanded': { minHeight: 48 },
    '& .MuiAccordionSummary-content.Mui-expanded': { my: 1.5 },
} as const;

export function FAQView({ onBack }: FAQViewProps) {
    return (
        <Box
            sx={{
                flex: 1,
                overflow: 'auto',
                px: { xs: 3, md: 6 },
                py: 4,
            }}>
            <Box sx={{ maxWidth: 800, mx: 'auto' }}>
                <Button
                    onClick={onBack}
                    startIcon={<ArrowBackIcon />}
                    sx={{
                        textTransform: 'none',
                        color: 'text.secondary',
                        mb: 3,
                    }}>
                    Back to MolmoWeb
                </Button>

                <Typography variant="h3" fontWeight={700} sx={{ mb: 4 }}>
                    MolmoWeb FAQ
                </Typography>

                {/* What makes a good prompt? */}
                <Accordion defaultExpanded sx={accordionSx}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={summarySx}>
                        What makes a good prompt?
                    </AccordionSummary>
                    <AccordionDetails sx={{ px: 0 }}>
                        <Typography sx={{ mb: 2 }}>
                            Be specific and direct. Tell the agent what information you need and
                            where to get it from. Good prompts typically:
                        </Typography>
                        <Box component="ul" sx={{ pl: 2.5, '& li': { mb: 1.5, lineHeight: 1.7 } }}>
                            <li>
                                <strong>Start with a clear target website or starting point</strong>,
                                e.g. &quot;On wikipedia…&quot;, &quot;Go to
                                https://github.com/trending…&quot;.
                            </li>
                            <li>
                                <strong>Be precise about your goal</strong>, e.g. &quot;find the
                                population of Seattle&quot;, &quot;find a one-bedroom apartment under
                                $1500&quot;. Avoid queries such as &quot;find me something
                                good&quot;.
                            </li>
                            <li>
                                <strong>Include constraints or details</strong> — dates, number of
                                guests, filters, language, price ranges, etc. so the agent doesn&apos;t
                                have to guess.
                            </li>
                            <li>
                                <strong>Avoid giving too many constraints at once.</strong> If you
                                have a complex task, break it into smaller components and give them to
                                the agent one-by-one as follow-up messages.
                            </li>
                            <li>
                                <strong>Try your queries multiple times.</strong>
                                MolmoWeb might take a different sequence of actions every time you run the same query. So a query that failed once may succeed on a second trial. If your query didn’t work, try rephrasing it by being explicit about what to click or what to find on a particular page. <br />

                            </li>
                            
                        </Box>
                        <Typography sx={{ mt: 3, mb: 1.5, fontWeight: 600 }}>
                            Examples:
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    gap: 1.5,
                                    p: 2,
                                    borderRadius: 2,
                                    bgcolor: 'rgba(46, 125, 50, 0.06)',
                                    borderLeft: '3px solid',
                                    borderColor: 'success.main',
                                }}>
                                <Chip label="Good" size="small" color="success" sx={{ mt: 0.25, flexShrink: 0 }} />
                                <Typography variant="body2" sx={{ lineHeight: 1.6, fontStyle: 'italic' }}>
                                    &quot;On wikipedia, search for &apos;Allen Institute for
                                    AI&apos;, click on Teams navlink, and tell me about the PRIOR
                                    team&quot;
                                </Typography>
                            </Box>
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    gap: 1.5,
                                    p: 2,
                                    borderRadius: 2,
                                    bgcolor: 'rgba(46, 125, 50, 0.06)',
                                    borderLeft: '3px solid',
                                    borderColor: 'success.main',
                                }}>
                                <Chip label="Good" size="small" color="success" sx={{ mt: 0.25, flexShrink: 0 }} />
                                <Typography variant="body2" sx={{ lineHeight: 1.6, fontStyle: 'italic' }}>
                                    &quot;On wikipedia, search for Allen Institute for AI, find the
                                    Teams section, and tell me about the PRIOR team&quot;
                                </Typography>
                            </Box>
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'flex-start',
                                    gap: 1.5,
                                    p: 2,
                                    borderRadius: 2,
                                    bgcolor: 'rgba(211, 47, 47, 0.06)',
                                    borderLeft: '3px solid',
                                    borderColor: 'error.main',
                                }}>
                                <Chip label="Bad" size="small" color="error" sx={{ mt: 0.25, flexShrink: 0 }} />
                                <Typography variant="body2" sx={{ lineHeight: 1.6, fontStyle: 'italic' }}>
                                    &quot;On Wikipedia search info about the PRIOR team at
                                    Ai2&quot;
                                </Typography>
                            </Box>
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* How to use */}
                <Accordion sx={accordionSx}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={summarySx}>
                        How to use the MolmoWeb demo?
                    </AccordionSummary>
                    <AccordionDetails sx={{ px: 0 }}>
                        <Box component="ul" sx={{ pl: 2.5, '& li': { mb: 1.5, lineHeight: 1.7 } }}>
                            <li>
                                Type your task into the chat box on the right and press Enter. You
                                can also click one of the pre-written suggestion chips to try an
                                example.
                            </li>
                            <li>
                                The agent will start working immediately. You can watch it browse in
                                real time in the center browser panel. (All web interactions happen in
                                a browser hosted remotely by BrowserBase, not on your personal
                                browser.)
                            </li>
                            <li>
                                While the agent is working, each action appears as a step card in the
                                chat. Click a step card to expand it and see the agent&apos;s
                                reasoning, what action it took, and a screenshot of the page at that
                                moment.
                            </li>
                            <li>
                                You can toggle the <strong>click indicator</strong> in the browser
                                toolbar to see exactly where the agent clicked. If the agent gets
                                stuck, click <strong>Take Control</strong> to interact with the
                                browser yourself, then <strong>Resume Agent</strong> to hand control
                                back.
                            </li>
                            <li>
                                You can also <strong>Stop</strong> the agent at any time and send a
                                follow-up message to redirect it.
                            </li>
                            <li>
                                Your past conversations are listed in the left sidebar. Click any
                                conversation to reload its full history, including all steps and
                                screenshots. You can delete individual conversations or all of them
                                from the sidebar.
                            </li>
                            <li>
                                You can also <strong>Share</strong> a conversation — this generates a
                                read-only link anyone can view.
                            </li>
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* What NOT to do */}
                <Accordion sx={accordionSx}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={summarySx}>
                        What should I NOT do with MolmoWeb?
                    </AccordionSummary>
                    <AccordionDetails sx={{ px: 0 }}>
                        <Box component="ul" sx={{ pl: 2.5, '& li': { mb: 1.5, lineHeight: 1.7 } }}>
                            <li>
                                <strong>Don&apos;t share passwords, credentials, or API keys.</strong>{' '}
                                Never ask the agent to sign in to your accounts, because the browser session
                                is not private to you.
                            </li>
                            <li>
                                <strong>Don&apos;t create new accounts.</strong> Do not ask the agent
                                to register or sign up for services on your behalf.
                            </li>
                            <li>
                                <strong>
                                    Don&apos;t submit personal or sensitive information.
                                </strong>{' '}
                                No credit card numbers, medical records, or other private data.
                            </li>
                            <li>
                                <strong>
                                    Don&apos;t make purchases or financial transactions.
                                </strong>
                            </li>
                            <li>
                                <strong>Don&apos;t attempt harmful, illegal, or abusive tasks.</strong>{' '}
                                This local build does not run automated content moderation; you are
                                responsible for lawful, ethical use.
                            </li>
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* Limitations */}
                <Accordion sx={accordionSx}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={summarySx}>
                        Limitations of MolmoWeb
                    </AccordionSummary>
                    <AccordionDetails sx={{ px: 0 }}>
                        <Box component="ul" sx={{ pl: 2.5, '& li': { mb: 1.5, lineHeight: 1.7 } }}>
                            <li>
                                As a vision-based agent, MolmoWeb may make mistakes while answering
                                questions or reading text from screenshots.{' '}
                                <strong>Always fact-check your answers.</strong>
                            </li>
                            <li>
                                The model has limited ability to follow subsequent queries and
                                performance may degrade with more follow-up queries.
                            </li>
                            <li>
                                MolmoWeb may struggle with actions requiring drag-and-drop or
                                scrolling at a specific element (as opposed to scrolling the entire
                                page).
                            </li>
                            <li>
                                MolmoWeb may get stuck in a loop while predicting the same action
                                repeatedly, like scrolling or clicking at the same location without
                                success.
                            </li>
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* Delete conversations */}
                <Accordion sx={accordionSx}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={summarySx}>
                        Can I delete my conversations?
                    </AccordionSummary>
                    <AccordionDetails sx={{ px: 0 }}>
                        <Typography>
                            Yes. You can delete individual conversations or all conversations from
                            the left sidebar. Deleting a conversation removes it from the server and
                            your local cache.
                        </Typography>
                    </AccordionDetails>
                </Accordion>

                <Accordion sx={accordionSx}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={summarySx}>
                        Which websites can MolmoWeb visit?
                    </AccordionSummary>
                    <AccordionDetails sx={{ px: 0 }}>
                        <Typography>
                            This local demo does not restrict which sites the agent may open. You are
                            responsible for how you use it; only browse sites you are allowed to
                            automate and respect their terms of use.
                        </Typography>
                    </AccordionDetails>
                </Accordion>

                <Box sx={{ height: 48 }} />
            </Box>
        </Box>
    );
}
