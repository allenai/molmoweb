/**
 * Build the display message list from a full session API response.
 * Interleaves step messages (from steps) with user/agent/system messages by order.
 */
import type { ChatMessage, Session, SessionStatus, Step } from '@/types';

interface FullSessionMessage {
    id: string;
    type: string;
    content: string;
    timestamp: string;
    step_ids?: number[];
    is_answer?: boolean;
    is_final?: boolean;
}

interface FullSessionStep {
    id: number;
    status: string;
    thought?: string;
    action_str?: string;
    action_description?: string;
    url?: string;
    screenshot_base64?: string;
    thumbnail_base64?: string;
    error?: string;
    created_at: string;
}

interface FullSessionResponse {
    messages: FullSessionMessage[];
    steps: FullSessionStep[];
}

function stepToUi(s: FullSessionStep): Step {
    return {
        id: s.id,
        status: s.status as Step['status'],
        thought: s.thought,
        actionStr: s.action_str,
        actionDescription: s.action_description,
        url: s.url,
        screenshotBase64: s.screenshot_base64,
        thumbnailBase64: s.thumbnail_base64,
        error: s.error,
        createdAt: s.created_at,
    };
}

export function buildMessagesFromFullSession(data: FullSessionResponse): ChatMessage[] {
    const stepsById = new Map(data.steps.map((s) => [s.id, s]));
    const referencedStepIds = new Set<number>();
    const result: ChatMessage[] = [];

    const sortedMessages = [...data.messages].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    for (const msg of sortedMessages) {
        if (msg.type === 'user' || msg.type === 'system') {
            result.push({
                id: msg.id,
                type: msg.type as 'user' | 'system',
                content: msg.content,
                timestamp: msg.timestamp,
            });
            continue;
        }
        if (msg.type === 'agent') {
            const stepIds = msg.step_ids || [];
            for (const stepId of stepIds) {
                referencedStepIds.add(stepId);
                const s = stepsById.get(stepId);
                if (s) {
                    const step = stepToUi(s);
                    result.push({
                        id: `step-${step.id}`,
                        type: 'step',
                        content: step.actionStr || 'Executing...',
                        timestamp: step.createdAt,
                        step,
                    });
                }
            }
            result.push({
                id: msg.id,
                type: 'agent',
                content: msg.content,
                timestamp: msg.timestamp,
                isAnswer: msg.is_answer || false,
                isFinal: msg.is_final || false,
            });
        }
    }

    // Append orphaned steps not referenced by any agent message
    // (e.g. session stopped mid-execution before an agent message was sent)
    const orphanedSteps = data.steps
        .filter((s) => !referencedStepIds.has(s.id))
        .sort((a, b) => a.id - b.id);

    for (const s of orphanedSteps) {
        const step = stepToUi(s);
        result.push({
            id: `step-${step.id}`,
            type: 'step',
            content: step.actionStr || 'Executing...',
            timestamp: step.createdAt,
            step,
        });
    }

    return result;
}

export function fullSessionToSession(data: {
    id: string;
    goal: string;
    start_url: string;
    status: string;
    created_at: string;
    updated_at: string;
    messages: FullSessionMessage[];
    steps: FullSessionStep[];
    share_token?: string | null;
}): Session {
    return {
        id: data.id,
        goal: data.goal,
        startUrl: data.start_url,
        status: data.status as SessionStatus,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
        messages: buildMessagesFromFullSession({ messages: data.messages, steps: data.steps }),
        shareToken: data.share_token ?? null,
    };
}
