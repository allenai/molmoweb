/**
 * Zustand store for session state management.
 */

import { create } from "zustand";

import type { Session, SessionStatus, ChatMessage, Step } from "@/types";

// Track timeout for click coords auto-clear
let clickCoordsTimeout: any | null = null;

interface SessionState {
  // Current session
  session: Session | null;

  // Connection state
  isConnected: boolean;
  isConnecting: boolean;

  // Current step being executed
  currentStepId: number | null;

  // Current browser screenshot
  currentScreenshot: string | null;
  currentUrl: string;
  liveViewUrl: string | null;

  // Status message
  statusMessage: string | null;

  // Last click coordinates for overlay indicator
  lastClickCoords: { x: number; y: number } | null;

  // Browser closed message
  browserClosedMessage: string | null;

  // Inactivity warning countdown (seconds remaining, or null if no warning)
  inactivityWarning: number | null;

  // Expanded steps (for collapsible UI)
  expandedSteps: Set<number>;

  // Actions
  setSession: (session: Session | null) => void;
  updateSessionStatus: (status: SessionStatus) => void;
  setConnected: (connected: boolean) => void;
  setConnecting: (connecting: boolean) => void;

  // Browser actions
  setCurrentScreenshot: (screenshot: string | null) => void;
  setCurrentUrl: (url: string) => void;
  setLiveViewUrl: (url: string | null) => void;
  setStatusMessage: (message: string | null) => void;
  setLastClickCoords: (coords: { x: number; y: number } | null) => void;
  setBrowserClosedMessage: (message: string | null) => void;
  setInactivityWarning: (seconds: number | null) => void;

  // Message actions
  addMessage: (message: ChatMessage) => void;

  // Step actions
  addStep: (step: Step) => void;
  updateStep: (stepId: number, updates: Partial<Step>) => void;
  setCurrentStepId: (stepId: number | null) => void;
  toggleStepExpanded: (stepId: number) => void;

  // Reset
  reset: () => void;
}

const initialState = {
  session: null,
  isConnected: false,
  isConnecting: false,
  currentStepId: null,
  currentScreenshot: null as string | null,
  currentUrl: "https://allenai.org/",
  liveViewUrl: null as string | null,
  statusMessage: null as string | null,
  lastClickCoords: null as { x: number; y: number } | null,
  browserClosedMessage: null as string | null,
  inactivityWarning: null as number | null,
  expandedSteps: new Set<number>(),
};

export const useSessionStore = create<SessionState>((set, get) => ({
  ...initialState,

  setSession: (session) => set({ session }),

  updateSessionStatus: (status) => {
    const { session } = get();
    if (session) {
      set({
        session: { ...session, status, updatedAt: new Date().toISOString() },
      });
    }
  },

  setConnected: (isConnected) => set({ isConnected }),
  setConnecting: (isConnecting) => set({ isConnecting }),

  setCurrentScreenshot: (currentScreenshot) => set({ currentScreenshot }),
  setCurrentUrl: (currentUrl) => set({ currentUrl }),
  setLiveViewUrl: (liveViewUrl) => set({ liveViewUrl }),
  setStatusMessage: (statusMessage) => set({ statusMessage }),
  setBrowserClosedMessage: (browserClosedMessage) =>
    set({ browserClosedMessage }),
  setInactivityWarning: (inactivityWarning) => set({ inactivityWarning }),
  setLastClickCoords: (lastClickCoords) => {
    // Cancel any pending timeout
    if (clickCoordsTimeout) {
      clearTimeout(clickCoordsTimeout);
      clickCoordsTimeout = null;
    }

    set({ lastClickCoords });

    // Fallback auto-clear after 5 seconds (in case no more steps happen)
    if (lastClickCoords) {
      clickCoordsTimeout = setTimeout(() => {
        set({ lastClickCoords: null });
        clickCoordsTimeout = null;
      }, 1000);
    }
  },

  addMessage: (message) => {
    const { session } = get();
    if (session) {
      set({
        session: {
          ...session,
          messages: [...session.messages, message],
          updatedAt: new Date().toISOString(),
        },
      });
    }
  },

  addStep: (step) => {
    const { session } = get();
    if (session) {
      // Add step as a message in the chat
      const stepMessage: ChatMessage = {
        id: `step-${step.id}`,
        type: "step",
        content: step.actionStr || "Executing...",
        timestamp: step.createdAt,
        step,
      };

      set({
        session: {
          ...session,
          messages: [...session.messages, stepMessage],
          updatedAt: new Date().toISOString(),
        },
        currentStepId: step.id,
      });
    }
  },

  updateStep: (stepId, updates) => {
    const { session } = get();
    if (session) {
      const messages = session.messages.map((msg) => {
        if (msg.type === "step" && msg.step?.id === stepId) {
          return {
            ...msg,
            step: { ...msg.step, ...updates },
            content: updates.actionStr || msg.content,
          };
        }
        return msg;
      });

      set({
        session: {
          ...session,
          messages,
          updatedAt: new Date().toISOString(),
        },
      });
    }
  },

  setCurrentStepId: (currentStepId) => set({ currentStepId }),

  toggleStepExpanded: (stepId) => {
    const { expandedSteps } = get();
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    set({ expandedSteps: newExpanded });
  },

  reset: () => set(initialState),
}));
