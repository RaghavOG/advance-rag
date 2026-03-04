"use client";

import { useCallback, useEffect, useState } from "react";
import { sendClarification, sendQuery } from "@/lib/api";
import type { ChatMessage, ConversationState, QueryResponse } from "@/lib/types";

function makeId() {
  return Math.random().toString(36).slice(2, 10);
}

const STORAGE_KEY = "rag_chat_state_v1";

export function useChat() {
  const [state, setState] = useState<ConversationState>({ messages: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── helpers ──────────────────────────────────────────────────────────────

  const addUserMessage = useCallback((content: string): string => {
    const id = makeId();
    setState((prev) => ({
      ...prev,
      messages: [
        ...prev.messages,
        { id, role: "user", content, timestamp: new Date() },
      ],
    }));
    return id;
  }, []);

  const addLoadingMessage = useCallback((): string => {
    const id = makeId();
    setState((prev) => ({
      ...prev,
      messages: [
        ...prev.messages,
        { id, role: "assistant", content: "", timestamp: new Date(), isLoading: true },
      ],
    }));
    return id;
  }, []);

  const resolveMessage = useCallback((id: string, response: QueryResponse) => {
    setState((prev) => ({
      ...prev,
      conversationId: response.conversation_id,
      pendingClarificationIndex:
        response.status === "clarification_required"
          ? response.sub_answers.findIndex(
              (sa) => sa.status === "clarification_required",
            )
          : undefined,
      messages: prev.messages.map((m) =>
        m.id === id
          ? { ...m, isLoading: false, response, content: response.sub_answers[0]?.answer ?? "" }
          : m,
      ),
    }));
  }, []);

  const failMessage = useCallback((id: string, errText: string) => {
    setState((prev) => ({
      ...prev,
      messages: prev.messages.map((m) =>
        m.id === id
          ? {
              ...m,
              isLoading: false,
              response: {
                conversation_id: prev.conversationId ?? "",
                status: "failure",
                sub_answers: [{ query: "", status: "failed", citations: [], answer: errText }],
              },
            }
          : m,
      ),
    }));
  }, []);

  // ── public actions ────────────────────────────────────────────────────────

  const submitQuery = useCallback(
    async (prompt: string, pdfPath?: string) => {
      setError(null);
      addUserMessage(prompt);
      const loadId = addLoadingMessage();
      setLoading(true);
      try {
        const resp = await sendQuery(prompt, state.conversationId, pdfPath);
        resolveMessage(loadId, resp);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setError(msg);
        failMessage(loadId, msg);
      } finally {
        setLoading(false);
      }
    },
    [state.conversationId, addUserMessage, addLoadingMessage, resolveMessage, failMessage],
  );

  const submitClarification = useCallback(
    async (answer: string) => {
      if (!state.conversationId || state.pendingClarificationIndex === undefined) return;
      setError(null);
      addUserMessage(`Clarification: ${answer}`);
      const loadId = addLoadingMessage();
      setLoading(true);
      try {
        const resp = await sendClarification(
          state.conversationId,
          state.pendingClarificationIndex,
          answer,
        );
        resolveMessage(loadId, resp);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Unknown error";
        setError(msg);
        failMessage(loadId, msg);
      } finally {
        setLoading(false);
      }
    },
    [
      state.conversationId,
      state.pendingClarificationIndex,
      addUserMessage,
      addLoadingMessage,
      resolveMessage,
      failMessage,
    ],
  );

  const retryLast = useCallback(async () => {
    const lastUser = [...state.messages].reverse().find((m) => m.role === "user");
    if (lastUser) await submitQuery(lastUser.content);
  }, [state.messages, submitQuery]);

  const clearChat = useCallback(() => {
    setState({ messages: [] });
    setError(null);
    if (typeof window !== "undefined") {
      try {
        window.localStorage.removeItem(STORAGE_KEY);
      } catch {
        // ignore
      }
    }
  }, []);

  const hasPendingClarification = state.pendingClarificationIndex !== undefined;

  // ── persistence: hydrate from / persist to localStorage ─────────────────────

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as ConversationState & {
        messages: Array<ChatMessage & { timestamp: string }>;
      };
      setState({
        ...parsed,
        messages: parsed.messages.map((m) => ({
          ...m,
          timestamp: new Date(m.timestamp),
        })),
      });
    } catch {
      // ignore corrupt cache
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const serialisable: ConversationState & {
        messages: Array<ChatMessage & { timestamp: string }>;
      } = {
        ...state,
        messages: state.messages.map((m) => ({
          ...m,
          timestamp: m.timestamp.toISOString(),
        })),
      };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(serialisable));
    } catch {
      // best-effort only
    }
  }, [state]);

  return {
    messages: state.messages,
    loading,
    error,
    hasPendingClarification,
    submitQuery,
    submitClarification,
    retryLast,
    clearChat,
  };
}
