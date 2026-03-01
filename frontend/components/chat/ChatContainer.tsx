"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { QueryBlock } from "./QueryBlock";
import { ChatInput } from "./ChatInput";
import { useChat } from "@/hooks/useChat";
import { Brain, Trash2 } from "lucide-react";

export function ChatContainer() {
  const { messages, loading, hasPendingClarification, submitQuery, submitClarification, retryLast, clearChat } =
    useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-surface-border bg-[#0f1117]/80 px-5 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-brand-light" />
          <span className="text-sm font-semibold text-slate-200">RAG Chat</span>
          {loading && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-brand/10 px-2 py-0.5 text-[10px] text-brand-light">
              <span className="h-1.5 w-1.5 rounded-full bg-brand-light animate-pulse" />
              Processing
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <a href="/" className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
            ← Home
          </a>
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="flex items-center gap-1 rounded-md border border-surface-border px-2 py-1 text-xs text-slate-500 hover:text-slate-300 hover:bg-surface-elevated transition-colors"
            >
              <Trash2 className="h-3 w-3" />
              Clear
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="flex flex-col items-center justify-center h-full text-center gap-4"
          >
            <div className="h-12 w-12 rounded-2xl bg-surface-elevated border border-surface-border flex items-center justify-center">
              <Brain className="h-6 w-6 text-brand-light" />
            </div>
            <div>
              <p className="text-slate-300 font-medium mb-1">Ask anything about your documents</p>
              <p className="text-xs text-slate-600 max-w-xs">
                Multi-question prompts are supported. The pipeline will answer each question independently
                and merge the results.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2 max-w-sm w-full">
              {[
                "What is the main contribution of this paper?",
                "How does the system handle failures? And also what are the retrieval strategies?",
              ].map((eg) => (
                <button
                  key={eg}
                  onClick={() => submitQuery(eg)}
                  className="text-left rounded-lg border border-surface-border bg-surface-card px-3 py-2.5 text-xs text-slate-400 hover:bg-surface-elevated hover:text-slate-200 transition-colors"
                >
                  &ldquo;{eg}&rdquo;
                </button>
              ))}
            </div>
          </motion.div>
        )}

        <div className="max-w-3xl mx-auto space-y-6">
          <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <QueryBlock
                key={msg.id}
                message={msg}
                onClarify={submitClarification}
                onRetry={retryLast}
                loading={loading}
              />
            ))}
          </AnimatePresence>
        </div>
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="max-w-3xl w-full mx-auto">
        <ChatInput
          onSubmit={submitQuery}
          disabled={loading}
          hasPendingClarification={hasPendingClarification}
        />
      </div>
    </div>
  );
}
