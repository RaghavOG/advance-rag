"use client";

import { motion } from "framer-motion";
import type { ChatMessage } from "@/lib/types";
import { SubAnswerCard } from "./SubAnswerCard";
import { ClarificationBox } from "./ClarificationBox";
import { FailureCard } from "./FailureCard";
import { LoadingBar } from "@/components/ui/LoadingBar";
import { Badge } from "@/components/ui/Badge";
import { User2, Bot } from "lucide-react";

interface Props {
  message: ChatMessage;
  onClarify?: (answer: string) => void;
  onRetry?: () => void;
  loading?: boolean;
}

export function QueryBlock({ message, onClarify, onRetry, loading }: Props) {
  if (message.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="flex gap-3 justify-end"
      >
        <div className="max-w-xl">
          <div className="rounded-2xl rounded-tr-sm bg-brand/20 border border-brand/30 px-4 py-3 text-sm text-slate-200 leading-relaxed">
            {message.content}
          </div>
          <p className="text-[10px] text-slate-600 mt-1 text-right">
            {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </p>
        </div>
        <div className="flex-shrink-0 h-7 w-7 rounded-full bg-brand/30 border border-brand/40 flex items-center justify-center">
          <User2 className="h-3.5 w-3.5 text-brand-light" />
        </div>
      </motion.div>
    );
  }

  // ── Assistant turn ─────────────────────────────────────────────────────────
  const response = message.response;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: "easeOut" }}
      className="flex gap-3"
    >
      <div className="flex-shrink-0 h-7 w-7 rounded-full bg-surface-elevated border border-surface-border flex items-center justify-center">
        <Bot className="h-3.5 w-3.5 text-slate-400" />
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {/* Loading skeleton */}
        {message.isLoading && !response && (
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <LoadingBar label="Running RAG pipeline…" />
          </div>
        )}

        {/* Hard failure with no sub-answers */}
        {response?.status === "failure" && !response.sub_answers.length && (
          <FailureCard
            message={response.error_message ?? "Pipeline failed to produce a response."}
            onRetry={onRetry}
          />
        )}

        {/* Sub-answer cards */}
        {response?.sub_answers.map((sa, i) => {
          if (sa.status === "clarification_required") {
            return (
              <div key={i} className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="warning">Sub-query {i + 1}</Badge>
                  <span className="text-xs text-slate-500 font-mono truncate">{sa.query}</span>
                </div>
                <ClarificationBox
                  question={sa.clarification_question ?? "Could you clarify this question?"}
                  onSubmit={onClarify ?? (() => {})}
                  disabled={loading}
                />
              </div>
            );
          }

          return (
            <SubAnswerCard
              key={i}
              subAnswer={sa}
              index={i}
              onRetry={onRetry}
            />
          );
        })}

        {/* Overall status badge */}
        {response && (
          <div className="flex items-center gap-2">
            {response.status === "partial" && (
              <Badge variant="warning">Partial — some sub-queries failed</Badge>
            )}
            {response.status === "answered" && response.sub_answers.length > 1 && (
              <Badge variant="success">{response.sub_answers.length} questions answered</Badge>
            )}
            <p className="text-[10px] text-slate-600">
              {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              {response.conversation_id && (
                <span className="ml-2 opacity-50">#{response.conversation_id.slice(0, 6)}</span>
              )}
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
