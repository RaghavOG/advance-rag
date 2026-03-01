"use client";

import { motion } from "framer-motion";
import { CheckCircle2, XCircle } from "lucide-react";
import type { SubAnswer } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { CitationList } from "./CitationList";
import { ReasoningSection } from "./ReasoningSection";
import { FailureCard } from "./FailureCard";
import { LoadingBar } from "@/components/ui/LoadingBar";
import { formatConfidence } from "@/lib/utils";

interface Props {
  subAnswer: SubAnswer;
  index: number;
  onRetry?: () => void;
}

const statusBadge = {
  answered: { variant: "success" as const, label: "Answered", icon: CheckCircle2 },
  failed:   { variant: "danger"  as const, label: "Failed",   icon: XCircle },
  clarification_required: { variant: "warning" as const, label: "Clarification needed", icon: null },
  processing: { variant: "muted" as const, label: "Processing…", icon: null },
};

export function SubAnswerCard({ subAnswer, index, onRetry }: Props) {
  const badge = statusBadge[subAnswer.status] ?? statusBadge.processing;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, duration: 0.25, ease: "easeOut" }}
      className="rounded-lg border border-surface-border bg-surface-card p-4 space-y-3"
    >
      {/* Query label */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-mono text-slate-400 leading-relaxed flex-1">
          <span className="text-slate-600 mr-1.5">Q{index + 1}.</span>
          {subAnswer.query}
        </p>
        <div className="flex items-center gap-2 flex-shrink-0">
          {subAnswer.confidence !== undefined && (
            <Badge variant="muted">{formatConfidence(subAnswer.confidence)} conf.</Badge>
          )}
          <Badge variant={badge.variant}>{badge.label}</Badge>
        </div>
      </div>

      {/* Loading */}
      {subAnswer.status === "processing" && (
        <LoadingBar label="Retrieving and compressing context…" />
      )}

      {/* Failure */}
      {subAnswer.status === "failed" && (
        <FailureCard message={subAnswer.answer} onRetry={onRetry} />
      )}

      {/* Answer */}
      {subAnswer.status === "answered" && subAnswer.answer && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          transition={{ duration: 0.22, ease: "easeOut" }}
          className="overflow-hidden"
        >
          <div className="answer-prose text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">
            {subAnswer.answer}
          </div>
          {subAnswer.reasoning && (
            <ReasoningSection reasoning={subAnswer.reasoning} />
          )}
          <CitationList citations={subAnswer.citations} />
        </motion.div>
      )}
    </motion.div>
  );
}
