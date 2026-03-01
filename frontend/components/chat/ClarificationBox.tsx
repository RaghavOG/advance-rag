"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { HelpCircle, Send } from "lucide-react";

interface Props {
  question: string;
  onSubmit: (answer: string) => void;
  disabled?: boolean;
}

export function ClarificationBox({ question, onSubmit, disabled }: Props) {
  const [answer, setAnswer] = useState("");

  const submit = () => {
    const trimmed = answer.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
    setAnswer("");
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.99 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
      className="rounded-lg border border-amber-700/60 bg-amber-950/20 p-4"
    >
      {/* Soft pulse ring to signal it needs attention */}
      <motion.div
        animate={{ boxShadow: ["0 0 0 0px rgba(245,158,11,0)", "0 0 0 4px rgba(245,158,11,0.08)", "0 0 0 0px rgba(245,158,11,0)"] }}
        transition={{ duration: 2.2, repeat: Infinity }}
        className="rounded-lg"
      >
        <div className="flex items-start gap-3 mb-3">
          <HelpCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
          <div>
            <p className="text-xs font-semibold text-amber-300 uppercase tracking-wider mb-1">
              Clarification needed
            </p>
            <p className="text-sm text-slate-200">{question}</p>
          </div>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !disabled && submit()}
            placeholder="Type your clarification…"
            disabled={disabled}
            className="flex-1 rounded-md border border-surface-border bg-surface-elevated px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-amber-600 transition-colors disabled:opacity-50"
          />
          <button
            onClick={submit}
            disabled={disabled || !answer.trim()}
            className="flex items-center gap-1.5 rounded-md bg-amber-600 px-3 py-2 text-xs font-semibold text-white hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="h-3.5 w-3.5" />
            Answer
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
