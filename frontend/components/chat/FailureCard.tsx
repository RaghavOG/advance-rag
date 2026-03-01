"use client";

import { motion } from "framer-motion";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  message?: string;
  onRetry?: () => void;
}

export function FailureCard({ message, onRetry }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25 }}
      className="flex items-start gap-3 rounded-lg border border-rose-800/60 bg-rose-950/20 px-4 py-3"
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-rose-400" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-rose-300">
          {message || "I couldn't find relevant information in the indexed documents."}
        </p>
        <p className="mt-1 text-xs text-rose-500">
          The retrieval confidence was too low, or the pipeline timed out.
          No answer was fabricated.
        </p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex-shrink-0 flex items-center gap-1 rounded-md border border-rose-800 bg-rose-950/50 px-2.5 py-1 text-xs text-rose-300 hover:bg-rose-900/40 transition-colors"
        >
          <RefreshCw className="h-3 w-3" />
          Retry
        </button>
      )}
    </motion.div>
  );
}
