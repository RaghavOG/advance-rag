"use client";

import { useState } from "react";
import { ArrowUp, Paperclip } from "lucide-react";
import { cn } from "@/lib/utils";
import { FileUploadZone } from "./FileUploadZone";

interface Props {
  onSubmit: (prompt: string) => void;
  disabled?: boolean;
  hasPendingClarification?: boolean;
}

export function ChatInput({ onSubmit, disabled, hasPendingClarification }: Props) {
  const [value, setValue] = useState("");
  const [showUpload, setShowUpload] = useState(false);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
  };

  const blocked = hasPendingClarification && !disabled;

  return (
    <div className="border-t border-surface-border bg-[#0f1117]/90 backdrop-blur-sm px-4 py-3">
      {showUpload && <FileUploadZone onClose={() => setShowUpload(false)} />}

      {blocked && (
        <div className="mb-2 flex items-center gap-2 rounded-md border border-amber-800/40 bg-amber-950/20 px-3 py-2">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse-soft" />
          <p className="text-xs text-amber-400">
            Please answer the clarification question above before sending a new query.
          </p>
        </div>
      )}

      <div className={cn(
        "flex items-end gap-2 rounded-xl border bg-surface-card px-3 py-2.5 transition-colors",
        disabled || blocked
          ? "border-surface-border opacity-60"
          : "border-surface-border focus-within:border-brand/60",
      )}>
        {/* Upload button */}
        <button
          type="button"
          onClick={() => setShowUpload((s) => !s)}
          disabled={disabled || blocked}
          className="flex-shrink-0 p-1.5 rounded-md text-slate-500 hover:text-slate-300 hover:bg-surface-elevated transition-colors disabled:opacity-40"
          title="Upload document (coming soon)"
        >
          <Paperclip className="h-4 w-4" />
        </button>

        {/* Textarea */}
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={blocked ? "Answer clarification above first…" : "Ask a question about your documents…"}
          disabled={disabled || blocked}
          rows={1}
          className="flex-1 resize-none bg-transparent text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none leading-relaxed max-h-32 overflow-y-auto disabled:cursor-not-allowed"
          style={{ minHeight: "24px" }}
        />

        {/* Send */}
        <button
          type="button"
          onClick={submit}
          disabled={!value.trim() || disabled || blocked}
          className="flex-shrink-0 h-7 w-7 rounded-lg bg-brand flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed hover:bg-brand-muted transition-colors"
        >
          <ArrowUp className="h-3.5 w-3.5 text-white" />
        </button>
      </div>

      <p className="text-center text-[10px] text-slate-700 mt-2">
        Multi-question prompts supported · Shift+Enter for new line
      </p>
    </div>
  );
}
