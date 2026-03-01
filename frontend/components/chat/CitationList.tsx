"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { ChevronDown, FileText } from "lucide-react";
import type { Citation } from "@/lib/types";
import { truncate } from "@/lib/utils";

export function CitationList({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  if (!citations.length) return null;

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
      >
        <FileText className="h-3 w-3" />
        {citations.length} source{citations.length !== 1 ? "s" : ""}
        <ChevronDown
          className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 space-y-1.5">
              {citations.map((c, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 rounded-md border border-surface-border bg-surface-elevated px-3 py-2"
                >
                  <span className="mt-0.5 flex-shrink-0 h-4 w-4 rounded bg-brand/20 text-brand-light text-[9px] flex items-center justify-center font-bold">
                    {i + 1}
                  </span>
                  <div className="text-[10px] text-slate-400 leading-relaxed">
                    {c.source && <span className="text-slate-300 font-medium">{c.source}</span>}
                    {c.page !== undefined && <span className="text-slate-500"> · p.{c.page}</span>}
                    {c.doc_id && <span className="text-slate-600 ml-1">[{c.doc_id.slice(0, 8)}…]</span>}
                    {c.snippet && (
                      <p className="mt-1 italic text-slate-500">&ldquo;{truncate(c.snippet, 140)}&rdquo;</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
