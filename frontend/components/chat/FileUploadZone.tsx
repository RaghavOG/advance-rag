"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Upload, FileText, X, Clock } from "lucide-react";

export function FileUploadZone({ onClose }: { onClose: () => void }) {
  const [dragging, setDragging] = useState(false);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 8 }}
        transition={{ duration: 0.2 }}
        className="mb-3 rounded-xl border-2 border-dashed border-surface-border bg-surface-card p-5 relative"
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); }}
      >
        <button
          onClick={onClose}
          className="absolute right-3 top-3 text-slate-600 hover:text-slate-400 transition-colors"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="flex flex-col items-center gap-3 text-center">
          <div className={`h-10 w-10 rounded-xl flex items-center justify-center transition-colors ${
            dragging ? "bg-brand/20" : "bg-surface-elevated"
          }`}>
            <Upload className={`h-5 w-5 ${dragging ? "text-brand-light" : "text-slate-500"}`} />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-300">
              Drop a PDF here
            </p>
            <p className="text-xs text-slate-600 mt-0.5">
              or click to browse
            </p>
          </div>

          {/* Coming soon badge */}
          <div className="flex items-center gap-1.5 rounded-full border border-amber-800/50 bg-amber-950/30 px-3 py-1">
            <Clock className="h-3 w-3 text-amber-500" />
            <span className="text-[10px] text-amber-400 font-medium">
              File upload — coming soon. The ingestion pipeline is ready.
            </span>
          </div>

          <div className="flex flex-wrap gap-2 justify-center">
            {["PDF", "Markdown", "Text"].map((t) => (
              <span
                key={t}
                className="inline-flex items-center gap-1 rounded-md border border-surface-border bg-surface-elevated px-2 py-0.5 text-[10px] text-slate-500"
              >
                <FileText className="h-2.5 w-2.5" />
                {t}
              </span>
            ))}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
