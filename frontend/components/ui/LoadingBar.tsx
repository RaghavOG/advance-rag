"use client";

import { motion } from "framer-motion";

export function LoadingBar({ label = "Retrieving context…" }: { label?: string }) {
  return (
    <div className="w-full space-y-1.5">
      <p className="text-xs text-slate-500">{label}</p>
      <div className="relative h-0.5 w-full overflow-hidden rounded-full bg-surface-elevated">
        <motion.div
          className="absolute h-full rounded-full bg-brand"
          animate={{ left: ["0%", "60%", "0%"] }}
          transition={{ duration: 1.8, ease: "easeInOut", repeat: Infinity }}
          style={{ width: "35%" }}
        />
      </div>
    </div>
  );
}
