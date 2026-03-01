"use client";

import { motion } from "framer-motion";
import {
  AlertTriangle, BookOpen, Brain, CheckCircle2,
  GitMerge, Layers, MessageSquare, Search, Zap,
} from "lucide-react";

const features = [
  {
    icon: GitMerge,
    title: "Multi-Query Decomposition",
    description:
      "Detects multiple independent questions in a single prompt. Each sub-query runs through a full RAG cycle — they never share context.",
    color: "text-indigo-400",
    bg: "bg-indigo-950/40 border-indigo-800",
  },
  {
    icon: MessageSquare,
    title: "Clarification Loops",
    description:
      "When a query is ambiguous, the system asks one targeted clarification. The UI locks further execution until clarification is provided.",
    color: "text-amber-400",
    bg: "bg-amber-950/30 border-amber-800",
  },
  {
    icon: AlertTriangle,
    title: "Failure Transparency",
    description:
      "Retrieval failure, compression failure, and LLM timeout are explicit nodes — not exceptions. Users see exactly what went wrong.",
    color: "text-rose-400",
    bg: "bg-rose-950/30 border-rose-800",
  },
  {
    icon: Zap,
    title: "HyDE + Query Rewriting",
    description:
      "Generates a hypothetical reference document (HyDE) and 2–4 alternative phrasings to maximize retrieval recall before ranking.",
    color: "text-purple-400",
    bg: "bg-purple-950/30 border-purple-800",
  },
  {
    icon: Search,
    title: "Adaptive Top-K",
    description:
      "Short factual queries retrieve top-3; explanatory or multi-rewrite queries scale up to top-10. Per-rewrite k stays modest.",
    color: "text-cyan-400",
    bg: "bg-cyan-950/30 border-cyan-800",
  },
  {
    icon: BookOpen,
    title: "Grounded Citations",
    description:
      "Answers cite doc_id and page numbers. The system explicitly separates retrieved facts from synthesized reasoning.",
    color: "text-emerald-400",
    bg: "bg-emerald-950/30 border-emerald-800",
  },
  {
    icon: Brain,
    title: "Context Compression",
    description:
      "Retrieved chunks are compressed by an LLM before generation. Tagged with markers to reduce prompt-injection risk.",
    color: "text-violet-400",
    bg: "bg-violet-950/30 border-violet-800",
  },
  {
    icon: CheckCircle2,
    title: "Structured Responses",
    description:
      "The API never returns raw strings. Every response carries status, per-sub-query answers, confidence scores, and citations.",
    color: "text-teal-400",
    bg: "bg-teal-950/30 border-teal-800",
  },
  {
    icon: Layers,
    title: "Modular Architecture",
    description:
      "Each layer (ingestion, embedding, retrieval, compression, generation) is independently replaceable behind clean interfaces.",
    color: "text-slate-300",
    bg: "bg-slate-800/40 border-slate-700",
  },
];

export function FeatureSection() {
  return (
    <section className="px-6 py-20 max-w-6xl mx-auto" id="features">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
        className="text-center mb-14"
      >
        <h2 className="text-3xl font-bold text-slate-100 mb-3">
          What makes this different
        </h2>
        <p className="text-slate-400 max-w-xl mx-auto">
          Production patterns that most RAG tutorials skip entirely.
        </p>
      </motion.div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {features.map(({ icon: Icon, title, description, color, bg }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.05, duration: 0.3 }}
            className={`rounded-xl border p-5 ${bg}`}
          >
            <Icon className={`h-5 w-5 mb-3 ${color}`} />
            <h3 className="text-sm font-semibold text-slate-100 mb-1.5">{title}</h3>
            <p className="text-xs text-slate-400 leading-relaxed">{description}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
