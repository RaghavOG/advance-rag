"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, Brain, GitBranch, Shield, Zap } from "lucide-react";

const pills = [
  { icon: Brain, label: "LangGraph Orchestration" },
  { icon: GitBranch, label: "Multi-Query Decomposition" },
  { icon: Zap, label: "HyDE + Query Rewriting" },
  { icon: Shield, label: "Grounded Citations" },
];

export function Hero() {
  return (
    <section className="relative overflow-hidden px-6 pt-32 pb-24 text-center">
      {/* Background glow */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute left-1/2 top-0 -translate-x-1/2 w-[600px] h-[400px] rounded-full bg-brand/10 blur-[120px]" />
        <div className="absolute left-1/4 top-40 w-[300px] h-[300px] rounded-full bg-purple-600/5 blur-[100px]" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        <span className="inline-flex items-center gap-2 rounded-full border border-brand/30 bg-brand/10 px-4 py-1.5 text-xs font-medium text-brand-light mb-6">
          <span className="h-1.5 w-1.5 rounded-full bg-brand-light animate-pulse-soft" />
          Production-grade · Local deployment · OpenAI
        </span>

        <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6 leading-tight">
          <span className="gradient-text">Multimodal RAG</span>
          <br />
          <span className="text-slate-200">Pipeline</span>
        </h1>

        <p className="max-w-2xl mx-auto text-lg text-slate-400 mb-10 leading-relaxed">
          A text-first Retrieval-Augmented Generation system with LangGraph orchestration,
          failure-aware nodes, clarification loops, adaptive top-k retrieval,
          HyDE, and grounded cited answers.
        </p>

        <div className="flex flex-wrap justify-center gap-3 mb-12">
          {pills.map(({ icon: Icon, label }) => (
            <span
              key={label}
              className="inline-flex items-center gap-2 rounded-full border border-surface-border bg-surface-card px-4 py-2 text-sm text-slate-300"
            >
              <Icon className="h-3.5 w-3.5 text-brand-light" />
              {label}
            </span>
          ))}
        </div>

        <div className="flex flex-wrap justify-center gap-4">
          <Link
            href="/chat"
            className="inline-flex items-center gap-2 rounded-lg bg-brand px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-brand/20 hover:bg-brand-muted transition-colors"
          >
            Open Chat
            <ArrowRight className="h-4 w-4" />
          </Link>
          <a
            href="#pipeline"
            className="inline-flex items-center gap-2 rounded-lg border border-surface-border bg-surface-card px-6 py-3 text-sm font-semibold text-slate-300 hover:bg-surface-elevated transition-colors"
          >
            See Architecture
          </a>
        </div>
      </motion.div>

      {/* Stats row */}
      <motion.div
        className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-px bg-surface-border rounded-xl overflow-hidden max-w-3xl mx-auto"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.4, ease: "easeOut" }}
      >
        {[
          { value: "15", label: "Graph Nodes" },
          { value: "4", label: "Failure Nodes" },
          { value: "HyDE", label: "Query Strategy" },
          { value: "100%", label: "Local Deployment" },
        ].map(({ value, label }) => (
          <div key={label} className="bg-surface-card px-6 py-5 text-center">
            <p className="text-2xl font-bold text-slate-100">{value}</p>
            <p className="text-xs text-slate-500 mt-1">{label}</p>
          </div>
        ))}
      </motion.div>
    </section>
  );
}
