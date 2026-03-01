import Link from "next/link";
import { ArrowRight, Github } from "lucide-react";
import { Hero } from "@/components/home/Hero";
import { PipelineFlow } from "@/components/home/PipelineFlow";
import { FeatureSection } from "@/components/home/FeatureSection";

export default function HomePage() {
  return (
    <main className="min-h-screen">
      {/* Nav */}
      <nav className="fixed inset-x-0 top-0 z-50 border-b border-surface-border bg-[#0f1117]/80 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-100">
            Multimodal <span className="text-brand-light">RAG</span>
          </span>
          <div className="flex items-center gap-4">
            <a href="#pipeline" className="text-xs text-slate-400 hover:text-slate-200 transition-colors hidden md:block">
              Pipeline
            </a>
            <a href="#features" className="text-xs text-slate-400 hover:text-slate-200 transition-colors hidden md:block">
              Features
            </a>
            <Link
              href="/chat"
              className="inline-flex items-center gap-1.5 rounded-lg bg-brand px-4 py-1.5 text-xs font-semibold text-white hover:bg-brand-muted transition-colors"
            >
              Chat <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </nav>

      <Hero />
      <PipelineFlow />
      <FeatureSection />

      {/* Invariants section */}
      <section className="px-6 py-20 max-w-3xl mx-auto">
        <h2 className="text-2xl font-bold text-slate-100 mb-8 text-center">System Invariants</h2>
        <div className="space-y-3">
          {[
            ["1", "One question → one retrieval context", "Sub-queries never share retrieved documents."],
            ["2", "Multi-query ≠ query rewriting", "Rewriting improves recall. Splitting handles intent."],
            ["3", "LLMs never silently control flow", "All routing is explicit in graph conditional edges."],
            ["4", "Failures are nodes, not exceptions", "Every failure has a recovery strategy and user message."],
            ["5", "Clarification fires once, max", "clarification_used flag enforced in state."],
            ["6", "Embeddings created in one place", "embeddings/factory.py:get_embedder() is the sole source."],
          ].map(([num, title, desc]) => (
            <div
              key={num}
              className="flex gap-4 p-4 rounded-lg border border-surface-border bg-surface-card"
            >
              <span className="flex-shrink-0 h-6 w-6 rounded-full bg-brand/20 text-brand-light text-xs flex items-center justify-center font-bold">
                {num}
              </span>
              <div>
                <p className="text-sm font-semibold text-slate-200">{title}</p>
                <p className="text-xs text-slate-500 mt-0.5">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-surface-border px-6 py-8 text-center">
        <p className="text-xs text-slate-600">
          Multimodal RAG Pipeline · Built with LangGraph, OpenAI, Chroma, and Next.js
        </p>
      </footer>
    </main>
  );
}
