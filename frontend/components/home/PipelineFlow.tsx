"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface FlowNode {
  id: string;
  label: string;
  sublabel?: string;
  kind: "input" | "decision" | "llm" | "tool" | "failure" | "output";
  layer?: string;
}

interface FlowEdge {
  label?: string;
  color?: string;
}

const layers: { title: string; color: string; nodes: FlowNode[]; edge?: FlowEdge }[] = [
  {
    title: "Layer 0 — Entry",
    color: "border-slate-600",
    nodes: [
      { id: "normalize", label: "normalize_user_prompt", sublabel: "trim · normalize", kind: "tool" },
    ],
    edge: {},
  },
  {
    title: "Layer 1 — Multi-Query",
    color: "border-indigo-500",
    nodes: [
      { id: "detect", label: "detect_multi_query", sublabel: "split on ? · numbered lists", kind: "decision" },
    ],
    edge: { label: "single query or loop per sub-query" },
  },
  {
    title: "Layer 2 — Per-Query Sub-graph",
    color: "border-purple-500",
    nodes: [
      { id: "ambiguity", label: "ambiguity_check", sublabel: "LLM (cheap)", kind: "llm" },
      { id: "clarify", label: "clarification_node", sublabel: "→ EXIT (single turn)", kind: "input" },
      { id: "rewrite", label: "query_rewrite_expand", sublabel: "2–4 retrieval rewrites", kind: "llm" },
      { id: "topk", label: "adaptive_top_k_decision", sublabel: "heuristic", kind: "tool" },
      { id: "retrieve", label: "retrieve_documents", sublabel: "vector + HyDE", kind: "tool" },
      { id: "merge", label: "merge_retrieval_results", sublabel: "dedup · score sort", kind: "tool" },
    ],
    edge: {},
  },
  {
    title: "Layer 3 — Failure-Aware Core",
    color: "border-rose-500",
    nodes: [
      { id: "ret_fail", label: "retrieval_failure_node", sublabel: "no docs / low confidence", kind: "failure" },
      { id: "compress", label: "compress_context_node", sublabel: "LLM summarize", kind: "llm" },
      { id: "comp_fail", label: "compression_failure_node", sublabel: "extractive fallback", kind: "failure" },
      { id: "generate", label: "generate_answer_node", sublabel: "grounded · cited", kind: "llm" },
      { id: "llm_fail", label: "llm_timeout_failure_node", sublabel: "retry exhausted", kind: "failure" },
    ],
    edge: {},
  },
  {
    title: "Layer 4 — Multi-Query Merge",
    color: "border-emerald-500",
    nodes: [
      { id: "collect", label: "collect_sub_answers", sublabel: "accumulator", kind: "tool" },
      { id: "final", label: "merge_final_answers", sublabel: "structured output", kind: "output" },
    ],
    edge: {},
  },
];

const kindStyle: Record<string, string> = {
  input:    "border-slate-500  bg-slate-800/60  text-slate-200",
  decision: "border-indigo-500 bg-indigo-950/60 text-indigo-200",
  llm:      "border-purple-500 bg-purple-950/60 text-purple-200",
  tool:     "border-cyan-600   bg-cyan-950/60   text-cyan-200",
  failure:  "border-rose-500   bg-rose-950/40   text-rose-300",
  output:   "border-emerald-500 bg-emerald-950/50 text-emerald-200",
};

export function PipelineFlow() {
  return (
    <section id="pipeline" className="px-6 py-20 max-w-5xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.4 }}
      >
        <h2 className="text-3xl font-bold text-slate-100 mb-2 text-center">
          LangGraph Pipeline
        </h2>
        <p className="text-slate-400 text-center mb-12 max-w-xl mx-auto">
          15 nodes organized across 4 layers. Failures are nodes — not exceptions.
          Every routing decision is explicit.
        </p>
      </motion.div>

      <div className="space-y-1">
        {layers.map((layer, li) => (
          <motion.div
            key={layer.title}
            initial={{ opacity: 0, x: -10 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: li * 0.08, duration: 0.35 }}
          >
            {/* Layer header */}
            <div className={cn("border-l-2 pl-4 mb-3 mt-6", layer.color)}>
              <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                {layer.title}
              </span>
            </div>

            {/* Nodes */}
            <div className="flex flex-wrap gap-2 pl-4">
              {layer.nodes.map((node) => (
                <div
                  key={node.id}
                  className={cn(
                    "border rounded-lg px-3 py-2 text-xs font-mono min-w-[180px]",
                    kindStyle[node.kind],
                  )}
                >
                  <p className="font-semibold">{node.label}</p>
                  {node.sublabel && (
                    <p className="mt-0.5 opacity-60 text-[10px]">{node.sublabel}</p>
                  )}
                </div>
              ))}
            </div>

            {/* Connector arrow */}
            {li < layers.length - 1 && (
              <div className="pl-4 mt-3 flex items-center gap-2 text-slate-600 text-xs">
                <div className="h-6 border-l border-dashed border-slate-700 ml-2" />
                {layer.edge?.label && (
                  <span className="text-slate-600 ml-1">{layer.edge.label}</span>
                )}
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Legend */}
      <div className="mt-12 flex flex-wrap gap-3 justify-center">
        {Object.entries(kindStyle).map(([kind, cls]) => (
          <span
            key={kind}
            className={cn("inline-flex items-center gap-1.5 border rounded px-2.5 py-1 text-xs", cls)}
          >
            <span className="capitalize">{kind}</span>
          </span>
        ))}
      </div>
    </section>
  );
}
