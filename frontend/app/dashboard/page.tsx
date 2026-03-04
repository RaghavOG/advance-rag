"use client";

import { useEffect, useState } from "react";
import { cn, formatMs } from "@/lib/utils";

interface IngestedDoc {
  doc_id: string;
  chunks: number;
  filename?: string | null;
  path?: string | null;
}

interface IngestionStats {
  total_documents: number;
  total_chunks: number;
  index_size: number;
  embedding_model: string;
  vector_store_backend: string;
}

type UploadStage =
  | "idle"
  | "selecting"
  | "uploading"
  | "ingesting"
  | "done"
  | "error";

export default function DashboardPage() {
  const [docs, setDocs] = useState<IngestedDoc[]>([]);
  const [stats, setStats] = useState<IngestionStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [uploadStage, setUploadStage] = useState<UploadStage>("idle");
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/admin/ingestion/docs");
      if (!res.ok) throw new Error(`Failed to load docs (${res.status})`);
      const json = await res.json();
      setDocs(json.documents ?? []);
      setStats(json.stats ?? null);
    } catch (e: any) {
      setError(e.message ?? "Failed to load ingestion data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  const handleDelete = async (docId: string) => {
    if (!confirm(`Delete all chunks for document ${docId}?`)) return;
    setError(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/admin/ingestion/docs/${docId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const json = await res.json().catch(() => null);
        throw new Error(json?.detail ?? `Delete failed (${res.status})`);
      }
      await loadData();
    } catch (e: any) {
      setError(e.message ?? "Delete failed");
    }
  };

  const handleUpload: React.ChangeEventHandler<HTMLInputElement> = async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploadStage("uploading");
    setUploadMessage("Uploading file…");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("http://127.0.0.1:8000/api/upload", {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const json = await res.json().catch(() => null);
        throw new Error(json?.detail ?? `Upload failed (${res.status})`);
      }
      setUploadStage("ingesting");
      setUploadMessage("Running ingestion pipeline (load → chunk → embed)…");
      const json = await res.json();
      setUploadStage("done");
      setUploadMessage(`Ingested ${json.chunks_ingested} chunks from ${json.filename}.`);
      await loadData();
    } catch (e: any) {
      setUploadStage("error");
      setUploadMessage(e.message ?? "Upload failed");
    } finally {
      // Reset file input so the same file can be selected again.
      ev.target.value = "";
      setTimeout(() => {
        setUploadStage("idle");
      }, 4000);
    }
  };

  return (
    <main className="min-h-screen bg-surface text-slate-100">
      <div className="max-w-6xl mx-auto px-6 py-8">
        <header className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold">Ingestion Dashboard</h1>
            <p className="text-xs text-slate-500 mt-1">
              Inspect and manage ingested documents, chunks, and upload pipeline.
            </p>
          </div>
          <a
            href="/"
            className="text-xs text-slate-400 hover:text-slate-200 transition-colors underline-offset-4 hover:underline"
          >
            ← Back to home
          </a>
        </header>

        {/* Stats row */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          <div className="rounded-lg border border-surface-border bg-surface-card px-4 py-3">
            <p className="text-xs text-slate-500">Total documents</p>
            <p className="mt-1 text-xl font-semibold">{stats?.total_documents ?? "—"}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card px-4 py-3">
            <p className="text-xs text-slate-500">Total chunks</p>
            <p className="mt-1 text-xl font-semibold">{stats?.total_chunks ?? "—"}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card px-4 py-3">
            <p className="text-xs text-slate-500">Embedding model</p>
            <p className="mt-1 text-xs text-slate-300">{stats?.embedding_model ?? "—"}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card px-4 py-3">
            <p className="text-xs text-slate-500">Vector store</p>
            <p className="mt-1 text-xs text-slate-300">{stats?.vector_store_backend ?? "—"}</p>
          </div>
        </section>

        {/* Upload panel */}
        <section className="mb-10 rounded-xl border border-surface-border bg-surface-card px-5 py-4">
          <h2 className="text-sm font-semibold mb-2">Upload &amp; Ingest Document</h2>
          <p className="text-xs text-slate-500 mb-3">
            Upload a PDF, TXT, or Markdown file. The dashboard shows the high-level steps from upload
            to embeddings written into the index.
          </p>
          <div className="flex flex-col sm:flex-row sm:items-center gap-3">
            <label className="inline-flex items-center gap-2 rounded-lg border border-dashed border-surface-border bg-surface-elevated px-4 py-2 text-xs cursor-pointer hover:border-brand/60 hover:bg-surface-elevated/80 transition-colors">
              <span className="font-medium text-slate-200">Choose file</span>
              <input type="file" accept=".pdf,.txt,.md,.markdown" className="hidden" onChange={handleUpload} />
            </label>
            <div className="flex-1 text-[11px] text-slate-500">
              {uploadStage === "idle" && <span>No upload in progress.</span>}
              {uploadStage !== "idle" && (
                <span className={cn(uploadStage === "error" ? "text-red-400" : "text-slate-400")}>
                  {uploadMessage}
                </span>
              )}
            </div>
          </div>
          {/* Simple step indicator */}
          <div className="mt-3 flex flex-wrap gap-2 text-[10px] text-slate-500">
            {["Upload", "Load", "Chunk", "Embed & Write"].map((label, idx) => {
              const active =
                uploadStage === "uploading" ? idx === 0 :
                uploadStage === "ingesting" ? idx >= 1 && idx <= 3 :
                uploadStage === "done" ? false :
                false;
              return (
                <span
                  key={label}
                  className={cn(
                    "inline-flex items-center gap-1 rounded-full border px-2 py-0.5",
                    active ? "border-brand/80 text-brand-light bg-brand/10" : "border-surface-border",
                  )}
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-current" />
                  {label}
                </span>
              );
            })}
          </div>
        </section>

        {/* Error banner */}
        {error && (
          <div className="mb-4 rounded-md border border-red-500/40 bg-red-500/10 px-4 py-2 text-xs text-red-200">
            {error}
          </div>
        )}

        {/* Dangerous actions */}
        <section className="mb-6 rounded-xl border border-red-500/40 bg-red-500/5 px-5 py-4">
          <h2 className="text-sm font-semibold mb-2 text-red-200">Danger zone</h2>
          <p className="text-[11px] text-red-300 mb-3">
            Clear all embeddings and conversation history. This cannot be undone.
          </p>
          <button
            onClick={async () => {
              if (!confirm("Clear ALL embeddings and conversation history? This cannot be undone.")) {
                return;
              }
              setError(null);
              try {
                const res = await fetch("http://127.0.0.1:8000/api/admin/ingestion/clear_all", {
                  method: "POST",
                });
                if (!res.ok) {
                  const json = await res.json().catch(() => null);
                  throw new Error(json?.detail ?? `Clear all failed (${res.status})`);
                }
                await loadData();
              } catch (e: any) {
                setError(e.message ?? "Clear all failed");
              }
            }}
            className="inline-flex items-center gap-1 rounded border border-red-500/70 bg-red-500/10 px-3 py-1.5 text-[11px] font-medium text-red-100 hover:bg-red-500/20 transition-colors"
          >
            Clear ALL embeddings & conversations
          </button>
        </section>

        {/* Documents table */}
        <section className="rounded-xl border border-surface-border bg-surface-card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-surface-border">
            <h2 className="text-sm font-semibold">Ingested Documents</h2>
            <button
              onClick={() => loadData()}
              className="text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
            >
              Refresh
            </button>
          </div>
          <div className="max-h-[420px] overflow-y-auto text-xs">
            {loading && docs.length === 0 && (
              <div className="px-4 py-6 text-center text-slate-500">Loading…</div>
            )}
            {!loading && docs.length === 0 && (
              <div className="px-4 py-6 text-center text-slate-500">No documents ingested yet.</div>
            )}
            {docs.length > 0 && (
              <table className="w-full border-t border-surface-border/60">
                <thead className="bg-surface-elevated sticky top-0 z-10">
                  <tr className="text-[11px] text-slate-500 text-left">
                    <th className="px-4 py-2 font-medium">doc_id</th>
                    <th className="px-2 py-2 font-medium">Filename</th>
                    <th className="px-2 py-2 font-medium text-right">Chunks</th>
                    <th className="px-2 py-2 font-medium w-28 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {docs.map((d) => (
                    <tr key={d.doc_id} className="border-t border-surface-border/40">
                      <td className="px-4 py-2 font-mono text-[11px] text-slate-300">{d.doc_id}</td>
                      <td className="px-2 py-2 text-[11px] text-slate-400">
                        {d.filename ?? d.path ?? "—"}
                      </td>
                      <td className="px-2 py-2 text-right">{d.chunks}</td>
                      <td className="px-2 py-2 text-right space-x-1">
                        {/* Delete only for now; chunk-level view can be added later using /chunks endpoint */}
                        <button
                          onClick={() => handleDelete(d.doc_id)}
                          className="inline-flex items-center justify-center rounded border border-red-500/70 px-2 py-0.5 text-[10px] text-red-300 hover:bg-red-500/10 transition-colors"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
