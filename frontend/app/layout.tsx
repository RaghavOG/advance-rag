import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Multimodal RAG — Production Pipeline",
  description:
    "Text-first RAG pipeline with LangGraph orchestration, HyDE, multi-query rewriting, and grounded citations.",
  keywords: ["RAG", "LangGraph", "OpenAI", "vector store", "multimodal"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#0f1117]">{children}</body>
    </html>
  );
}
