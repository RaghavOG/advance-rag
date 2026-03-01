import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0f1117",
          card: "#161b27",
          elevated: "#1e2535",
          border: "#252d3d",
        },
        brand: {
          DEFAULT: "#6366f1",
          muted: "#4f46e5",
          light: "#818cf8",
        },
        success: { DEFAULT: "#10b981", muted: "#065f46", light: "#6ee7b7" },
        warning: { DEFAULT: "#f59e0b", muted: "#78350f", light: "#fcd34d" },
        danger:  { DEFAULT: "#f43f5e", muted: "#881337", light: "#fda4af" },
        neutral: { 50: "#f8fafc", 400: "#94a3b8", 500: "#64748b", 700: "#334155", 900: "#0f172a" },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      animation: {
        "progress-loop": "progressLoop 2s ease-in-out infinite",
        "pulse-soft": "pulseSoft 2s ease-in-out infinite",
        "fade-up": "fadeUp 0.25s ease-out forwards",
      },
      keyframes: {
        progressLoop: {
          "0%":   { width: "20%", marginLeft: "0%" },
          "50%":  { width: "40%", marginLeft: "30%" },
          "100%": { width: "20%", marginLeft: "80%" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0.6" },
        },
        fadeUp: {
          from: { opacity: "0", transform: "translateY(8px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
