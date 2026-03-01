import { cn } from "@/lib/utils";

type Variant = "default" | "success" | "warning" | "danger" | "muted";

const styles: Record<Variant, string> = {
  default: "bg-brand/15 text-brand-light border-brand/30",
  success: "bg-emerald-950/60 text-emerald-400 border-emerald-800",
  warning: "bg-amber-950/60 text-amber-400 border-amber-800",
  danger:  "bg-rose-950/60 text-rose-400 border-rose-800",
  muted:   "bg-surface-elevated text-slate-400 border-surface-border",
};

export function Badge({
  children,
  variant = "default",
  className,
}: {
  children: React.ReactNode;
  variant?: Variant;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
        styles[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}
