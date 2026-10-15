import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { Severity, SpillStatus } from "@/lib/oricean-data";

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
  icon,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "default" | "critical" | "warning" | "satellite" | "resolved";
  icon?: ReactNode;
}) {
  const bar = {
    default: "bg-ocean",
    critical: "bg-critical",
    warning: "bg-warning",
    satellite: "bg-satellite",
    resolved: "bg-resolved",
  }[tone];
  return (
    <div className="relative overflow-hidden rounded-md border border-border bg-card p-4 shadow-panel">
      <span className={cn("absolute inset-y-0 left-0 w-[3px]", bar)} />
      <div className="flex items-start justify-between gap-2">
        <p className="label-caps">{label}</p>
        {icon && <span className="text-muted-foreground">{icon}</span>}
      </div>
      <p className="mono-num mt-2 text-3xl leading-none font-semibold">{value}</p>
      {hint && <p className="mt-1.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

const sevTone: Record<Severity, string> = {
  critical: "bg-critical/12 text-critical border-critical/30",
  high: "bg-warning/12 text-warning border-warning/30",
  medium: "bg-caution/15 text-caution border-caution/40",
  low: "bg-resolved/12 text-resolved border-resolved/30",
};

const statusTone: Record<string, string> = {
  confirmed: "bg-critical/12 text-critical border-critical/30",
  investigating: "bg-warning/12 text-warning border-warning/30",
  potential: "bg-caution/15 text-caution border-caution/40",
  resolved: "bg-resolved/12 text-resolved border-resolved/30",
  Active: "bg-warning/12 text-warning border-warning/30",
  New: "bg-satellite/15 text-ocean border-satellite/40",
  Investigating: "bg-warning/12 text-warning border-warning/30",
  "Response Active": "bg-critical/12 text-critical border-critical/30",
  Resolved: "bg-resolved/12 text-resolved border-resolved/30",
  "Awaiting Review": "bg-satellite/15 text-ocean border-satellite/40",
  Submitted: "bg-secondary text-secondary-foreground border-border",
  Closed: "bg-secondary text-muted-foreground border-border",
};

export function Pill({
  children,
  tone,
  severity,
  status,
  className,
}: {
  children: ReactNode;
  tone?: string;
  severity?: Severity;
  status?: SpillStatus | string;
  className?: string;
}) {
  const cls =
    tone ??
    (severity ? sevTone[severity] : undefined) ??
    (status ? statusTone[status] : undefined) ??
    "bg-secondary text-secondary-foreground border-border";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-sm border px-2 py-0.5 text-[10px] font-semibold tracking-[0.1em] uppercase",
        cls,
        className,
      )}
    >
      {children}
    </span>
  );
}

export function ScoreBar({
  label,
  value,
  tone = "ocean",
}: {
  label: string;
  value: number;
  tone?: "ocean" | "satellite" | "spill" | "warning";
}) {
  const bg = { ocean: "bg-ocean", satellite: "bg-satellite", spill: "bg-spill", warning: "bg-warning" }[tone];
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="mono-num font-semibold">{value}%</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full transition-all", bg)} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

export function Panel({
  title,
  right,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cn("rounded-md border border-border bg-card shadow-panel", className)}>
      {title && (
        <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
          <h2 className="label-caps text-foreground">{title}</h2>
          {right}
        </header>
      )}
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-border py-1.5 last:border-0">
      <span className="text-xs text-muted-foreground">{k}</span>
      <span className="mono-num text-right text-xs font-medium">{v}</span>
    </div>
  );
}
