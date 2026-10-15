import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Clock,
  Compass,
  FileText,
  Globe2,
  Layers,
  LayoutDashboard,
  Lock,
  LogOut,
  Map as MapIcon,
  Radar,
  Radio,
  Satellite,
  Ship,
  Waves,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSession, roleHome } from "@/lib/session";

function AccountBlock() {
  const { session, signOut } = useSession();

  if (!session) {
    return (
      <div>
        <p className="label-caps px-1 pb-1.5 text-sidebar-foreground/45">Public access</p>
        <Link
          to="/signin"
          className="flex items-center justify-center gap-2 rounded-sm bg-sidebar-primary px-2 py-2 text-[11px] font-semibold tracking-wider text-sidebar-primary-foreground uppercase transition-opacity hover:opacity-90"
        >
          <Lock className="h-3.5 w-3.5" /> Secure sign in
        </Link>
      </div>
    );
  }

  return (
    <div>
      <p className="label-caps px-1 pb-1.5 text-sidebar-foreground/45">Signed in</p>
      <div className="rounded-sm bg-sidebar-accent/60 px-2.5 py-2">
        <p className="truncate text-xs font-semibold">{session.name}</p>
        <p className="truncate text-[10px] text-sidebar-foreground/60">{session.agency}</p>
      </div>
      <div className="mt-1 grid grid-cols-2 gap-1">
        <Link
          to={roleHome[session.role]}
          className="rounded-sm bg-sidebar-accent/60 px-1.5 py-1.5 text-center text-[10px] font-semibold tracking-wider uppercase hover:bg-sidebar-accent"
        >
          Workspace
        </Link>
        <button
          onClick={signOut}
          className="flex items-center justify-center gap-1 rounded-sm bg-sidebar-accent/60 px-1.5 py-1.5 text-[10px] font-semibold tracking-wider uppercase hover:bg-sidebar-accent"
        >
          <LogOut className="h-3 w-3" /> Sign out
        </button>
      </div>
    </div>
  );
}

export type Role = "analyst" | "officer" | "public";

const navs: Record<Role, { to: string; label: string; icon: typeof MapIcon }[]> = {
  analyst: [
    { to: "/analyst", label: "Dashboard", icon: LayoutDashboard },
    { to: "/analyst/new", label: "New Investigation", icon: Compass },
    { to: "/analyst/satellite", label: "Satellite Analysis", icon: Satellite },
    { to: "/analyst/source", label: "Source Reconstruction", icon: Waves },
    { to: "/analyst/vessels", label: "Vessel Attribution", icon: Ship },
    { to: "/analyst/timeline", label: "Evidence Timeline", icon: Clock },
    { to: "/analyst/reports", label: "Reports", icon: FileText },
  ],
  officer: [
    { to: "/officer", label: "Command Dashboard", icon: Radar },
    { to: "/officer/alerts", label: "Active Alerts", icon: AlertTriangle },
    { to: "/officer/incidents", label: "Incident Map", icon: MapIcon },
    { to: "/officer/case", label: "Case Review", icon: Layers },
    { to: "/officer/response", label: "Response Status", icon: Radio },
    { to: "/officer/reports", label: "Reports", icon: FileText },
  ],
  public: [
    { to: "/explore", label: "Global Spill Map", icon: Globe2 },
    { to: "/explore/live", label: "Live Incidents", icon: Activity },
    { to: "/explore/historical", label: "Historical Spills", icon: Clock },
    { to: "/explore/insights", label: "Global Insights", icon: BarChart3 },
  ],
};

const roleMeta: Record<Role, { label: string; sub: string }> = {
  analyst: { label: "Analyst", sub: "Investigation Workspace" },
  officer: { label: "Monitoring Officer", sub: "Command & Response" },
  public: { label: "Public Access", sub: "Global Spill Explorer" },
};

export function AppShell({
  role,
  title,
  subtitle,
  actions,
  children,
}: {
  role: Role;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const nav = navs[role];

  return (
    <div className="flex min-h-screen bg-background">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-sidebar text-sidebar-foreground lg:flex">
        <Link to="/" className="flex items-center gap-2.5 border-b border-sidebar-border px-5 py-4">
          <span className="flex h-8 w-8 items-center justify-center rounded-sm bg-sidebar-primary text-sidebar-primary-foreground">
            <Waves className="h-4.5 w-4.5" />
          </span>
          <span>
            <span className="block font-display text-lg leading-none font-semibold tracking-[0.18em]">
              ORICEAN
            </span>
            <span className="block text-[10px] tracking-[0.14em] text-sidebar-foreground/60">
              OIL SPILL INTELLIGENCE
            </span>
          </span>
        </Link>

        <div className="px-5 pt-5 pb-2">
          <p className="label-caps text-sidebar-foreground/50">{roleMeta[role].label}</p>
          <p className="text-xs text-sidebar-foreground/70">{roleMeta[role].sub}</p>
        </div>

        <nav className="flex-1 space-y-0.5 px-3 py-2">
          {nav.map((item) => {
            const active = pathname === item.to || (item.to !== "/explore" && pathname.startsWith(item.to + "/"));
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "flex items-center gap-2.5 rounded-sm px-2.5 py-2 text-sm transition-colors",
                  active
                    ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-[inset_2px_0_0_0_var(--sidebar-primary)]"
                    : "text-sidebar-foreground/75 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
                )}
              >
                <item.icon className="h-4 w-4 shrink-0" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-sidebar-border p-3">
          <AccountBlock />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 border-b border-border bg-card/90 backdrop-blur">
          <div className="flex flex-wrap items-center gap-3 px-4 py-3 md:px-6">
            <div className="min-w-0 flex-1">
              <h1 className="truncate font-display text-xl font-semibold tracking-wide">{title}</h1>
              {subtitle && <p className="truncate text-xs text-muted-foreground">{subtitle}</p>}
            </div>
            <div className="flex items-center gap-2">{actions}</div>
          </div>
          <div className="flex gap-1 overflow-x-auto border-t border-border px-3 py-1.5 lg:hidden">
            {nav.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className="rounded-sm px-2.5 py-1 text-xs whitespace-nowrap text-muted-foreground hover:bg-muted"
                activeProps={{ className: "bg-secondary text-secondary-foreground" }}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </header>

        <main className="flex-1 p-4 md:p-6">{children}</main>
      </div>
    </div>
  );
}
