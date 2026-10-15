import { useState } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { z } from "zod";
import { Globe2, Radar, Satellite, Waves } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSession, demoAccounts, roleHome } from "@/lib/session";
import type { Role } from "@/components/oricean/AppShell";

export const Route = createFileRoute("/signin")({
  validateSearch: z.object({ redirect: z.string().optional() }),
  head: () => ({
    meta: [
      { title: "Sign In — ORICEAN Secure Access" },
      { name: "description", content: "Sign in to the ORICEAN analyst or monitoring officer workspace, or continue to the public spill explorer." },
      { property: "og:title", content: "Sign In — ORICEAN Secure Access" },
      { property: "og:description", content: "Role-based access to ORICEAN oil spill intelligence workspaces." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: SignInPage,
});

const roleCards: { role: Role; label: string; desc: string; icon: typeof Waves }[] = [
  { role: "analyst", label: "Analyst", desc: "Satellite analysis, source reconstruction, vessel attribution", icon: Satellite },
  { role: "officer", label: "Monitoring Officer", desc: "Command dashboard, alerts, case review, response", icon: Radar },
];

function SignInPage() {
  const { redirect } = Route.useSearch();
  const navigate = useNavigate();
  const { signIn, session, signOut } = useSession();
  const [role, setRole] = useState<Role>("analyst");
  const [email, setEmail] = useState(demoAccounts[0].email);
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const account = demoAccounts.find((a) => a.role === role);
    if (!account || email.trim().toLowerCase() !== account.email || password !== account.password) {
      setError("Invalid credentials for this clearance level.");
      return;
    }
    setError("");
    signIn({ name: account.name, role: account.role, agency: account.agency });
    navigate({ to: redirect && redirect.startsWith("/") ? redirect : roleHome[account.role] });
  }

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="hidden flex-col justify-between bg-sidebar p-10 text-sidebar-foreground lg:flex">
        <Link to="/" className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-sm bg-sidebar-primary text-sidebar-primary-foreground">
            <Waves className="h-5 w-5" />
          </span>
          <span className="font-display text-xl font-semibold tracking-[0.18em]">ORICEAN</span>
        </Link>
        <div className="max-w-sm">
          <h2 className="font-display text-3xl leading-tight font-semibold tracking-wide">
            Restricted maritime intelligence
          </h2>
          <p className="mt-3 text-sm leading-6 text-sidebar-foreground/70">
            Vessel identities, attribution evidence and response operations are limited to authorised
            personnel. Public users can explore released detections without an account.
          </p>
        </div>
        <p className="text-[11px] tracking-[0.14em] text-sidebar-foreground/45 uppercase">
          Access is logged · Clearance enforced per role
        </p>
      </div>

      <div className="flex items-center justify-center bg-background p-6">
        <div className="w-full max-w-md">
          <h1 className="font-display text-2xl font-semibold tracking-wide">Secure sign in</h1>
          <p className="mt-1 text-sm text-muted-foreground">Select your clearance level to continue.</p>

          {session && (
            <div className="mt-4 flex items-center justify-between border border-border bg-muted/40 p-3 text-xs">
              <span>
                Signed in as <span className="font-semibold">{session.name}</span> ({session.role})
              </span>
              <Button size="sm" variant="outline" onClick={signOut}>Sign out</Button>
            </div>
          )}

          <div className="mt-5 grid gap-2">
            {roleCards.map((c) => (
              <button
                key={c.role}
                type="button"
                onClick={() => {
                  setRole(c.role);
                  setEmail(demoAccounts.find((a) => a.role === c.role)!.email);
                  setPassword("");
                  setError("");
                }}
                className={`flex items-start gap-3 border p-3 text-left transition-colors ${
                  role === c.role ? "border-satellite bg-satellite/10" : "border-border hover:bg-muted/50"
                }`}
              >
                <c.icon className="mt-0.5 h-4 w-4 shrink-0 text-satellite" />
                <span>
                  <span className="block text-sm font-semibold">{c.label}</span>
                  <span className="block text-xs text-muted-foreground">{c.desc}</span>
                </span>
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="mt-5 space-y-3">
            <label className="block text-xs">
              <span className="label-caps text-muted-foreground">Work email</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-1 h-9 w-full border border-input bg-background px-2.5 text-sm outline-none focus:ring-1 focus:ring-ring"
              />
            </label>
            <label className="block text-xs">
              <span className="label-caps text-muted-foreground">Passphrase</span>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 h-9 w-full border border-input bg-background px-2.5 text-sm outline-none focus:ring-1 focus:ring-ring"
              />
            </label>
            {error && <p className="text-xs font-medium text-critical">{error}</p>}
            <Button type="submit" className="w-full">Sign in</Button>
          </form>

          <p className="mono-num mt-3 text-[11px] leading-5 text-muted-foreground">
            Demo credentials — analyst@oricean.org / analyst · officer@oricean.org / officer
          </p>

          <div className="mt-5 border-t border-border pt-4">
            <Button asChild variant="outline" size="sm" className="w-full">
              <Link to="/explore"><Globe2 /> Continue as public user</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
