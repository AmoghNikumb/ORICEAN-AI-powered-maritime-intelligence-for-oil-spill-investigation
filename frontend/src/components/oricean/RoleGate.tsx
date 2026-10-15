import type { ComponentType } from "react";
import { Link } from "@tanstack/react-router";
import { Lock, ShieldAlert, Waves } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSession, roleHome } from "@/lib/session";
import type { Role } from "@/components/oricean/AppShell";

const roleLabel: Record<Role, string> = {
  analyst: "Analyst",
  officer: "Monitoring Officer",
  public: "Public",
};

function Denied({ required, current }: { required: Role; current?: Role }) {
  const signedIn = Boolean(current);
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-md border border-border bg-card p-8 text-center">
        <span className="mx-auto flex h-11 w-11 items-center justify-center rounded-sm bg-secondary text-secondary-foreground">
          {signedIn ? <ShieldAlert className="h-5 w-5" /> : <Lock className="h-5 w-5" />}
        </span>
        <h1 className="mt-4 font-display text-xl font-semibold tracking-wide">
          {signedIn ? "Insufficient clearance" : "Restricted workspace"}
        </h1>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          {signedIn
            ? `This area requires ${roleLabel[required]} clearance. You are signed in as ${roleLabel[current!]}.`
            : `The ${roleLabel[required]} workspace holds restricted vessel and attribution evidence. Sign in to continue.`}
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <Button asChild size="sm">
            <Link to="/signin" search={{ redirect: roleHome[required] }}>
              {signedIn ? "Switch account" : "Sign in"}
            </Link>
          </Button>
          <Button asChild size="sm" variant="outline">
            <Link to={current ? roleHome[current] : "/explore"}>
              <Waves /> {current && current !== "public" ? "My workspace" : "Public explorer"}
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

export function RoleGate({ role, children }: { role: Role; children: React.ReactNode }) {
  const { session, ready } = useSession();
  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <p className="label-caps text-muted-foreground">Verifying clearance…</p>
      </div>
    );
  }
  if (role !== "public" && session?.role !== role) {
    return <Denied required={role} current={session?.role} />;
  }
  return <>{children}</>;
}

export function gated<P extends object>(role: Role, Component: ComponentType<P>) {
  return function GatedRoute(props: P) {
    return (
      <RoleGate role={role}>
        <Component {...props} />
      </RoleGate>
    );
  };
}
