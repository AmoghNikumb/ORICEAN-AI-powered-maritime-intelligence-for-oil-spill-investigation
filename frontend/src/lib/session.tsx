import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Role } from "@/components/oricean/AppShell";

export type Session = { name: string; role: Role; agency: string };

const KEY = "oricean.session";

type Ctx = {
  session: Session | null;
  ready: boolean;
  signIn: (s: Session) => void;
  signOut: () => void;
  hasRole: (role: Role) => boolean;
};

const SessionContext = createContext<Ctx | null>(null);

export const roleHome: Record<Role, string> = {
  analyst: "/analyst",
  officer: "/officer",
  public: "/explore",
};

export const demoAccounts: { role: Role; name: string; agency: string; email: string; password: string }[] = [
  { role: "analyst", name: "Dr. A. Rahman", agency: "ORICEAN Analysis Cell", email: "analyst@oricean.org", password: "analyst" },
  { role: "officer", name: "Cdr. L. Marchetti", agency: "Maritime Command Centre", email: "officer@oricean.org", password: "officer" },
];

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) setSession(JSON.parse(raw) as Session);
    } catch {
      /* ignore malformed session */
    }
    setReady(true);
  }, []);

  const signIn = useCallback((s: Session) => {
    setSession(s);
    try {
      localStorage.setItem(KEY, JSON.stringify(s));
    } catch {
      /* storage unavailable */
    }
  }, []);

  const signOut = useCallback(() => {
    setSession(null);
    try {
      localStorage.removeItem(KEY);
    } catch {
      /* storage unavailable */
    }
  }, []);

  const value = useMemo<Ctx>(
    () => ({
      session,
      ready,
      signIn,
      signOut,
      hasRole: (role: Role) => (role === "public" ? true : session?.role === role),
    }),
    [session, ready, signIn, signOut],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used inside SessionProvider");
  return ctx;
}
