import { createFileRoute, Navigate } from "@tanstack/react-router";
import { useSession, roleHome } from "@/lib/session";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ORICEAN — AI-Powered Oil Spill Intelligence" },
      { name: "description", content: "Satellite detection, source reconstruction and vessel attribution for marine oil spills, with role-based analyst, officer and public access." },
      { property: "og:title", content: "ORICEAN — AI-Powered Oil Spill Intelligence" },
      { property: "og:description", content: "Satellite detection, source reconstruction and vessel attribution for marine oil spills." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: HomeRedirect,
});

function HomeRedirect() {
  const { session, ready } = useSession();
  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <p className="label-caps text-muted-foreground">Loading ORICEAN…</p>
      </div>
    );
  }
  return <Navigate to={session ? roleHome[session.role] : "/explore"} replace />;
}
