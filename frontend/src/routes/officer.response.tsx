import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { ResponseStatus } from "@/components/oricean/OfficerPages";

export const Route = createFileRoute("/officer/response")({
  head: () => ({
    meta: [
      { title: "Response Status — ORICEAN" },
      { name: "description", content: "Track field resources, containment operations, and response unit readiness." },
      { property: "og:title", content: "Response Status — ORICEAN" },
      { property: "og:description", content: "Track field resources, containment operations, and response unit readiness." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: gated("officer", ResponseStatus),
});
