import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { OfficerReports } from "@/components/oricean/OfficerPages";

export const Route = createFileRoute("/officer/reports")({
  head: () => ({
    meta: [
      { title: "Officer Reports — ORICEAN" },
      { name: "description", content: "Review, approve, and distribute investigation handoff packages." },
      { property: "og:title", content: "Officer Reports — ORICEAN" },
      { property: "og:description", content: "Review, approve, and distribute investigation handoff packages." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: gated("officer", OfficerReports),
});
