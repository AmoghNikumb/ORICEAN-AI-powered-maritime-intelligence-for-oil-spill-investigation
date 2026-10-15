import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { IncidentMap } from "@/components/oricean/OfficerPages";

export const Route = createFileRoute("/officer/incidents")({
  head: () => ({
    meta: [
      { title: "Incident Map — ORICEAN" },
      { name: "description", content: "Unified view of alerts, response units, and vessel tracks for command officers." },
      { property: "og:title", content: "Incident Map — ORICEAN" },
      { property: "og:description", content: "Unified view of alerts, response units, and vessel tracks for command officers." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: gated("officer", IncidentMap),
});
