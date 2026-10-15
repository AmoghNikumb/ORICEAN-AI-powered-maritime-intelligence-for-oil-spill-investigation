import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { ActiveAlerts } from "@/components/oricean/OfficerPages";

export const Route = createFileRoute("/officer/alerts")({
  head: () => ({
    meta: [
      { title: "Active Alerts — ORICEAN" },
      { name: "description", content: "Triage, assign, and monitor incoming marine pollution alerts." },
      { property: "og:title", content: "Active Alerts — ORICEAN" },
      { property: "og:description", content: "Triage, assign, and monitor incoming marine pollution alerts." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: gated("officer", ActiveAlerts),
});
