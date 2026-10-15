import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { Dashboard } from "@/components/oricean/AnalystPages";

export const Route = createFileRoute("/analyst/")({
  head: () => ({
    meta: [
      { title: "Analyst Dashboard — ORICEAN" },
      { name: "description", content: "Monitor oil spill investigations, detections, and analyst priorities in ORICEAN." },
      { property: "og:title", content: "Analyst Dashboard — ORICEAN" },
      { property: "og:description", content: "Monitor oil spill investigations, detections, and analyst priorities in ORICEAN." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: gated("analyst", Dashboard),
});