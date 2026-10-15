import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { EvidenceTimeline } from "@/components/oricean/AnalystPages";

export const Route = createFileRoute("/analyst/timeline")({
  head: () => ({ meta: [{ title: "Evidence Timeline — ORICEAN" }, { name: "description", content: "Trace satellite, model, vessel, and report evidence through an oil spill investigation timeline." }, { property: "og:title", content: "Evidence Timeline — ORICEAN" }, { property: "og:description", content: "Trace satellite, model, vessel, and report evidence through an oil spill investigation timeline." }, { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" }] }),
  component: gated("analyst", EvidenceTimeline),
});