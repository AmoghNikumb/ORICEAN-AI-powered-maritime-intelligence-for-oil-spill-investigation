import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { NewInvestigation } from "@/components/oricean/AnalystPages";

export const Route = createFileRoute("/analyst/new")({
  head: () => ({ meta: [{ title: "New Investigation — ORICEAN" }, { name: "description", content: "Start a structured ORICEAN oil spill investigation." }, { property: "og:title", content: "New Investigation — ORICEAN" }, { property: "og:description", content: "Start a structured ORICEAN oil spill investigation." }, { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" }] }),
  component: gated("analyst", NewInvestigation),
});