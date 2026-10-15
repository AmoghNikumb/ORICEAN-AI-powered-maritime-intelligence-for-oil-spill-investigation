import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { Reports } from "@/components/oricean/AnalystPages";

export const Route = createFileRoute("/analyst/reports")({
  head: () => ({ meta: [{ title: "Analyst Reports — ORICEAN" }, { name: "description", content: "Assemble evidence packages and submit oil spill attribution reports for review." }, { property: "og:title", content: "Analyst Reports — ORICEAN" }, { property: "og:description", content: "Assemble evidence packages and submit oil spill attribution reports for review." }, { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" }] }),
  component: gated("analyst", Reports),
});