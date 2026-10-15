import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { SourceReconstruction } from "@/components/oricean/AnalystPages";

export const Route = createFileRoute("/analyst/source")({
  head: () => ({ meta: [{ title: "Source Reconstruction — ORICEAN" }, { name: "description", content: "Backtrack oil spill drift and reconstruct probable release regions with ocean-current models." }, { property: "og:title", content: "Source Reconstruction — ORICEAN" }, { property: "og:description", content: "Backtrack oil spill drift and reconstruct probable release regions with ocean-current models." }, { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" }] }),
  component: gated("analyst", SourceReconstruction),
});