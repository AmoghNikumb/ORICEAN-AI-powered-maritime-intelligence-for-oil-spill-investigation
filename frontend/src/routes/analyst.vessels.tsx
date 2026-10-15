import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { VesselAttribution } from "@/components/oricean/AnalystPages";

export const Route = createFileRoute("/analyst/vessels")({
  head: () => ({ meta: [{ title: "Vessel Attribution — ORICEAN" }, { name: "description", content: "Correlate AIS vessel movement with probable oil spill source regions in ORICEAN." }, { property: "og:title", content: "Vessel Attribution — ORICEAN" }, { property: "og:description", content: "Correlate AIS vessel movement with probable oil spill source regions in ORICEAN." }, { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" }] }),
  component: gated("analyst", VesselAttribution),
});