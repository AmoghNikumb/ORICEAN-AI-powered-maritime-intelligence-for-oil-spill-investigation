import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { SatelliteAnalysis } from "@/components/oricean/AnalystPages";

export const Route = createFileRoute("/analyst/satellite")({
  head: () => ({ meta: [{ title: "Satellite Analysis — ORICEAN" }, { name: "description", content: "Review calibrated SAR scenes and oil slick detection evidence in ORICEAN." }, { property: "og:title", content: "Satellite Analysis — ORICEAN" }, { property: "og:description", content: "Review calibrated SAR scenes and oil slick detection evidence in ORICEAN." }, { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" }] }),
  component: gated("analyst", SatelliteAnalysis),
});