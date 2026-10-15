import { createFileRoute } from "@tanstack/react-router";
import { GlobalInsights } from "@/components/oricean/PublicPages";

export const Route = createFileRoute("/explore/insights")({
  head: () => ({
    meta: [
      { title: "Global Insights — ORICEAN" },
      { name: "description", content: "Global trends and statistics across published oil spill detections." },
      { property: "og:title", content: "Global Insights — ORICEAN" },
      { property: "og:description", content: "Global trends and statistics across published oil spill detections." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: GlobalInsights,
});
