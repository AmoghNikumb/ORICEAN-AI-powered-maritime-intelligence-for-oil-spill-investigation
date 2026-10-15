import { createFileRoute } from "@tanstack/react-router";
import { HistoricalSpills } from "@/components/oricean/PublicPages";

export const Route = createFileRoute("/explore/historical")({
  head: () => ({
    meta: [
      { title: "Historical Spills — ORICEAN" },
      { name: "description", content: "Archive of released marine oil spill detections and closed incidents." },
      { property: "og:title", content: "Historical Spills — ORICEAN" },
      { property: "og:description", content: "Archive of released marine oil spill detections and closed incidents." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: HistoricalSpills,
});
