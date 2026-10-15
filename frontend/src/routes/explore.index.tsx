import { createFileRoute } from "@tanstack/react-router";
import { GlobalMap } from "@/components/oricean/PublicPages";

export const Route = createFileRoute("/explore/")({
  head: () => ({
    meta: [
      { title: "ORICEAN Global Spill Map" },
      { name: "description", content: "Explore publicly released marine oil spill detections worldwide on an interactive map." },
      { property: "og:title", content: "ORICEAN Global Spill Map" },
      { property: "og:description", content: "Explore publicly released marine oil spill detections worldwide on an interactive map." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: GlobalMap,
});
