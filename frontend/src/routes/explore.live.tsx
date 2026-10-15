import { createFileRoute } from "@tanstack/react-router";
import { LiveIncidents } from "@/components/oricean/PublicPages";

export const Route = createFileRoute("/explore/live")({
  head: () => ({
    meta: [
      { title: "Live Incidents — ORICEAN" },
      { name: "description", content: "Ongoing marine pollution events currently being tracked by ORICEAN." },
      { property: "og:title", content: "Live Incidents — ORICEAN" },
      { property: "og:description", content: "Ongoing marine pollution events currently being tracked by ORICEAN." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: LiveIncidents,
});
