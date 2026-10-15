import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { CommandDashboard } from "@/components/oricean/OfficerPages";

export const Route = createFileRoute("/officer/")({
  head: () => ({
    meta: [
      { title: "Command Dashboard — ORICEAN" },
      { name: "description", content: "National maritime pollution monitoring and response coordination for monitoring officers." },
      { property: "og:title", content: "Command Dashboard — ORICEAN" },
      { property: "og:description", content: "National maritime pollution monitoring and response coordination for monitoring officers." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: gated("officer", CommandDashboard),
});
