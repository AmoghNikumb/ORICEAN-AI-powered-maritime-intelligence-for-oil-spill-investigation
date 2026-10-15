import { gated } from "@/components/oricean/RoleGate";
import { createFileRoute } from "@tanstack/react-router";
import { CaseReview } from "@/components/oricean/OfficerPages";

export const Route = createFileRoute("/officer/case")({
  head: () => ({
    meta: [
      { title: "Case Review — ORICEAN" },
      { name: "description", content: "Validate analyst evidence and make handoff decisions for marine oil spill cases." },
      { property: "og:title", content: "Case Review — ORICEAN" },
      { property: "og:description", content: "Validate analyst evidence and make handoff decisions for marine oil spill cases." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: gated("officer", CaseReview),
});
