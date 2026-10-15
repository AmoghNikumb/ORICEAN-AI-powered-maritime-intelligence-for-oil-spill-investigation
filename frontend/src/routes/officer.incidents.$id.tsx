import { createFileRoute } from "@tanstack/react-router";
import { gated } from "@/components/oricean/RoleGate";
import { IncidentDetail } from "@/components/oricean/OfficerPages";

function IncidentDetailRoute() {
  const { id } = Route.useParams();
  return <IncidentDetail investigationId={id} />;
}

export const Route = createFileRoute("/officer/incidents/$id")({
  head: () => ({
    meta: [
      { title: "Incident Detail — ORICEAN" },
      { name: "description", content: "Full officer view of a single oil-spill investigation." },
    ],
  }),
  component: gated("officer", IncidentDetailRoute),
});
