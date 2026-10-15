/**
 * PipelineProgressPanel.tsx
 *
 * Shows live step-by-step pipeline progress while an investigation is running.
 * Polls investigationsApi.get() every 5 seconds via TanStack Query refetchInterval.
 * Maps backend status to PIPELINE_STEPS completion states.
 *
 * Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
 */

import { useQuery } from "@tanstack/react-query";
import { Check, AlertTriangle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { investigationsApi, type InvestigationStatus } from "@/lib/api";

// ---------------------------------------------------------------------------
// Pipeline step definitions (from design doc)
// ---------------------------------------------------------------------------

const PIPELINE_STEPS = [
  {
    id: "aoi",
    label: "AOI Submitted",
    doneWhen: ["running_gee", "running_ml", "complete"] as InvestigationStatus[],
  },
  {
    id: "scene",
    label: "Scene Retrieved",
    doneWhen: ["running_ml", "complete"] as InvestigationStatus[],
  },
  {
    id: "sar",
    label: "SAR Prepared",
    doneWhen: ["running_ml", "complete"] as InvestigationStatus[],
  },
  {
    id: "tiles",
    label: "Tile Processing",
    doneWhen: ["complete"] as InvestigationStatus[],
  },
  {
    id: "detection",
    label: "Detection Complete",
    doneWhen: ["complete"] as InvestigationStatus[],
  },
  {
    id: "source",
    label: "Source Reconstruction",
    doneWhen: [] as InvestigationStatus[], // future
  },
  {
    id: "ais",
    label: "AIS Correlation",
    doneWhen: [] as InvestigationStatus[], // future
  },
] as const;

// ---------------------------------------------------------------------------
// Determine which step is currently "in progress" given a status
// ---------------------------------------------------------------------------

function getActiveStepIndex(status: InvestigationStatus | string): number {
  switch (status) {
    case "pending":
      return 0;
    case "running_gee":
      return 1; // retrieving scene
    case "running_ml":
      return 3; // tile processing
    case "complete":
      return -1; // all done
    case "failed":
      return -2; // error state
    default:
      return 0;
  }
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface PipelineProgressPanelProps {
  investigationId: string;
  isVisible: boolean;
  className?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PipelineProgressPanel({
  investigationId,
  isVisible,
  className,
}: PipelineProgressPanelProps) {
  // Poll every 5 seconds while visible; disable when hidden or no id
  const { data: investigation, error } = useQuery({
    queryKey: ["investigation-progress", investigationId],
    queryFn: () => investigationsApi.get(investigationId),
    enabled: isVisible && Boolean(investigationId),
    refetchInterval: isVisible ? 5_000 : false,
    staleTime: 0,
  });

  if (!isVisible) return null;

  const status: InvestigationStatus = (investigation?.status as InvestigationStatus) ?? "pending";
  const isFailed = status === "failed";
  const isComplete = status === "complete";
  const activeIndex = getActiveStepIndex(status);

  return (
    <div
      className={cn(
        "rounded-md border border-white/15 bg-black/85 shadow-xl backdrop-blur-sm",
        className,
      )}
      role="status"
      aria-live="polite"
      aria-label="Pipeline progress"
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 border-b border-white/10 px-4 py-2.5">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-white/50">
          Pipeline Progress
        </p>
        {!isComplete && !isFailed && (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-white/40" />
        )}
        {isComplete && (
          <span className="text-[10px] font-semibold uppercase tracking-widest text-green-400">
            Complete
          </span>
        )}
        {isFailed && (
          <span className="text-[10px] font-semibold uppercase tracking-widest text-red-400">
            Failed
          </span>
        )}
      </div>

      {/* Steps list */}
      <div className="px-4 py-3">
        <ol className="space-y-2.5" aria-label="Pipeline steps">
          {PIPELINE_STEPS.map((step, index) => {
            const isDone = (step.doneWhen as readonly string[]).includes(status);
            const isActive = !isFailed && !isDone && index === activeIndex;
            const isFutureStep = step.doneWhen.length === 0 && !isComplete;

            // A step is in error if the pipeline failed and it was the active step
            const isErrorStep = isFailed && index === activeIndex;

            return (
              <li
                key={step.id}
                className={cn(
                  "flex items-center gap-3 text-xs",
                  isDone && "text-white/80",
                  isActive && "text-white",
                  isErrorStep && "text-red-400",
                  !isDone && !isActive && !isErrorStep && "text-white/35",
                )}
              >
                {/* Step icon */}
                <span
                  className={cn(
                    "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px]",
                    isDone && "border-green-500/50 bg-green-500/20 text-green-400",
                    isActive && "border-blue-400/50 bg-blue-400/15 text-blue-300",
                    isErrorStep && "border-red-500/50 bg-red-500/20 text-red-400",
                    !isDone &&
                      !isActive &&
                      !isErrorStep &&
                      "border-white/20 bg-white/5 text-white/30",
                  )}
                  aria-hidden="true"
                >
                  {isDone ? (
                    <Check className="h-3 w-3" />
                  ) : isActive ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : isErrorStep ? (
                    <AlertTriangle className="h-3 w-3" />
                  ) : (
                    <span className="font-mono">{index + 1}</span>
                  )}
                </span>

                {/* Step label */}
                <span className={cn("font-medium", isFutureStep && "text-white/25")}>
                  {step.label}
                  {isFutureStep && (
                    <span className="ml-1 text-[10px] font-normal text-white/20">(future)</span>
                  )}
                </span>
              </li>
            );
          })}
        </ol>

        {/* Error message */}
        {isFailed && investigation?.error_message && (
          <div className="mt-3 flex items-start gap-2 rounded border border-red-500/25 bg-red-500/10 px-3 py-2 text-xs text-red-300">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            <span>{investigation.error_message}</span>
          </div>
        )}

        {/* Polling error fallback */}
        {error && !investigation && (
          <div className="mt-3 text-[11px] text-white/30">
            Could not fetch status — retrying…
          </div>
        )}
      </div>
    </div>
  );
}

export default PipelineProgressPanel;
