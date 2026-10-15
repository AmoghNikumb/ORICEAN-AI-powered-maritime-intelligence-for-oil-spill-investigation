/**
 * Unit tests for PipelineProgressPanel step transitions.
 *
 * Validates: Requirements 6.4, 6.7
 *
 * Tests:
 * 1. "Detection Complete" step shows a Check icon when status is "complete"
 * 2. A Loader2 spinner is rendered for the active step when status is "running_ml"
 * 3. Component returns null when isVisible=false
 * 4. All 7 pipeline steps render when status is "pending"
 */

import { vi, describe, it, expect, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { PipelineProgressPanel } from "../PipelineProgressPanel";

// ---------------------------------------------------------------------------
// Mock @/lib/api — replace investigationsApi.get with a controllable vi.fn()
// ---------------------------------------------------------------------------

vi.mock("@/lib/api", () => ({
  investigationsApi: {
    get: vi.fn(),
  },
}));

// Import the mocked module AFTER vi.mock so we get the mock reference
import { investigationsApi } from "@/lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal InvestigationRead-shaped object with the given status. */
function makeInvestigation(status: string) {
  return {
    id: "test-id-123",
    name: "Test Investigation",
    aoi_geojson: {},
    start_date: "2024-01-01",
    end_date: "2024-01-07",
    status,
    ml_result: null,
    ml_run_at: null,
    error_message: null,
    created_at: null,
    updated_at: null,
  };
}

/** Wrap the component in a fresh QueryClient to isolate each test's cache. */
function renderWithQuery(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        // No retries so errors surface immediately
        retry: false,
        // Default staleTime=0 so the query always fetches
        staleTime: 0,
      },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

/**
 * Find the icon <span> (aria-hidden="true") within a given list item.
 * The icon span is the first child of the <li>.
 */
function getIconSpanInLi(li: Element): Element | null {
  return li.querySelector("span[aria-hidden='true']");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("PipelineProgressPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows a checkmark (Check icon) on "Detection Complete" when status is "complete"', async () => {
    // Arrange: mock API to return status "complete"
    vi.mocked(investigationsApi.get).mockResolvedValue(
      makeInvestigation("complete") as never,
    );

    renderWithQuery(
      <PipelineProgressPanel investigationId="test-id-123" isVisible={true} />,
    );

    // Wait for the API to resolve and the step to show the green "done" styling.
    // When status="complete", the "Detection Complete" step has isDone=true,
    // so its icon span gains class "bg-green-500/20" and renders a <Check /> SVG.
    const stepsList = await screen.findByRole("list", { name: "Pipeline steps" });

    await waitFor(() => {
      const detectionLi = Array.from(stepsList.querySelectorAll("li")).find(
        (li) => li.textContent?.includes("Detection Complete"),
      );
      expect(detectionLi).toBeTruthy();

      const iconSpan = getIconSpanInLi(detectionLi!);
      expect(iconSpan).toBeTruthy();

      // The done step icon span has the green background class
      expect(iconSpan!.className).toMatch(/bg-green-500/);
    });

    // At this point the step is confirmed done — now verify the SVG (Check icon)
    const detectionLi = Array.from(stepsList.querySelectorAll("li")).find(
      (li) => li.textContent?.includes("Detection Complete"),
    )!;
    const iconSpan = getIconSpanInLi(detectionLi)!;
    const svg = iconSpan.querySelector("svg");

    expect(svg).toBeTruthy();
    const svgClass = svg!.getAttribute("class") ?? "";
    // animate-spin must NOT be present on a done step's icon (it's a Check, not a Loader)
    expect(svgClass).not.toMatch(/animate-spin/);
    // The lucide Check icon has class "lucide-check" or "lucide lucide-check"
    expect(svgClass).toMatch(/lucide-check/);
  });

  it('shows a spinner (Loader2 with animate-spin) on the active step when status is "running_ml"', async () => {
    // Arrange: mock API to return status "running_ml"
    // Active step index for "running_ml" is 3 → "Tile Processing"
    vi.mocked(investigationsApi.get).mockResolvedValue(
      makeInvestigation("running_ml") as never,
    );

    renderWithQuery(
      <PipelineProgressPanel investigationId="test-id-123" isVisible={true} />,
    );

    const stepsList = await screen.findByRole("list", { name: "Pipeline steps" });

    // Wait for the active step styling to appear on "Tile Processing"
    // When status="running_ml", index=3 ("Tile Processing") is the active step,
    // so its icon span gains class "bg-blue-400/15" and renders a <Loader2 animate-spin> SVG.
    await waitFor(() => {
      const tileProcessingLi = Array.from(stepsList.querySelectorAll("li")).find(
        (li) => li.textContent?.includes("Tile Processing"),
      );
      expect(tileProcessingLi).toBeTruthy();

      const iconSpan = getIconSpanInLi(tileProcessingLi!);
      expect(iconSpan).toBeTruthy();

      // The active step icon span has the blue background class
      expect(iconSpan!.className).toMatch(/bg-blue-400/);
    });

    // Verify the SVG is a Loader2 with animate-spin
    const tileProcessingLi = Array.from(stepsList.querySelectorAll("li")).find(
      (li) => li.textContent?.includes("Tile Processing"),
    )!;
    const iconSpan = getIconSpanInLi(tileProcessingLi)!;
    const svg = iconSpan.querySelector("svg");

    expect(svg).toBeTruthy();
    const svgClass = svg!.getAttribute("class") ?? "";
    // animate-spin should be present on the active step's Loader2 icon
    expect(svgClass).toMatch(/animate-spin/);
  });

  it("returns null (renders nothing) when isVisible is false", () => {
    vi.mocked(investigationsApi.get).mockResolvedValue(
      makeInvestigation("running_ml") as never,
    );

    const { container } = renderWithQuery(
      <PipelineProgressPanel investigationId="test-id-123" isVisible={false} />,
    );

    // Component should render nothing at all
    expect(container.firstChild).toBeNull();
  });

  it('renders all 7 pipeline steps when status is "pending"', async () => {
    vi.mocked(investigationsApi.get).mockResolvedValue(
      makeInvestigation("pending") as never,
    );

    renderWithQuery(
      <PipelineProgressPanel investigationId="test-id-123" isVisible={true} />,
    );

    // The 7 step labels are always rendered — the list is present immediately
    const stepsList = await screen.findByRole("list", { name: "Pipeline steps" });
    const steps = stepsList.querySelectorAll("li");
    expect(steps).toHaveLength(7);
  });
});
