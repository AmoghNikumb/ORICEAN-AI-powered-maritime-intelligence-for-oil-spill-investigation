/**
 * Unit tests for pure helper functions exported from OriceanMapLibre.tsx
 *
 * Validates: Requirements 4.1, 4.2, 9.2, 9.3
 *
 * Note: maplibre-gl is mocked at the module level because the component file
 * imports it at the top, but none of the pure helper functions under test
 * actually call the maplibre runtime — it is only used as a TypeScript type.
 */

import { describe, it, expect, vi, beforeAll } from "vitest";

// ---------------------------------------------------------------------------
// Mock maplibre-gl before importing the component module.
// The pure helpers don't use maplibregl at runtime, but the module-level
// `import maplibregl from "maplibre-gl"` would fail in a jsdom environment
// without WebGL support. A minimal stub is enough.
// ---------------------------------------------------------------------------
vi.mock("maplibre-gl", () => {
  return {
    default: {
      Map: vi.fn(),
    },
  };
});

// Also mock the CSS import (vitest ignores unknown asset types by default,
// but the explicit mock makes the intent clear)
vi.mock("maplibre-gl/dist/maplibre-gl.css", () => ({}));

// Import after mocks are set up
import {
  getTileState,
  tileStateToColor,
  buildTileGeoJSON,
  reclassifyTiles,
} from "../OriceanMapLibre";
import type { TilePrediction } from "@/lib/api";

// ---------------------------------------------------------------------------
// Test fixture helpers
// ---------------------------------------------------------------------------

/** Build a minimal TilePrediction with sensible defaults. */
function makeTile(overrides: Partial<TilePrediction> = {}): TilePrediction {
  return {
    tile_id: "tile-001",
    label: "No Oil",
    confidence: 0.3,
    is_positive: false,
    bounds: [10, 20, 11, 21], // min_lon, min_lat, max_lon, max_lat
    acquisition_date: "2024-01-15",
    mock: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 1. getTileState
// ---------------------------------------------------------------------------

describe("getTileState", () => {
  it('returns "OIL_DETECTED" when is_positive=true and confidence >= threshold', () => {
    const tile = makeTile({ is_positive: true, confidence: 0.8 });
    expect(getTileState(tile, 0.5)).toBe("OIL_DETECTED");
  });

  it('returns "ANALYSED_NO_OIL" when is_positive=false regardless of confidence', () => {
    const tile = makeTile({ is_positive: false, confidence: 0.9 });
    expect(getTileState(tile, 0.5)).toBe("ANALYSED_NO_OIL");
  });

  it('returns "PENDING" when is_positive=true, confidence below threshold, and acquisition_date is empty', () => {
    // confidence(0.3) < threshold(0.5) → NOT OIL_DETECTED
    // is_positive=true → NOT ANALYSED_NO_OIL
    // acquisition_date="" → PENDING
    const tile = makeTile({ is_positive: true, confidence: 0.3, acquisition_date: "" });
    expect(getTileState(tile, 0.5)).toBe("PENDING");
  });

  it('returns "ERROR" when is_positive=true, confidence below threshold, and acquisition_date is present', () => {
    // is_positive=true but confidence(0.3) < threshold(0.5) → not OIL_DETECTED
    // is_positive=true → not ANALYSED_NO_OIL
    // acquisition_date is present → not PENDING
    // falls through to ERROR
    const tile = makeTile({ is_positive: true, confidence: 0.3, acquisition_date: "2024-01-15" });
    expect(getTileState(tile, 0.5)).toBe("ERROR");
  });

  it('returns "OIL_DETECTED" exactly at the threshold boundary (confidence === threshold)', () => {
    const tile = makeTile({ is_positive: true, confidence: 0.5 });
    expect(getTileState(tile, 0.5)).toBe("OIL_DETECTED");
  });
});

// ---------------------------------------------------------------------------
// 2. tileStateToColor
// ---------------------------------------------------------------------------

describe("tileStateToColor", () => {
  it('maps "PENDING" to the expected rgba grey string', () => {
    expect(tileStateToColor("PENDING")).toBe("rgba(128,128,128,0.3)");
  });

  it('maps "PROCESSING" to the satellite CSS variable fallback', () => {
    expect(tileStateToColor("PROCESSING")).toBe("var(--satellite, #3b82f6)");
  });

  it('maps "ANALYSED_NO_OIL" to the resolved CSS variable fallback', () => {
    expect(tileStateToColor("ANALYSED_NO_OIL")).toBe("var(--resolved, #22c55e)");
  });

  it('maps "OIL_DETECTED" to the spill CSS variable fallback', () => {
    expect(tileStateToColor("OIL_DETECTED")).toBe("var(--spill, #f97316)");
  });

  it('maps "ERROR" to the critical CSS variable fallback', () => {
    expect(tileStateToColor("ERROR")).toBe("var(--critical, #ef4444)");
  });

  it("returns a non-empty string for every valid TileState", () => {
    const states = ["PENDING", "PROCESSING", "ANALYSED_NO_OIL", "OIL_DETECTED", "ERROR"] as const;
    for (const state of states) {
      const color = tileStateToColor(state);
      expect(typeof color).toBe("string");
      expect(color.length).toBeGreaterThan(0);
    }
  });
});

// ---------------------------------------------------------------------------
// 3. buildTileGeoJSON
// ---------------------------------------------------------------------------

describe("buildTileGeoJSON", () => {
  const twoTiles: TilePrediction[] = [
    makeTile({
      tile_id: "tile-A",
      is_positive: true,
      confidence: 0.9,
      bounds: [10, 20, 11, 21],
    }),
    makeTile({
      tile_id: "tile-B",
      is_positive: false,
      confidence: 0.1,
      bounds: [12, 22, 13, 23],
    }),
  ];

  it("returns a FeatureCollection with feature count equal to the number of input tiles", () => {
    const result = buildTileGeoJSON(twoTiles, 0.5);
    expect(result.type).toBe("FeatureCollection");
    expect(result.features).toHaveLength(2);
  });

  it("each feature has a Polygon geometry", () => {
    const result = buildTileGeoJSON(twoTiles, 0.5);
    for (const feature of result.features) {
      expect(feature.geometry.type).toBe("Polygon");
    }
  });

  it("each feature's tile_id property matches the source tile", () => {
    const result = buildTileGeoJSON(twoTiles, 0.5);
    expect(result.features[0].properties?.tile_id).toBe("tile-A");
    expect(result.features[1].properties?.tile_id).toBe("tile-B");
  });

  it("polygon coordinates are derived from tile.bounds (min_lon, min_lat, max_lon, max_lat)", () => {
    const tile = makeTile({ bounds: [10, 20, 11, 21] });
    const result = buildTileGeoJSON([tile], 0.5);
    const coords = (result.features[0].geometry as GeoJSON.Polygon).coordinates[0];
    // Ring: SW → SE → NE → NW → SW (closed)
    expect(coords).toHaveLength(5); // closed ring has 5 points
    expect(coords[0]).toEqual([10, 20]); // min_lon, min_lat
    expect(coords[1]).toEqual([11, 20]); // max_lon, min_lat
    expect(coords[2]).toEqual([11, 21]); // max_lon, max_lat
    expect(coords[3]).toEqual([10, 21]); // min_lon, max_lat
    expect(coords[4]).toEqual([10, 20]); // closed
  });

  it("returns an empty FeatureCollection for an empty tile array", () => {
    const result = buildTileGeoJSON([], 0.5);
    expect(result.type).toBe("FeatureCollection");
    expect(result.features).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// 4. reclassifyTiles
// ---------------------------------------------------------------------------

describe("reclassifyTiles", () => {
  const sampleTiles: TilePrediction[] = [
    makeTile({ tile_id: "t1", confidence: 0.0, is_positive: false }),
    makeTile({ tile_id: "t2", confidence: 0.5, is_positive: false }),
    makeTile({ tile_id: "t3", confidence: 0.99, is_positive: true }),
  ];

  it("with threshold=0, classifies ALL tiles as positive (confidence >= 0 is always true)", () => {
    const result = reclassifyTiles(sampleTiles, 0);
    expect(result).toHaveLength(sampleTiles.length);
    for (const tile of result) {
      expect(tile.is_positive).toBe(true);
      expect(tile.label).toBe("Oil Spill");
    }
  });

  it("with threshold=1, classifies ALL tiles as negative when confidence < 1.0", () => {
    // All three tiles have confidence < 1.0, so none should be positive
    const result = reclassifyTiles(sampleTiles, 1);
    expect(result).toHaveLength(sampleTiles.length);
    for (const tile of result) {
      expect(tile.is_positive).toBe(false);
      expect(tile.label).toBe("No Oil");
    }
  });

  it("preserves the original confidence values unchanged", () => {
    const result = reclassifyTiles(sampleTiles, 0.5);
    for (let i = 0; i < sampleTiles.length; i++) {
      expect(result[i].confidence).toBe(sampleTiles[i].confidence);
    }
  });

  it("returns a new array and does not mutate the original", () => {
    const origPositive = sampleTiles.map((t) => t.is_positive);
    reclassifyTiles(sampleTiles, 0);
    // Original tiles should be unchanged
    for (let i = 0; i < sampleTiles.length; i++) {
      expect(sampleTiles[i].is_positive).toBe(origPositive[i]);
    }
  });

  it("sets label to 'Oil Spill' for positive tiles and 'No Oil' for negative tiles", () => {
    // threshold=0.5: t1(0.0)→neg, t2(0.5)→pos (>=0.5), t3(0.99)→pos
    const result = reclassifyTiles(sampleTiles, 0.5);
    expect(result[0].is_positive).toBe(false);
    expect(result[0].label).toBe("No Oil");
    expect(result[1].is_positive).toBe(true); // 0.5 >= 0.5
    expect(result[1].label).toBe("Oil Spill");
    expect(result[2].is_positive).toBe(true);
    expect(result[2].label).toBe("Oil Spill");
  });
});
