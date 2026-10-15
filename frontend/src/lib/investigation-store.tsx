/**
 * Investigation workflow store.
 *
 * Holds the "active investigation" context — the investigation currently
 * being worked on by the analyst.  Persists to sessionStorage so the state
 * survives page navigation without a full backend re-fetch.
 *
 * Components subscribe via useActiveInvestigation().
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { InferenceResult, InvestigationRead } from "./api";

const KEY = "oricean.active_investigation";

interface WorkflowState {
  /** The backend Investigation record */
  investigation: InvestigationRead | null;
  /** Latest inference result (also stored in investigation.ml_result) */
  inferenceResult: InferenceResult | null;
  /** True while the backend is processing */
  isProcessing: boolean;
  /** Last error string from the backend */
  error: string | null;
}

interface Ctx extends WorkflowState {
  setInvestigation: (inv: InvestigationRead | null) => void;
  setInferenceResult: (r: InferenceResult | null) => void;
  setProcessing: (v: boolean) => void;
  setError: (e: string | null) => void;
  clear: () => void;
}

const InvestigationContext = createContext<Ctx | null>(null);

const DEFAULT_STATE: WorkflowState = {
  investigation: null,
  inferenceResult: null,
  isProcessing: false,
  error: null,
};

function loadFromSession(): WorkflowState {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (raw) return JSON.parse(raw) as WorkflowState;
  } catch {
    /* ignore */
  }
  return DEFAULT_STATE;
}

export function InvestigationProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<WorkflowState>(loadFromSession);

  // Persist to sessionStorage on every state change
  const stateRef = useRef(state);
  stateRef.current = state;
  useEffect(() => {
    try {
      sessionStorage.setItem(KEY, JSON.stringify(state));
    } catch {
      /* storage unavailable */
    }
  }, [state]);

  const setInvestigation = useCallback((inv: InvestigationRead | null) => {
    setState((s) => ({ ...s, investigation: inv }));
  }, []);

  const setInferenceResult = useCallback((r: InferenceResult | null) => {
    setState((s) => ({ ...s, inferenceResult: r }));
  }, []);

  const setProcessing = useCallback((v: boolean) => {
    setState((s) => ({ ...s, isProcessing: v }));
  }, []);

  const setError = useCallback((e: string | null) => {
    setState((s) => ({ ...s, error: e }));
  }, []);

  const clear = useCallback(() => {
    setState(DEFAULT_STATE);
    try {
      sessionStorage.removeItem(KEY);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo<Ctx>(
    () => ({ ...state, setInvestigation, setInferenceResult, setProcessing, setError, clear }),
    [state, setInvestigation, setInferenceResult, setProcessing, setError, clear],
  );

  return (
    <InvestigationContext.Provider value={value}>
      {children}
    </InvestigationContext.Provider>
  );
}

export function useActiveInvestigation() {
  const ctx = useContext(InvestigationContext);
  if (!ctx) throw new Error("useActiveInvestigation must be used inside InvestigationProvider");
  return ctx;
}
