"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import { API_BASE, checkHealth } from "@/lib/api";

type Status = "checking" | "online" | "offline";

/**
 * Probes /api/health on mount and surfaces a friendly banner when the free-tier
 * backend is asleep/unreachable - so data pages explain the blank state instead
 * of silently failing. Hitting Retry also nudges the service awake.
 */
export default function BackendStatus() {
  const [status, setStatus] = useState<Status>("checking");

  const run = useCallback(async () => {
    setStatus("checking");
    setStatus((await checkHealth()) ? "online" : "offline");
  }, []);

  useEffect(() => {
    run();
  }, [run]);

  if (status === "online") return null;

  if (status === "checking") {
    return (
      <div className="mb-6 flex items-center gap-2 rounded-lg border border-line bg-surface/60 px-4 py-2 text-xs text-fg-muted">
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
        Connecting to the Fraud Detection API...
      </div>
    );
  }

  return (
    <div
      role="status"
      className="mb-6 flex flex-col gap-3 rounded-lg border border-warning/30 bg-warning/10 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="flex items-start gap-2 text-sm">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden />
        <span className="text-fg-muted">
          <span className="font-medium text-warning">API offline.</span> The free backend (
          <code className="nums text-fg-subtle">{API_BASE.replace(/^https?:\/\//, "")}</code>) sleeps
          after ~15 min and takes ~30s to wake. The Risk Scorecard needs it.
        </span>
      </div>
      <button
        onClick={run}
        className="focus-ring inline-flex shrink-0 cursor-pointer items-center gap-2 self-start rounded-lg border border-warning/40 px-3 py-1.5 text-xs font-medium text-warning transition hover:bg-warning/10 sm:self-auto"
      >
        <RefreshCw className="h-3.5 w-3.5" aria-hidden /> Retry
      </button>
    </div>
  );
}
