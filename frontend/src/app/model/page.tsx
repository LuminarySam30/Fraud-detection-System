"use client";

/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useState } from "react";
import { Activity, BarChart3, Boxes, Gauge, Info, X } from "lucide-react";
import Plot from "@/components/Plot";
import { apiFetch, ApiError } from "@/lib/api";
import { ErrorState, PageHeader, Panel, Skeleton, StatCard } from "@/components/ui";

interface Meta {
  threshold: number;
  n_features: number;
  features: string[];
  categories: string[];
  feature_importance: Record<string, number>;
}

// Reported by the training notebook on the held-out test set. Labelled as such
// in the UI - these are not recomputed live (that needs the Kaggle dataset).
const REPORTED = [
  { label: "ROC-AUC", value: "0.9967" },
  { label: "Recall (fraud caught)", value: "88.0%" },
  { label: "Training rows", value: "1.30M" },
  { label: "Test rows", value: "555,719" },
  { label: "Fraud rate", value: "0.58%" },
  { label: "Trees", value: "500" },
];

export default function ModelPage() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setError(null);
    setMeta(null);
    try {
      setMeta(await apiFetch<Meta>("/api/metadata"));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load model metadata.");
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (error) {
    return (
      <div>
        <PageHeader eyebrow="Insights" title="Model" />
        <ErrorState message={error} onRetry={load} icon={X} />
      </div>
    );
  }

  if (!meta) {
    return (
      <div>
        <PageHeader eyebrow="Insights" title="Model" />
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-28" />)}</div>
        <Skeleton className="mt-6 h-[460px]" />
      </div>
    );
  }

  const ranked = Object.entries(meta.feature_importance)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 15);
  const importancePlot: any[] = [{
    type: "bar",
    orientation: "h",
    x: ranked.map(([, v]) => v).reverse(),
    y: ranked.map(([k]) => k.replace(/_/g, " ")).reverse(),
    marker: { color: "#10b981" },
    hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
  }];

  return (
    <div>
      <PageHeader eyebrow="Insights" title="Model" description="XGBoost gradient-boosted trees. Feature importances below are read live from the deployed model - not hardcoded." />

      <div className="mb-6 grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
        <StatCard label="Algorithm" value="XGBoost" icon={Boxes} />
        <StatCard label="Features" value={`${meta.n_features}`} icon={BarChart3} tone="accent" />
        <StatCard label="Decision threshold" value={`${(meta.threshold * 100).toFixed(1)}%`} icon={Gauge} tone="success" />
        <StatCard label="Categories" value={`${meta.categories.length}`} icon={Activity} />
      </div>

      <Panel className="mb-6 flex h-[520px] flex-col p-6">
        <h3 className="text-lg font-bold text-fg">Feature importance (top 15)</h3>
        <p className="mb-4 text-xs text-fg-muted">Gain-based importance straight from the model. Amount and amount-deviation dominate, followed by category one-hots.</p>
        <div className="min-h-0 flex-1"><Plot data={importancePlot} layout={{ margin: { l: 150, r: 16, t: 8, b: 32 } }} /></div>
      </Panel>

      <Panel className="p-6">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-lg font-bold text-fg">Reported test performance</h3>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {REPORTED.map((m) => (
            <div key={m.label} className="rounded-lg bg-white/5 p-4">
              <div className="nums text-xl font-semibold text-fg">{m.value}</div>
              <div className="mt-1 text-xs text-fg-muted">{m.label}</div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-line bg-white/5 p-3 text-xs text-fg-muted">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-fg-subtle" aria-hidden />
          <span>
            These figures are <span className="text-fg">as reported by the training notebook</span> on a held-out test set. A full
            precision/PR re-evaluation against the source dataset is on the roadmap - until then they're shown for context, not as live metrics.
          </span>
        </div>
      </Panel>
    </div>
  );
}
